# -*- coding: utf-8 -*-
# Time    : 2019/10/21 19:45
# Author  : Yichen Lu


import os
import sys
from os import path as osp
import time
import random
import yaml

import numpy as np
import torch

from .zipper import Zipper
from .logging import Logger


class DivisibleDict(dict):
    def __init__(self, mapping):
        super(DivisibleDict, self).__init__(mapping)

    def divide(self, sections):
        def recursive_divide(obj, sections):

            if isinstance(obj, torch.Tensor):
                return obj.split(obj.size(0) // sections)
            elif isinstance(obj, (list, tuple)):
                sub_rets = [[] for _ in range(sections)]
                for sub_obj in obj:
                    for sub_ret, splitted in zip(sub_rets, recursive_divide(sub_obj, sections)):
                        sub_ret.append(splitted)
                return sub_rets

        assert isinstance(sections, int)

        subdicts = [DivisibleDict({}) for _ in range(sections)]
        for key, value in self.items():
            for subdict, splitted in zip(subdicts, recursive_divide(value, sections)):
                subdict[key] = splitted
        return subdicts

    def detach(self):
        def recursive_detach(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach()
            elif isinstance(obj, (list, tuple)):
                return [recursive_detach(sub_obj) for sub_obj in obj]
        for key, value in self.items():
            self[key] = recursive_detach(value)
        return self


class Timer(object):
    def __init__(self):
        self._last = time.time()
        self._now = time.time()

    def __call__(self):
        self._last, self._now = self._now, time.time()
        return self._now - self._last

    def reset(self):
        self._last = time.time()
        self._now = time.time()


def before_run(args):
    # Set log file
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'), args.resume)

    if hasattr(args, "algo_config") and args.algo_config:
        assert osp.isfile(args.algo_config), f"Algorithm config file {args.algo_config} does not exist."
        with open(args.algo_config, 'r') as file:
            algo_config = yaml.load(file, yaml.Loader)
        args.algo_config = algo_config

    print(args)

    # Reproducibility
    if args.seed:
        set_seed(args)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Running with reproducibility.")
    else:
        torch.backends.cudnn.benchmark = True
        print("Running without reproducibility.")

    # Zip the codes
    zipper = Zipper(root='.', zippath=osp.join(args.logs_dir, 'Red.zip'))
    zipper.zip()

    # Set CUDA env
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def build_optimizer(backbone, classifier, args):
    if isinstance(backbone, torch.nn.DataParallel):
        backbone = backbone.module

    # Optimizer for feature extractor
    base_param_ids = set(map(id, backbone.base.parameters()))
    new_params = [p for p in backbone.parameters() if id(p) not in base_param_ids]

    param_groups = [
        {'params': backbone.base.parameters(), 'lr_mult': 1.},
        {'params': new_params, 'lr_mult': 10.},
        {'params': classifier.parameters(), 'lr_mult': 10.}
    ]

    # Optimizer for feature extractor
    if args.optimizer == 'Adam':
        optimizer_main = torch.optim.Adam(param_groups, lr=args.learning_rate)
        print("Build Adam optimizer for backbone.")
    elif args.optimizer == 'SGD':
        optimizer_main = torch.optim.SGD(param_groups, lr=args.learning_rate, momentum=0.9, weight_decay=5e-4,
                                         nesterov=True)
        print("Build SGD optimizer with momentum: 0.9 for backbone.")
    else:
        raise RuntimeError

    return optimizer_main
