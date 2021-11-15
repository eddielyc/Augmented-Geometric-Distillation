# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2020 paper
# Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion
# Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek
# Hoiem, Niraj K. Jha, and Jan Kautz
# --------------------------------------------------------

import os
import sys
import argparse
from easydict import EasyDict as edict
from pathlib import Path
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.parallel
import torch.utils.data

from deep_inversion.deepinversion import DeepInversionClass
from deep_inversion.model import ResNet
from reid.models import Linear, ResNet as FeatureExtractor
from reid.utils.data.transforms import RandomErasingMask
from reid.utils.logging import Logger


def validate_one(input, target, model):
    """Perform validation on the validation set"""

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.generation_dir):
        os.mkdir(args.generation_dir)
        print(f"Make dir {args.generation_dir}")

    sys.stdout = Logger(os.path.join(args.generation_dir, 'log.txt'))
    print(args)

    coefficients = edict({})
    coefficients.r_feature = args.r_feature
    coefficients.first_bn_multiplier = args.first_bn_multiplier
    coefficients.tv_l1 = args.tv_l1
    coefficients.tv_l2 = args.tv_l2
    coefficients.l2 = args.l2
    coefficients.lr = args.lr
    coefficients.main_loss_multiplier = args.main_loss_multiplier
    coefficients.adi_scale = args.adi_scale

    hook_for_display = None

    # preprocessor = RandomErasingMask(0.5)
    preprocessor = None

    ckpt_p = os.path.join(args.teacher, 'checkpoint.pth.tar') if os.path.isdir(args.teacher) else args.teacher
    checkpoint = torch.load(ckpt_p)
    num_classes = checkpoint['classifier']['W'].size(0)

    feature_extractor = FeatureExtractor(depth=args.depth, last_stride=2, last_pooling='avg', embedding=args.embedding)
    classifier = Linear(args.embedding, num_classes, device)
    feature_extractor.load_state_dict(checkpoint['backbone'])
    classifier.load_state_dict(checkpoint['classifier'])
    net = ResNet(feature_extractor, classifier)
    net = net.to(device)
    net.eval()

    net_s = None
    if args.student is not None:

        feature_extractor_s = FeatureExtractor(depth=34, last_stride=2, last_pooling='avg', embedding=512)
        classifier_s = Linear(512, 1041+751, device)
        # classifier_s = Linear(512, 1041, device)

        ckpt_p = os.path.join(args.student, 'checkpoint.pth.tar') if os.path.isdir(args.student) else args.student
        checkpoint = torch.load(ckpt_p)
        feature_extractor_s.load_state_dict(checkpoint['backbone'])
        classifier_s.load_state_dict(checkpoint['classifier'])
        net_s = ResNet(feature_extractor_s, classifier_s)
        net_s = net_s.to(device)
        net_s.eval()

    deep_inversion_engine = DeepInversionClass(net_teacher=net,
                                               net_student=net_s,
                                               bs=args.bs,
                                               coefficients=coefficients,
                                               hook_for_display=hook_for_display)

    bags = {i: 0 for i in range(num_classes)}
    if args.resume:
        assert os.path.isdir(args.resume), "Invalid resume dir."
        fpaths = Path(args.resume).glob("*.jpg")
        for fpath in fpaths:
            identity = int(fpath.stem.split("_")[0])
            bags[identity] += 1
            if bags[identity] >= args.shots:
                bags.pop(identity)

    while bags and deep_inversion_engine.num_generations < args.iters:
        targets = torch.tensor(np.random.choice(list(bags.keys()), args.bs, replace=len(bags) < args.bs),
                               dtype=torch.long).to('cuda')

        generateds = deep_inversion_engine.generate_batch(targets=targets, preprocessor=preprocessor)

        for target, generated in zip(targets, generateds):
            identity = int(target)
            if identity in bags:
                image_np = generated.detach().cpu().numpy().transpose((1, 2, 0))
                pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
                digest = hash(pil_image.tobytes())

                place_to_store = f'{args.generation_dir}/{identity}_{digest}.jpg'
                pil_image.save(place_to_store)

                bags[identity] += 1
                if bags[identity] >= args.shots:
                    bags.pop(identity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", nargs='*', type=str, default=['0', '1'])
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--generation-dir', type=str, default='./generations')
    parser.add_argument('--shots', type=int, default=10)
    parser.add_argument('--iters', type=int, default=320)
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--depth', type=int, default=50)
    parser.add_argument('--embedding', type=int, default=2048)
    parser.add_argument('--teacher', type=str, default='./logs/r34/baselines/msmt17/checkpoint.pth.tar')
    parser.add_argument('--student', type=str, default=None)

    parser.add_argument('--adi_scale', type=float, default=0.0, help='Coefficient for Adaptive Deep Inversion')

    parser.add_argument('--verifier', action='store_true', help='evaluate batch with another model')
    parser.add_argument('--verifier_arch', type=str, default='mobilenet_v2',
                        help="arch name from torchvision models to act as a verifier")

    parser.add_argument('--r_feature', type=float, default=0.01,
                        help='coefficient for feature distribution regularization')
    parser.add_argument('--first_bn_multiplier', type=float, default=10.,
                        help='additional multiplier on first bn layer of R_feature')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.0001, help='coefficient for total variation L2 loss')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
    parser.add_argument('--l2', type=float, default=0.00001, help='l2 loss on the image')
    parser.add_argument('--main_loss_multiplier', type=float, default=1.0,
                        help='coefficient for the main loss in optimization')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    run(args)


if __name__ == '__main__':
    main()
