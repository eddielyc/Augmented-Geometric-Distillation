# -*- coding: utf-8 -*-
# Time    : 2021/1/31 22:16
# Author  : Yichen Lu


import argparse
import os.path as osp
from copy import deepcopy

import torch

from reid.utils.data.dataset import Dataset, InversionDataset
from reid.utils.data import build_test_loader, build_train_loader
from reid import trainers
from reid.evaluation.evaluators import Evaluator
from reid.utils.serialization import load_checkpoint, CheckpointManager
from reid.utils.lr_schedulers import WarmupLRScheduler
from reid.models import ResNet, Linear, Networks
from reid.utils import before_run, build_optimizer


def main(args):
    before_run(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Dataset(args.data_root, args.dataset)
    training_loader = build_train_loader(dataset, args)
    query_loader, gallery_loader = build_test_loader(dataset, args)

    # Load from checkpoint
    epoch = 1
    checkpoint = load_checkpoint(args.previous)
    num_previous_classes = checkpoint['classifier']['W'].size(0)

    backbone_prec = ResNet(depth=args.depth, last_stride=args.last_stride, last_pooling=args.last_pooling, embedding=args.embedding)
    classifier_prec = Linear(args.embedding, num_previous_classes, device)
    manager = CheckpointManager(backbone=backbone_prec, classifier=classifier_prec)
    manager.load(checkpoint)

    backbone = deepcopy(backbone_prec)
    classifier = deepcopy(classifier_prec)
    classifier.expand(len(dataset.train_ids))

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        manager = CheckpointManager(backbone=backbone, classifier=classifier)
        epoch = manager.load(checkpoint)

        print("=> Start epoch {} ".format(epoch))

    backbone = backbone.to(device)
    classifier = classifier.to(device)
    networks = Networks(backbone, classifier)

    # Evaluator
    evaluator = Evaluator(backbone)

    # Checkpoint Manager
    manager = CheckpointManager(logs_dir=args.logs_dir,
                                backbone=backbone, classifier=classifier)

    # Optimizer
    optimizer_main = build_optimizer(backbone, classifier, args)

    # Lr Scheduler
    lr_scheduler = WarmupLRScheduler(optimizer_main, warmup_epochs=args.warmup, base_lr=args.learning_rate,
                                     milestones=args.epochs_decay, start_epoch=epoch)

    # Trainer
    trainer = trainers.IncrementalFinetuner(networks,
                                            optimizer_main,
                                            lr_scheduler
                                            )

    # ------------------- Training -------------------

    for epoch in range(epoch, args.epochs + 1):
        trainer.train(epoch, training_loader)

        if args.evaluate and epoch % args.evaluate == 0:
            evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, re_ranking=False,
                               output_feature='embedding', print_freq=1000)

        manager.save(epoch=epoch, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        if epoch % args.save_freq == 0:
            manager.save(epoch=epoch)

        print(f"\n * Finished epoch {epoch} learning rate {lr_scheduler.get_lr()} \n")

    # ------------------- Training -------------------


if __name__ == '__main__':
    working_dir = osp.dirname(osp.abspath(__file__))

    parser = argparse.ArgumentParser(description="Incremental learning for person Re-ID")

    # basic configs
    parser.add_argument("-g", "--gpu", nargs='*', type=str, default=['0'])
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument('--data-root', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--dataset', type=str, default="market")
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument("--epochs-decay", nargs='*', type=int, default=[41, ])
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'], default="SGD")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument("--evaluate", type=int, default=10)
    parser.add_argument("--previous", type=str, default='./logs/r50/baselines/msmt17')
    # data configs
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--re', type=float, default=0.5)
    parser.add_argument("--re-area", type=float, default=0.4)
    parser.add_argument("--preload", action="store_true", default=True)
    # backbone configs
    parser.add_argument("--last-pooling", type=str, default="avg", choices=["avg", "max"])
    parser.add_argument("--last-stride", type=int, default=2, choices=[1, 2])
    parser.add_argument("--depth", type=int, default=50, choices=[34, 50])
    parser.add_argument("--embedding", type=int, default=2048)

    args = parser.parse_args()

    main(args)

