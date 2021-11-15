import os
import sys
import argparse
import os.path as osp
from copy import deepcopy

import torch
from torch import nn as nn
from torch.backends import cudnn
from torchvision import transforms as T

from reid.utils.data.dataset import Dataset, InversionDataset
from reid.utils.data import build_train_loader
from reid.utils.serialization import load_checkpoint, CheckpointManager
from reid.models import Linear, ResNet


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu)
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Dataset(args.data_root, args.dataset)

    # Inversion Dataset
    inversion_dataset = InversionDataset('./generations')
    training_loader_old = build_train_loader(inversion_dataset, args, metric=False)

    # Load from checkpoint
    epoch = 1
    assert osp.isdir(args.previous) or osp.isfile(args.previous), 'previous checkpoint path not exists or not a folder'
    ckpt_p = osp.join(args.previous, 'checkpoint.pth.tar') if osp.isdir(args.previous) else args.previous
    print(f"Loading previous checkpoint from {ckpt_p}")
    checkpoint = load_checkpoint(ckpt_p)
    num_previous_classes = checkpoint['classifier']['W'].size(0)

    model_old = ResNet(last_stride=args.last_stride, last_pooling=args.last_pooling, embedding=args.embedding)
    classifier_old = Linear(args.embedding, num_previous_classes, device)
    manager = CheckpointManager(model=model_old, classifier=classifier_old)
    manager.load(checkpoint)

    model_old = nn.DataParallel(model_old).to(device)
    model_old.eval()
    classifier_old = classifier_old.to(device)
    classifier_old.eval()

    for inputs in training_loader_old:
        def _parse_data(inputs):
            imgs, _, pids, _, *_ = inputs
            inputs = imgs.to(device)
            pids = pids.to(device)
            return inputs, pids

        inputs, pids = _parse_data(inputs)
        outputs, = model_old(inputs)
        preds = classifier_old(outputs)
        print(torch.mean((torch.argmax(preds, dim=1) == pids).float()))


if __name__ == '__main__':
    working_dir = osp.dirname(osp.abspath(__file__))

    parser = argparse.ArgumentParser(description="Incremental learning for person Re-ID")

    # basic configs
    parser.add_argument("-g", "--gpu", nargs='*', type=str, default=['0', '1'])
    parser.add_argument('--data-root', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--dataset', type=str, default="market")
    parser.add_argument('-b', '--batch-size', type=int, default=64)

    parser.add_argument("--previous", type=str)
    # data configs
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--re', type=float, default=0.0)
    parser.add_argument("--re-area", type=float, default=0.4)
    # model configs
    parser.add_argument("--last-pooling", type=str, default="avg", choices=["avg", "max"])
    parser.add_argument("--last-stride", type=int, default=1, choices=[1, 2])
    parser.add_argument("--embedding", type=int, default=2048)
    args = parser.parse_args()

    args.previous = './logs/baselines/msmt17'
    main(args)


