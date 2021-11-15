# -*- coding: utf-8 -*-
# Time    : 2020/2/10 10:03
# Author  : Yichen Lu

from math import inf
import torch
from reid.models.backbone.resnet import ResNet

# step function should be called before picking off epoch training.


class CombinedLRSchduler(object):
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def step(self, epoch):
        for scheduler in self.schedulers:
            scheduler.step(epoch)
        current_lrs = [scheduler.get_lr() for scheduler in self.schedulers]

        return current_lrs

    def get_lr(self):
        return [scheduler.get_lr() for scheduler in self.schedulers]

    def set_lr(self, lr):
        for scheduler in self.schedulers:
            scheduler.set_lr(lr)


class WarmupLRScheduler(object):
    def __init__(self, optimizer, warmup_epochs=10, base_lr=1e-2, milestones=(inf, ), start_epoch=1):
        self.start_epoch = start_epoch
        self.last_epoch = self.start_epoch
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.milestones = milestones
        assert self.milestones[0] > self.warmup_epochs, "First milestone epoch should be greater than warmup-epochs."

    def step(self, epoch):
        self.last_epoch = epoch
        current_lr = self.get_lr()

        for g in self.optimizer.param_groups:
            g['lr'] = current_lr * g.get('lr_mult', 1)

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            current_lr = self.base_lr * (self.last_epoch - self.start_epoch + 1) / self.warmup_epochs
        else:
            current_lr = self.base_lr
            for milestone in self.milestones:
                if self.last_epoch >= milestone:
                    current_lr *= 0.1
        return current_lr


if __name__ == '__main__':
    model = ResNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = WarmupLRScheduler(optimizer, milestones=(30, 50))
    for epoch in range(1, 61):
        scheduler.step(epoch)
        print(f"cuurent learning rate: {scheduler.get_lr()}")
