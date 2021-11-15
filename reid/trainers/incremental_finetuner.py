# -*- coding: utf-8 -*-
# Time    : 2021/8/9 15:41
# Author  : Yichen Lu


import torch

from reid.loss import base
from reid.trainers import Trainer
from reid.utils import AverageMeters, AverageMeter


class IncrementalFinetuner(Trainer):
    def __init__(self,
                 networks,
                 optimizer,
                 lr_scheduler,
                 ):

        super(IncrementalFinetuner, self).__init__()
        self.networks = networks
        self.trainables = [self.networks]
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.meters = AverageMeters(AverageMeter("Batch Time"),
                                    AverageMeter("Pid Loss"),
                                    AverageMeter("Triplet Loss"),
                                    )

    def train(self, epoch, training_loader):
        self.before_train(epoch)

        for i, inputs in enumerate(training_loader):
            inputs, pids = self._parse_data(inputs)
            losses = self.train_step(inputs, pids)

            self.meters.update([self.timer(),
                                *losses,
                                ])
            print(f"Epoch: [{epoch}][{i + 1}/{len(training_loader)}], " + self.meters())

        self.after_train()

    def before_train(self, epoch):
        super(IncrementalFinetuner, self).before_train()
        self.lr_scheduler.step(epoch)

    def train_step(self, inputs, pids):
        self.optimizer.zero_grad()
        outputs = self.networks(inputs)
        loss, losses = self._compute_loss(outputs, pids)
        loss.backward()
        self.optimizer.step()
        return losses

    def _compute_loss(self, outputs, pids):
        pooled, preds = outputs["global"], outputs["preds"]
        pid_loss, triplet_loss = self.basic_criterion(pooled, preds, pids)

        loss = pid_loss + triplet_loss
        return loss, [pid_loss.item(), triplet_loss.item()]
