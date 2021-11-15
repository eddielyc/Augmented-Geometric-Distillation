# -*- coding: utf-8 -*-
# Time    : 2021/8/7 17:26
# Author  : Yichen Lu
import torch

from reid.trainers import IncrementalTrainer
from reid.utils import AverageMeters, AverageMeter


class EWCTrainer(IncrementalTrainer):
    def __init__(self,
                 networks_prec,
                 networks,
                 incremental_classes,
                 optimizer,
                 lr_scheduler,
                 algo_config=None,
                 ):

        super(EWCTrainer, self).__init__(networks_prec,
                                         networks,
                                         incremental_classes,
                                         optimizer,
                                         lr_scheduler,
                                         algo_config
                                         )

        self.weight = 1e4
        self.fisher = {}

        self.meters = AverageMeters(AverageMeter("Batch Time"),
                                    AverageMeter("Pid Loss"),
                                    AverageMeter("Triplet Loss"),
                                    AverageMeter("Consolidation Loss"),
                                    )

    def train(self, epoch, training_loader):
        if not self.fisher:
            self.fisher = self.fisher_matrix_diag(training_loader)

        self.before_train(epoch)

        for i, inputs in enumerate(training_loader):

            inputs, pids = self._parse_data(inputs)
            losses = self.train_step(inputs, pids)

            self.meters.update([self.timer(),
                                *losses,
                                ])
            print(f"Epoch: [{epoch}][{i + 1}/{len(training_loader)}], " + self.meters())

    def fisher_matrix_diag(self, train_loader):

        print("Computing fisher matrix...")
        fisher = {n: 0. * p.data for n, p in self.networks_prec.backbone.named_parameters()}
        self.networks_prec.backbone.train()
        self.networks_prec.backbone.requires_grad_(True)

        for i, inputs in enumerate(train_loader, 1):

            inputs, pids = self._parse_data(inputs)

            self.networks_prec.backbone.zero_grad()

            outputs = self.networks_prec(inputs)
            triplet_loss, *_ = self.triplet_criterion(outputs['global'], pids)

            triplet_loss.backward()

            for n, p in self.networks_prec.backbone.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)

            print(f"Fisher Matrix Estimating: [{i}/{len(train_loader)}]")

        for n, _ in self.networks_prec.backbone.named_parameters():
            fisher[n] = fisher[n] / len(train_loader)
            fisher[n] = fisher[n].detach()
            fisher[n].requires_grad_(False)

        return fisher

    def train_step(self, inputs, pids):
        self.optimizer.zero_grad()
        outputs = self.networks(inputs)
        loss, losses = self._compute_loss(outputs, pids)
        loss.backward()
        self.optimizer.step()
        return losses

    def _compute_loss(self, outputs, pids, *args):
        pooled, preds = outputs["global"], outputs["preds"]
        pid_loss, triplet_loss = self.basic_criterion(pooled, preds, pids)

        loss_drift = [(self.fisher[name] * (param_old - param).pow(2)).sum() / 2.
                      for (name, param), (_, param_old) in
                      zip(self.networks.backbone.named_parameters(), self.networks_prec.backbone.named_parameters())]
        loss_drift = self.weight * sum(loss_drift)

        loss = pid_loss + triplet_loss + loss_drift
        return loss, [pid_loss.item(), triplet_loss.item(), loss_drift.item()]


class MASTrainer(EWCTrainer):
    def __init__(self,
                 networks_prec,
                 networks,
                 incremental_classes,
                 optimizer,
                 lr_scheduler
                 ):

        super(MASTrainer, self).__init__(networks_prec,
                                         networks,
                                         incremental_classes,
                                         optimizer,
                                         lr_scheduler,
                                         )
        self.weight = 1e5

    def fisher_matrix_diag(self, train_loader):

        print("Computing fisher matrix...")
        fisher = {n: 0. * p.data for n, p in self.networks_prec.backbone.named_parameters()}
        self.networks_prec.backbone.train()
        self.networks_prec.backbone.requires_grad_(True)

        for i, inputs in enumerate(train_loader, 1):

            inputs, pids = self._parse_data(inputs)

            self.networks_prec.backbone.zero_grad()

            outputs = self.networks_prec(inputs)
            loss = torch.norm(outputs['global'], 2, dim=1).mean()
            loss.backward()

            for n, p in self.networks_prec.backbone.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)

            print(f"Fisher Matrix Estimating: [{i}/{len(train_loader)}]")

        for n, _ in self.networks_prec.backbone.named_parameters():
            fisher[n] = fisher[n] / len(train_loader)
            fisher[n] = fisher[n].detach()
            fisher[n].requires_grad_(False)

        return fisher
