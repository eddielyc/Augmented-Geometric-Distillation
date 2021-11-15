# -*- coding: utf-8 -*-
# Time    : 2021/8/7 14:29
# Author  : Yichen Lu

import torch

from reid.trainers import Trainer
from reid.loss import distillation
from reid.utils import AverageMeter, AverageMeters


class InversionIncrementalTrainer(Trainer):
    def __init__(self,
                 networks_prec,
                 networks,
                 incremental_classes,
                 optimizer,
                 lr_scheduler,
                 algo_config=None,
                 **kwargs
                 ):

        super(InversionIncrementalTrainer, self).__init__()
        self.networks_prec = networks_prec
        self.networks = networks
        self.incremental_classes = incremental_classes

        self.trainables = [self.networks]
        self.untrainables = [self.networks_prec]

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.distillation_config = algo_config["distillation"]

        self.distillation_criterion = distillation.factory[self.distillation_config["type"]](
            self.distillation_config["config"],
        )

        self.meters = AverageMeters(AverageMeter("Batch Time"),
                                    AverageMeter("Pid Loss"),
                                    AverageMeter("Triplet Loss"),
                                    AverageMeter("Distillation Loss"),
                                    )

    def train(self, epoch, training_loader, training_loader_prec):
        self.before_train(epoch)

        training_loader_prec_iter = iter(training_loader_prec)

        for i, inputs in enumerate(training_loader):
            try:
                inputs_prec = next(training_loader_prec_iter)
            except StopIteration:
                training_loader_prec_iter = iter(training_loader_prec)
                inputs_prec = next(training_loader_prec_iter)

            inputs, pids = self._parse_data(inputs)
            inputs_prec, pids_prec = self._parse_data(inputs_prec)
            # To avoid overlapped pids
            pids_prec = pids_prec + self.incremental_classes
            losses = self.train_step(inputs, pids, inputs_prec, pids_prec)

            self.meters.update([self.timer(),
                                *losses
                                ])
            print(f"Epoch: [{epoch}][{i + 1}/{len(training_loader)}], " + self.meters())

        self.after_train()

    def before_train(self, epoch):
        super(InversionIncrementalTrainer, self).before_train()
        self.lr_scheduler.step(epoch)

    def train_step(self, inputs, pids, inputs_prec, pids_prec):
        """
        New data + model -> outputs
        New data + preceding model -> outdated_outputs
        Old data + model -> outputs_prec
        Old data + preceding model -> outdated_outputs_prec
        """

        self.optimizer.zero_grad()
        outdated_outputs_prec = self.networks_prec(inputs_prec).detach()
        outputs, outputs_prec = self.networks(torch.cat([inputs, inputs_prec], dim=0)).divide(2)
        loss, losses = self._compute_loss(outputs, outputs_prec, outdated_outputs_prec, pids, pids_prec)
        loss.backward()
        self.optimizer.step()
        return losses

    def _compute_loss(self, outputs, outputs_prec, outdated_outputs_prec, pids, pids_prec):
        pooled, preds = outputs["global"], outputs["preds"]
        pooled_prec, preds_prec = outputs_prec["global"], outputs_prec["preds"]
        pid_loss, triplet_loss = self.basic_criterion(torch.cat([pooled, pooled_prec], dim=0),
                                                      torch.cat([preds, preds_prec], dim=0),
                                                      torch.cat([pids, pids_prec], dim=0)
                                                      )

        distillation_loss = self.distillation_criterion(outputs_prec, outdated_outputs_prec, pids_prec)
        distillation_loss = self.distillation_config["factor"] * distillation_loss

        loss = pid_loss + triplet_loss + distillation_loss
        return loss, [pid_loss.item(), triplet_loss.item(), distillation_loss.item()]


class InverXionIncrementalTrainer(InversionIncrementalTrainer):
    def __init__(self,
                 networks_prec,
                 networks,
                 incremental_classes,
                 optimizer,
                 lr_scheduler,
                 algo_config=None,
                 **kwargs
                 ):

        super(InverXionIncrementalTrainer, self).__init__(networks_prec,
                                                          networks,
                                                          incremental_classes,
                                                          optimizer,
                                                          lr_scheduler,
                                                          algo_config,
                                                          **kwargs)

    def before_train(self, epoch):
        super(InverXionIncrementalTrainer, self).before_train(epoch)
        self.lr_scheduler.step(epoch)

    def train_step(self, inputs_list, pids, inputs_prec_list, pids_prec):
        """
        New data + model -> outputs
        New data + preceding model -> outdated_outputs
        Old data + model -> outputs_prec
        Old data + preceding model -> outdated_outputs_prec
        """

        self.optimizer.zero_grad()

        outdated_outputs_prec_list = [
            self.networks_prec(inputs_prec).detach()
            for inputs_prec in inputs_prec_list
        ]

        for i, (inputs, inputs_prec) in enumerate(zip(inputs_list, inputs_prec_list)):
            outputs, outputs_prec = self.networks(torch.cat([inputs, inputs_prec], dim=0)).divide(2)
            loss, losses = self._compute_loss(outputs, outputs_prec,
                                              [outdated_outputs_prec_list[i]] + outdated_outputs_prec_list[:i] + outdated_outputs_prec_list[i+1:],
                                              pids,
                                              pids_prec,
                                              )
            (loss / len(inputs_list)).backward()

        self.optimizer.step()
        return losses

    def _compute_loss(self, outputs, outputs_prec, outdated_outputs_prec, pids, pids_prec):
        pooled, preds = outputs["global"], outputs["preds"]
        pooled_prec, preds_prec = outputs_prec["global"], outputs_prec["preds"]
        pid_loss, triplet_loss = self.basic_criterion(torch.cat([pooled, pooled_prec], dim=0),
                                                      torch.cat([preds, preds_prec], dim=0),
                                                      torch.cat([pids, pids_prec], dim=0)
                                                      )

        distillation_loss = [self.distillation_criterion(outputs_prec, outdated_outputs, pids_prec)
                             for outdated_outputs in outdated_outputs_prec]

        distillation_loss = self.distillation_config["cross_factor"] * distillation_loss[0] + \
            (1. - self.distillation_config["cross_factor"]) * sum(distillation_loss[1:]) / (len(distillation_loss) - 1)
        distillation_loss = self.distillation_config["factor"] * distillation_loss

        loss = pid_loss + triplet_loss + distillation_loss
        return loss, [pid_loss.item(), triplet_loss.item(), distillation_loss.item()]

    def _parse_data(self, inputs):
        imgs_list, _, pids, _ = inputs
        inputs_list = [imgs.to(self.device) for imgs in imgs_list]
        pids = pids.to(self.device)
        return inputs_list, pids
