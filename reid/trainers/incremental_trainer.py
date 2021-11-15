# -*- coding: utf-8 -*-
# Time    : 2021/8/6 11:42
# Author  : Yichen Lu


from reid.trainers import Trainer
from reid.loss import distillation
from reid.utils.meters import AverageMeter, AverageMeters


class IncrementalTrainer(Trainer):
    def __init__(self,
                 networks_prec,
                 networks,
                 incremental_classes,
                 optimizer,
                 lr_scheduler,
                 algo_config=None,
                 ):

        super(IncrementalTrainer, self).__init__()
        # preceding networks
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
        super(IncrementalTrainer, self).before_train()
        self.lr_scheduler.step(epoch)

    def train_step(self, inputs, pids):
        """
        New data + model -> outputs
        New data + preceding model -> outdated_outputs
        Old data + model -> outputs_prec
        Old data + preceding model -> outdated_outputs_prec
        """
        self.optimizer.zero_grad()
        outputs_prec = self.networks_prec(inputs).detach()
        outputs = self.networks(inputs)
        loss, losses = self._compute_loss(outputs, outputs_prec, pids)
        loss.backward()
        self.optimizer.step()
        return losses

    def _compute_loss(self, outputs, outputs_prec, pids):
        pooled, preds = outputs["global"], outputs["preds"]
        pid_loss, triplet_loss = self.basic_criterion(pooled, preds, pids)

        distillation_loss = self.distillation_criterion(outputs, outputs_prec, pids)
        distillation_loss = self.distillation_config["factor"] * distillation_loss

        loss = pid_loss + triplet_loss + distillation_loss
        return loss, [pid_loss.item(), triplet_loss.item(), distillation_loss.item()]
