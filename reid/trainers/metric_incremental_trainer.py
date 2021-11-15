# -*- coding: utf-8 -*-
# Time    : 2021/8/7 9:56
# Author  : Yichen Lu


from reid.trainers import IncrementalTrainer
from reid.loss import MetricDistillationLoss
from reid.utils import MemoryBank


class MetricIncrementalTrainer(IncrementalTrainer):
    def __init__(self,
                 networks_prec,
                 networks,
                 incremental_classes,
                 optimizer,
                 lr_scheduler,
                 memory_size=1024,
                 ):

        super(MetricIncrementalTrainer, self).__init__(networks_prec,
                                                       networks,
                                                       incremental_classes,
                                                       optimizer,
                                                       lr_scheduler
                                                       )

        self.distillation_criterion = MetricDistillationLoss()
        self.memory_size = memory_size
        self.memory_bank = MemoryBank(self.memory_size)
        self.outdated_memory_bank = MemoryBank(self.memory_size)

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

    def train_step(self, inputs, pids):
        self.optimizer.zero_grad()
        outdated_outputs = self.networks_prec(inputs).detach()
        self.outdated_memory_bank.push(outdated_outputs["global"])
        outputs = self.networks(inputs)
        self.memory_bank.push(outputs["global"])
        loss, losses = self._compute_loss(outputs, outdated_outputs, pids)
        loss.backward()
        self.optimizer.step()
        return losses

    def _compute_loss(self, outputs, outdated_outputs, pids):
        pooled, preds = outputs["global"], outputs["preds"]
        pid_loss, triplet_loss = self.basic_criterion(pooled, preds, pids)

        distillation_loss = self.distillation_criterion(pooled,
                                                        outdated_outputs["global"],
                                                        self.memory_bank(),
                                                        self.outdated_memory_bank(),
                                                        )

        loss = pid_loss + triplet_loss + distillation_loss
        return loss, [pid_loss.item(), triplet_loss.item(), distillation_loss.item()]
