# -*- coding: utf-8 -*-
# Time    : 2021/8/7 10:11
# Author  : Yichen Lu


import torch
from torch import nn as nn

from reid.trainers import IncrementalTrainer
from reid.loss import ClassificationDistillationLoss, MetricDistillationLoss


class GenerationIncrementalTrainer(IncrementalTrainer):
    def __init__(self,
                 networks_prec,
                 networks,
                 incremental_classes,
                 optimizer,
                 lr_scheduler,
                 generator,
                 ):

        super(GenerationIncrementalTrainer, self).__init__(networks_prec,
                                                           networks,
                                                           incremental_classes,
                                                           optimizer,
                                                           lr_scheduler,
                                                           )

        self.generator = generator

        self.cls_distillation_criterion = ClassificationDistillationLoss()
        self.metric_distillation_criterion = MetricDistillationLoss()

        self.label_encoder = nn.Embedding(*self.networks_prec.classifier.W.size())
        self.label_encoder.weight.data = self.networks_prec.classifier.W.data.cpu()

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

    def generate_batch(self, batch_size=64):
        targets = torch.randint(1041, size=(batch_size // 4,), dtype=torch.long)
        targets = targets.repeat(4)
        embedded_targets = self.label_encoder(targets)
        noise = torch.randn((batch_size, 256))
        merged = torch.cat([embedded_targets, noise], dim=-1).cuda()

        fakes = self.generator(merged)
        return fakes, targets.cuda()

    def train_step(self, inputs, pids):
        """
        New data + model -> outputs
        New data + preceding model -> outdated_outputs
        Old data + model -> outputs_prec
        Old data + preceding model -> outdated_outputs_prec
        """
        inputs_prec, pids_prec = self.generate_batch(inputs.size(0))

        self.optimizer.zero_grad()
        outdated_outputs_prec = self.networks_prec(inputs_prec).detach()
        # Outputs here contain "outputs" and "outputs_prec"
        outputs = self.networks(torch.cat([inputs, inputs_prec], dim=0))
        loss, losses = self._compute_loss(outputs, outdated_outputs_prec, pids)
        loss.backward()
        self.optimizer.step()
        return losses

    def _compute_loss(self, outputs, outdated_outputs_prec, pids):
        outputs, outputs_prec = outputs.divide(2)
        pooled, preds = outputs["global"], outputs["preds"]
        pid_loss, triplet_loss = self.basic_criterion(pooled, preds, pids)

        distillation_loss = self.metric_distillation_criterion(outputs_prec["global"],
                                                               outdated_outputs_prec["global"])

        loss = pid_loss + triplet_loss + distillation_loss
        return loss, [pid_loss.item(), triplet_loss.item(), distillation_loss.item()]
