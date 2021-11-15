# -*- coding: utf-8 -*-
# Time    : 2021/8/7 14:29
# Author  : Yichen Lu

import torch

from reid.trainers import InversionIncrementalTrainer
from reid.loss import distillation
from reid.utils import AverageMeter, AverageMeters


class AlwaysBeDreamingTrainer(InversionIncrementalTrainer):
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
        outputs_prec["outdated_preds"] = self.networks_prec.classifier(outputs_prec["embedding"])
        loss, losses = self._compute_loss(outputs, outputs_prec, outdated_outputs_prec, pids, pids_prec)
        loss.backward()
        self.optimizer.step()
        return losses
