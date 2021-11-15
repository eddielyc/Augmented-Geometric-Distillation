# -*- coding: utf-8 -*-
# Time    : 2020/8/4 16:56
# Author  : Yichen Lu

import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(ResNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, inputs):
        outputs = self.feature_extractor(inputs)
        preds = self.classifier(outputs['embedding'])
        return preds, outputs
