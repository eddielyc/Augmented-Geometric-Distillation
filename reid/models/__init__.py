from torch import nn as nn
from reid.models.backbone import *
from reid.models.classifier import *


class Networks(nn.Module):
    def __init__(self, backbone, classifier):
        super(Networks, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, inputs, *args, **kwargs):
        outputs = self.backbone(inputs, *args, **kwargs)
        preds = self.classifier(outputs["embedding"])
        outputs["preds"] = preds
        return outputs
