# -*- coding: utf-8 -*-
# Time    : 2020/12/14 15:09
# Author  : Yichen Lu

from os import path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from reid.utils.utils import DivisibleDict
import math


model_path = {
    'resnet50_ibn': '~/.cache/torch/hub/checkpoints/resnet50_ibn_a.pth',
}


class ResNetIBN(nn.Module):

    def __init__(self, depth=50, pretrained=True, last_stride=2, last_pooling='avg', embedding=2048):
        super(ResNetIBN, self).__init__()

        assert depth in (50, 34), "Invalid depth for ResNet-IBN, expect depth in (34, 50)."
        self.depth = depth
        self.pretrained = pretrained
        self.last_pooling = last_pooling
        self.last_stride = last_stride
        self.embedding = embedding
        self.forward_hooks, self.backward_hooks = [], []
        self.forward_records, self.backward_records = {}, {}

        # Construct base (pretrained) resnet
        block = Bottleneck
        self.base = Base(block, layers=[3, 4, 6, 3], last_stride=self.last_stride)

        if pretrained:
            ckpt = model_path[f'resnet{depth}_ibn']
            assert osp.isfile(
                osp.expanduser(ckpt)), "torch pretrained model doesn't exists."
            state_dict = torch.load(osp.expanduser(ckpt))
            self.base.load_state_dict(state_dict, strict=False)

        self.bn = nn.BatchNorm1d(self.embedding)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        self.bn.bias.requires_grad_(False)

        print(self)

    def forward(self, x, out=None, *kwargs):
        x1, x2, x3, feature_map = self.base(x)

        B, C, H, W = feature_map.size()

        if self.last_pooling == "max":
            feature = F.adaptive_max_pool2d(feature_map, 1).view(B, -1)
        else:
            feature = F.adaptive_avg_pool2d(feature_map, 1).view(B, -1)

        embedded_features = self.bn(feature)

        if out is not None:
            return DivisibleDict({'embedding': embedded_features.detach().cuda(out),
                                  'global': feature.detach().cuda(out),
                                  'maps': [x1.detach().cuda(out), x2.detach().cuda(out), x3.detach().cuda(out),
                                           feature_map.detach().cuda(out)],
                                  })

        return DivisibleDict({'embedding': embedded_features,
                              'global': feature,
                              'maps': [x1, x2, x3, feature_map],
                              })

    def create_forward_hook(self, name):
        def forward_hook(module, input, output):
            self.forward_records[name] = output
            return None

        return forward_hook

    def create_backward_hook(self, name):
        def backward_hook(module, grad_input, grad_output):
            self.backward_records[name] = grad_output[0]
            return None

        return backward_hook

    def remove_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()

        for hook in self.backward_hooks:
            hook.remove()
        print("Removed all hooks.")

    def __repr__(self):
        return f"Build ResNet{self.depth}-IBN(pretrained={self.pretrained}, last_stride={self.last_stride}, " \
               f"last_pooling={self.last_pooling}, embedding={self.embedding}) \n" \
               f"Modules: {self._modules} \n"


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, dim=1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), dim=1)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = IBN(planes) if ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Base(nn.Module):

    def __init__(self, block, layers, last_stride=2):
        super(Base, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn = False if planes == 512 else True
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


if __name__ == '__main__':
    # model = Base(Bottleneck, [3, 4, 6, 3])
    # state_dict = torch.load(osp.expanduser(model_path['resnet50_ibn']))
    # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    model = ResNetIBN(50, True, 2, 'avg', 2048)
    pass
