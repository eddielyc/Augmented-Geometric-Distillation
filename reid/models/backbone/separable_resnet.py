# -*- coding: utf-8 -*-
# Time    : 2020/3/12 15:37
# Author  : Yichen Lu

from os import path as osp
from copy import deepcopy
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from reid.models.backbone.modules import SeparateBN, Sequential


class SeparableResNet(nn.Module):

    def __init__(self, pretrained=True, last_stride=2, last_pooling='avg', embedding=256, num_domains=2):
        super(SeparableResNet, self).__init__()

        self.pretrained = pretrained
        self.last_pooling = last_pooling
        self.last_stride = last_stride
        self.embedding = embedding
        self.num_domains = num_domains

        # Construct base (pretrained) resnet
        self.base = Base(Bottleneck, layers=[3, 4, 6, 3], last_stride=self.last_stride, pretrained=pretrained,
                         num_domains=self.num_domains)

        bn = nn.BatchNorm1d(self.embedding)
        nn.init.constant_(bn.weight, 1)
        nn.init.constant_(bn.bias, 0)
        bn.bias.requires_grad_(False)
        self.bn = SeparateBN(bn, self.num_domains)

        print(self)

    def split(self, src_key, new_key):
        for module in self.modules():
            if isinstance(module, SeparateBN):
                module.clone(src_key, new_key)

    def forward(self, x, domain_indices=0, output_feature=None):
        feature_map = self.base(x, domain_indices)

        B, C, H, W = feature_map.size()

        if self.last_pooling == "max":
            feature = F.adaptive_max_pool2d(feature_map, 1).view(B, -1)
        else:
            feature = F.adaptive_avg_pool2d(feature_map, 1).view(B, -1)

        embedded_features = self.bn(feature, domain_indices)

        if not self.training:
            feature = F.normalize(feature, 2, dim=1).view(B, -1)

        return {'embedding': embedded_features,
                'global': feature,
                'map': feature_map,
                }

    def __repr__(self):
        return f"Build SeparableResNet(pretrained={self.pretrained}, last_stride={self.last_stride}, " \
               f"last_pooling={self.last_pooling}, embedding={self.embedding}, num_domains={self.num_domains}) \n" \
               f"Modules: {self._modules} \n"


class Base(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, num_domains=2, last_stride=2,
                 pretrained=False):
        super(Base, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_domains = num_domains

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = SeparateBN(norm_layer(self.inplanes), self.num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                # modified from torchvision version
                nn.init.constant_(m.bn3.weight, 0)

        if pretrained:
            assert osp.isfile(
                osp.expanduser('~/.cache/torch/checkpoints/resnet50-19c8e357.pth')), "torch pretrained model " \
                                                                                     "doesn't exists."
            state_dict = torch.load(osp.expanduser('~/.cache/torch/checkpoints/resnet50-19c8e357.pth'))
            state_dict = self._preprocess_state_dict(state_dict)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

    def _preprocess_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            # replace all BN layer parameters with SeparateBN layer parameters.
            # note that: all the second modules in downsample blocks are BN layers.
            if "bn" in key or "downsample.1" in key:
                value = state_dict[key]
                parts = key.split(".")
                for bn_key in range(self.num_domains):
                    new_key = ".".join(parts[:-1]) + "." + ".".join(["bns", str(bn_key), parts[-1]])
                    new_state_dict[new_key] = deepcopy(value)
        for key in [key for key in state_dict.keys() if "bn" in key or "downsample.1" in key]:
            state_dict.pop(key)
        state_dict.update(new_state_dict)

        return state_dict

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                SeparateBN(norm_layer(planes * block.expansion), self.num_domains),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return Sequential(*layers)

    def forward(self, x, domain_indices):
        x = self.conv1(x)
        x = self.bn1(x, domain_indices)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, domain_indices)
        x = self.layer2(x, domain_indices)
        x = self.layer3(x, domain_indices)
        x = self.layer4(x, domain_indices)

        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, num_domains=2):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = SeparateBN(norm_layer(width), num_domains)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = SeparateBN(norm_layer(width), num_domains)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = SeparateBN(norm_layer(planes * self.expansion), num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, domain_indices):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, domain_indices)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, domain_indices)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, domain_indices)

        if self.downsample is not None:
            identity = self.downsample(x, domain_indices)

        out += identity
        out = self.relu(out)

        return out


if __name__ == '__main__':
    model = Base(Bottleneck, [3, 4, 6, 3])
    pass