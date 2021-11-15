import math
import torch
from torch import nn
from torch.nn import functional as F


class MultiBranchClassifier(nn.Module):
    def __init__(self, **classifiers):
        super(MultiBranchClassifier, self).__init__()
        self.classifiers = nn.ModuleDict(classifiers)

        print(self)

    def forward(self, key, embedded_features, labels, **kwargs):

        return self.classifiers[key](embedded_features, labels)

    def reset_w(self, key, groups, memory_bank):
        self.classifiers[key].reset_w(groups, memory_bank)

    def add_classifiers(self, **classifiers):
        self.classifiers.update(classifiers)
        print(f"Update classifiers: {classifiers} \n"
              f"MultiBranchClassifier with classifiers: {self.classifiers} \n")

    def __repr__(self):
        return f"Build MultiBranchClassifier with classifiers: {self.classifiers} \n"


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device, *args, **kwargs):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        self.W = nn.Parameter(torch.randn(self.out_features, self.in_features, device=self.device), requires_grad=True)
        nn.init.normal_(self.W, std=0.001)

        print(self)

    def forward(self, input, *args):
        return input.mm(self.W.t())

    def reset_w(self, groups, memory_bank):
        weights = torch.randn(len(groups), self.in_features, device=self.device)
        for idx, members in groups.items():
            member_features = memory_bank[list(members)].mean(dim=0)
            weights[idx] = member_features.detach()

        self.W = nn.Parameter(weights, requires_grad=True)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, " \
               f"device={self.device}) \n"

    def expand(self, num_classes):
        self.out_features += num_classes
        new_W = nn.Parameter(torch.randn(num_classes, self.in_features, device=self.device), requires_grad=True)
        nn.init.normal_(new_W, std=0.001)

        expanded_W = torch.cat([new_W.data, self.W.data], dim=0)
        self.W = nn.Parameter(expanded_W, requires_grad=True)
        print(f"Expand W from {self.out_features-num_classes} classes to {self.out_features}")


class Sphere(nn.Module):
    def __init__(self, in_features, out_features, device, scale):
        super(Sphere, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.scale = scale

        self.W = nn.Parameter(torch.randn(self.out_features, self.in_features), requires_grad=True)
        nn.init.normal_(self.W, std=0.001)

        print(self)

    def forward(self, input, *args):
        input_l2 = input.pow(2).sum(dim=1, keepdim=True).pow(0.5).clamp(min=1e-12)
        input_norm = input / input_l2
        W_l2 = self.W.pow(2).sum(dim=1, keepdim=True).pow(0.5).clamp(min=1e-12)
        W_norm = self.W / W_l2
        cos_th = input_norm.mm(W_norm.t())
        s_cos_th = self.scale * cos_th

        return s_cos_th

    def reset_w(self, groups, memory_bank):
        weights = torch.randn(len(groups), self.in_features, device=self.device)
        for idx, members in groups.items():
            member_features = memory_bank[list(members)].mean(dim=0)
            weights[idx] = member_features.detach()

        self.W = nn.Parameter(weights, requires_grad=True)

    def __repr__(self):
        return f"Sphere(in_features={self.in_features}, out_features={self.out_features}, " \
               f"device={self.device}, scale={self.scale}) \n"


class Arc(nn.Module):

    def __init__(self, in_features, out_features, device, scale=20.0, margin=0.1):
        super(Arc, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features).to(self.device), requires_grad=True)

        nn.init.normal_(self.weight, std=0.001)

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

        print(self)

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output

    def __repr__(self):
        return f"Arc(in_features={self.in_features}, out_features={self.out_features}, " \
               f"device={self.device}, scale={self.scale}), margin={self.margin}. \n"


class NoNormArc(nn.Module):

    def __init__(self, in_features, out_features, device, margin=0.06):
        super(NoNormArc, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features).to(self.device), requires_grad=True)

        nn.init.normal_(self.weight, std=0.001)

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

        print(self)

    def forward(self, x, label):
        x_norm = x.norm(p=2, dim=1, keepdim=True)
        w_norm = self.weight.norm(p=2, dim=1, keepdim=True)

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * x_norm * w_norm.t()

        return output

    def __repr__(self):
        return f"NoNormArc(in_features={self.in_features}, out_features={self.out_features}, " \
               f"device={self.device}, margin={self.margin}. \n"
