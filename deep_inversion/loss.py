# -*- coding: utf-8 -*-
# Time    : 2020/8/4 14:04
# Author  : Yichen Lu

import time
import math
import torch
from torch.nn import functional as F
import pickle
from reid.loss.utils import cos


def prior_losses(inputs):
    # COMPUTE total variation regularization loss
    diff1 = inputs[:, :, :, :-1] - inputs[:, :, :, 1:]
    diff2 = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]
    diff3 = inputs[:, :, 1:, :-1] - inputs[:, :, :-1, 1:]
    diff4 = inputs[:, :, :-1, :-1] - inputs[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


def negative_js_divergence(outputs, outputs_student, T=3.0):
    # Jensen Shanon divergence:
    # another way to force KL between negative probabilities
    P = F.softmax(outputs_student / T, dim=1).clamp(0.001, 0.999)
    P = P / P.sum(dim=1, keepdim=True)
    Q = F.softmax(outputs / T, dim=1).clamp(0.001, 0.999)
    Q = Q / Q.sum(dim=1, keepdim=True)
    M = 0.5 * (P + Q)

    loss_verifier_cig = 0.5 * F.kl_div(M.log(), P, reduction='batchmean') + 0.5 * F.kl_div(M.log(), Q, reduction='batchmean')
    # JS criteria - 0 means full correlation, ln(2) - means completely different
    loss_verifier_cig = math.log(2) - torch.clamp(loss_verifier_cig, 0.0, math.log(2))
    return loss_verifier_cig


def hist_loss(data, target=None, lb=-1., ub=1., R=20, H_target=None):
    assert target is not None or H_target is not None
    H_data = differential_hist(data, lb, ub, R)
    H_target = differential_hist(target, lb, ub, R) if H_target is None else H_target
    H_avg = [(h_data + h_target) / 2 for h_data, h_target in zip(H_data, H_target)]

    # divergence = 0.5 * F.kl_div(H_avg.log(), H_data, reduction='batchmean') + \
    #              0.5 * F.kl_div(H_avg.log(), H_target, reduction='batchmean')

    divergence = 0.5 * kl_div_list(H_avg, H_target) + 0.5 * kl_div_list(H_avg, H_data)

    return divergence


def differential_hist(data, lb=-1., ub=1., R=20):
    bin_width = (ub - lb) / R
    bins = [[lb + (r - 1) * bin_width,
             lb + r * bin_width,
             lb + (r + 1) * bin_width] for r in range(1, R)]
    H = []
    for sub_lb, center, sub_ub in bins:
        left_mask = (sub_lb <= data) & (data <= center)
        right_mask = (center < data) & (data <= sub_ub)
        delta = (data[left_mask] - sub_lb).sum() + (sub_ub - data[right_mask]).sum()
        H.append(delta / bin_width / data.size(0))

    H = [h / sum(H) for h in H]
    # H = torch.tensor(H, requires_grad=True)
    # H = H / H.sum()
    return H


def kl_div_list(data, target):
    assert len(data) == len(target), "Data and target must be the same size."
    return sum([h_target * math.log(h_target / h_data) if h_target > 1e-6 else 0
                for h_data, h_target in zip(data, target)])


def hist_loss_beta(data, target=None, lb=-1., ub=1., R=20, H_target=None):
    assert target is not None or H_target is not None
    bin_width = (ub - lb) / R
    bins = [[lb + r * bin_width,
             lb + (r + 1) * bin_width] for r in range(0, R)]
    H_data = data.histc(R, min=lb, max=ub) / data.size(0)
    H_data = H_data.detach()
    H_target = target.histc(R, min=lb, max=ub) / target.size(0) if H_target is None else H_target
    H_target = H_target.detach()

    delta = (H_target - H_data).unsqueeze(dim=1)
    mask_l = torch.ones(R, R, device=data.device).tril(diagonal=-1)
    Theta_l = mask_l.mm(delta).squeeze()
    mask_r = torch.ones(R, R, device=data.device).triu(diagonal=1)
    Theta_r = mask_r.mm(delta).squeeze()

    divergence = 0
    for (sub_lb, sub_ub), theta_l, theta_r in zip(bins, Theta_l, Theta_r):
        values = data[(sub_lb < data) & (data < sub_ub)]
        divergence = divergence + (values * (theta_l - theta_r)).sinh().sum()
    divergence = divergence.div(data.size(0))

    return divergence


class GravityCriterion(object):
    def __init__(self, lb=-1., ub=1., R=20, H_target=None):
        self.lb = lb
        self.ub = ub
        self.R = R
        self.H_target = H_target.detach()

        self.bin_width = (self.ub - self.lb) / self.R
        self.bins = [[self.lb + r * self.bin_width,
                      self.lb + (r + 1) * self.bin_width] for r in range(0, self.R)]
        self.bin_lbs = torch.tensor([self.lb + r * self.bin_width for r in range(0, self.R)]).cuda()
        self.bin_ubs = torch.tensor([self.lb + (r + 1) * self.bin_width for r in range(0, self.R)]).cuda()

    def __call__(self, embeddings, labels):
        label_pool = labels.unique()
        divergences = []
        for label in label_pool:
            embeddings_intra = embeddings[labels == label]
            similarities = cos(embeddings_intra, embeddings_intra)
            similarities = similarities[torch.ones_like(similarities, dtype=torch.bool).triu(diagonal=1)]
            H_data = similarities.histc(self.R, min=self.lb, max=self.ub) / similarities.size(0)
            H_data = H_data.detach()

            delta = (self.H_target - H_data).unsqueeze(dim=1)
            mask_l = torch.ones(self.R, self.R, device=embeddings.device).tril(diagonal=-1)
            Theta_l = mask_l.mm(delta)
            mask_r = torch.ones(self.R, self.R, device=embeddings.device).triu(diagonal=1)
            Theta_r = mask_r.mm(delta)

            # divergence = 0
            # for (sub_lb, sub_ub), theta_l, theta_r in zip(self.bins, Theta_l, Theta_r):
            #     values = similarities[(sub_lb < similarities) & (similarities < sub_ub)]
            #     divergence = divergence + (values * (theta_l - theta_r)).sinh().sum()
            #
            # divergence = divergence.div(similarities.size(0))

            # now = time.time()
            bin_mask = (similarities.unsqueeze(dim=1) > self.bin_lbs.unsqueeze(dim=0)) & \
                       (similarities.unsqueeze(dim=1) < self.bin_ubs.unsqueeze(dim=0))
            gravity = bin_mask.float().mm(Theta_l - Theta_r).squeeze()
            divergence = (similarities * gravity).sinh().mean()
            # print(f"1: {time.time() - now}")

            divergences.append(divergence)

        return sum(divergences) / len(divergences)


class CrossEntropyCriterion(object):
    def __call__(self, preds, labels):
        if len(labels.size()) == 1:
            return F.cross_entropy(preds, labels)
        elif len(labels.size()) > 1:
            probs = preds.softmax(dim=1)
            loss = (-1 * labels * probs.log()).sum(dim=1)
            loss = loss.mean()
            return loss
        else:
            raise RuntimeError("Invalid labels.")

if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt

    # target = (torch.randn(10000) / 2).clamp(-1, 1)
    # H_target = differential_hist(target)

    with open('distribution.pkl', 'rb') as file:
        H_target_h = pickle.load(file)
    H_target_h = torch.tensor(H_target_h, dtype=torch.float)

    target = torch.load('./similarities.pth')
    H_target = differential_hist(target, R=100)
    # data = torch.randn(1000, requires_grad=True)
    data = (torch.randn(1000, requires_grad=True) / 18 + 0.5).detach().requires_grad_(True)
    optimizer = torch.optim.SGD([data, ], lr=0.1, momentum=0.9)

    # STEPS = 5000
    # for i in range(STEPS):
    #     loss = hist_loss(data, R=100, H_target=H_target)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     print(f"Step: [{i + 1}] / [{STEPS}], Loss: {loss:.4f}")
    #     data.data = data.data.clamp_(-0.99, 0.99)
    #
    #     if (i < 1000 and (i + 1) % 5 == 0) or (i > 1000 and (i + 1) % 20 == 0) or (i > 2500 and (i + 1) % 50 == 0):
    #         fig, ax = plt.subplots()
    #         ax.plot(H_target_h)
    #         ax.scatter(list(range(100)), H_target_h)
    #
    #         H_data = data.histc(100, min=-1., max=1.) / data.size(0)
    #         H_data = H_data.detach()
    #         ax.plot(H_data)
    #         ax.scatter(list(range(100)), H_data)
    #
    #         plt.savefig(f'./frames/{i:04d}.png')

    # with open('distribution.pkl', 'rb') as file:
    #     H_target = pickle.load(file)
    # H_target = torch.tensor(H_target, dtype=torch.float)
    #
    # plt.plot(H_target)
    # plt.scatter(list(range(100)), H_target)
    # H_data = data.histc(100, min=-1., max=1.) / data.size(0)
    # H_data = H_data.detach()
    # plt.plot(H_data)
    # plt.scatter(list(range(100)), H_data)
    # plt.show()

    # H_target = target.histc(20, min=-1., max=1.) / target.size(0)
    # H_target = H_target.detach()

    with open('distribution.pkl', 'rb') as file:
        H_target = pickle.load(file)
    H_target = torch.tensor(H_target, dtype=torch.float)

    fig, ax = plt.subplots()
    STEPS = 5000
    for i in range(STEPS):
        loss = hist_loss_beta(data, R=100, H_target=H_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Step: [{i + 1}] / [{STEPS}], Loss: {loss:.4f}")
        data.data = data.data.clamp_(-0.99, 0.99)

        if (i < 1000 and (i + 1) % 5 == 0) or (i > 1000 and (i + 1) % 20 == 0) or (i > 2500 and (i + 1) % 50 == 0):
            plt.clf()
            plt.plot(H_target)
            plt.scatter(list(range(100)), H_target)

            H_data = data.histc(100, min=-1., max=1.) / data.size(0)
            H_data = H_data.detach()
            plt.plot(H_data)
            plt.scatter(list(range(100)), H_data)

            plt.savefig(f'./frames/{i:04d}.png')
