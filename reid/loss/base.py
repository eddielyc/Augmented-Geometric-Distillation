# -*- coding: utf-8 -*-
# Time    : 2021/8/6 18:49
# Author  : Yichen Lu
import torch
from torch import nn as nn, nn
from torch.nn import functional as F

from reid.loss.utils import euclidean_dist


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        B, num_classes = inputs.size()
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (- targets * log_probs).sum(dim=1).mean()
        return loss


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TriHardLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None, dist_func=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

        self.dist_func = euclidean_dist if dist_func is None else dist_func
        print(f"Using {self.dist_func.__name__} as distance function.")

    def __call__(self, global_feat, labels, attentions=None):
        dist_mat = self.dist_func(global_feat, global_feat, attentions, attentions)
        dist_ap, dist_an, p_inds, n_inds = hard_example_mining(dist_mat, labels, return_inds=True)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an, p_inds, n_inds

    def set_margin(self, margin):
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)


class TriSoftLoss(object):
    def __init__(self, margin, dist_func=None):
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

        self.dist_func = euclidean_dist if dist_func is None else dist_func
        print(f"Using {self.dist_func.__name__} as distance function.")

    def __call__(self, global_feat, labels):
        dist_mat = self.dist_func(global_feat, global_feat)

        N, N = dist_mat.size()
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        dist_pos = dist_mat[is_pos].contiguous().view(N, -1)
        weights_pos = F.softmax(dist_pos, dim=1)
        dist_pos = (weights_pos * dist_pos).sum(dim=1)

        dist_neg = dist_mat[is_neg].contiguous().view(N, -1)
        weights_neg = F.softmax(-dist_neg, dim=1)
        dist_neg = (weights_neg * dist_neg).sum(dim=1)

        y = torch.ones_like(dist_pos)
        loss = self.ranking_loss(dist_neg, dist_pos, y)

        return loss, dist_pos, dist_neg


class TriSoftPlusLoss(object):
    def __init__(self, margin=0.0, dist_func=None):
        self.margin = margin

        self.dist_func = euclidean_dist if dist_func is None else dist_func
        print(f"Using {self.dist_func.__name__} as distance function.")

    def __call__(self, global_feat, labels, attentions=None):
        dist_mat = self.dist_func(global_feat, global_feat, attentions, attentions)
        # _, dist_neg, _, n_inds = hard_example_mining(dist_mat, labels, return_inds=True)

        N, N = dist_mat.size()
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        dist_pos = dist_mat[is_pos].contiguous().view(N, -1)
        weights_pos = F.softmax(dist_pos, dim=1)
        dist_pos = (weights_pos * dist_pos).sum(dim=1)

        dist_neg = dist_mat[is_neg].contiguous().view(N, -1)
        weights_neg = F.softmax(-dist_neg, dim=1)
        dist_neg = (weights_neg * dist_neg).sum(dim=1)

        loss = (1. + (dist_pos - dist_neg + self.margin).exp()).log().mean()

        return loss, dist_pos, dist_neg


class TriHardPlusLoss(object):
    def __init__(self, margin=0.0, dist_func=None):
        self.margin = margin

        self.dist_func = euclidean_dist if dist_func is None else dist_func
        print(f"Using {self.dist_func.__name__} as distance function.")

    def __call__(self, global_feat, labels, attentions=None):
        dist_mat = self.dist_func(global_feat, global_feat, attentions, attentions)
        dist_ap, dist_an, p_inds, n_inds = hard_example_mining(dist_mat, labels, return_inds=True)

        Xp, Xn = global_feat[p_inds], global_feat[n_inds]
        Xp2_Xn2 = Xp.pow(2).sum(dim=1) - Xn.pow(2).sum(dim=1)
        XaXn = (global_feat * Xn).sum(dim=1)
        XaXp = (global_feat * Xp).sum(dim=1)

        dist_ap_dist_an = (Xp2_Xn2 + 2. * XaXn - 2. * XaXp) / (dist_ap + dist_an)

        loss = (1. + (dist_ap_dist_an + self.margin).exp()).log().mean()

        return loss, dist_ap, dist_an


class TriHardPlusLossEnhanced(object):
    def __init__(self, margin=0.0, dist_func=None):
        self.margin = margin

        self.dist_func = euclidean_dist if dist_func is None else dist_func
        print(f"Using {self.dist_func.__name__} as distance function.")

    def __call__(self, global_featA, global_featB, labelsA, labelsB):
        dist_mat = self.dist_func(global_featA, global_featB)
        dist_ap, dist_an = self.hard_example_mining(dist_mat, labelsA, labelsB)

        dist_ap_dist_an = dist_ap - dist_an

        loss = (1. + (dist_ap_dist_an + self.margin).exp()).log().mean()

        return loss, dist_ap, dist_an

    @staticmethod
    def hard_example_mining(dist_mat, labelsA, labelsB):
        M, N = dist_mat.size()

        is_pos = labelsA.expand(N, M).t().eq(labelsB.expand(M, N))
        is_neg = labelsA.expand(N, M).t().ne(labelsB.expand(M, N))

        # `dist_ap` means distance(anchor, positive)
        dist_ap, relative_p_inds = torch.max(dist_mat * is_pos.to(torch.float32), 1, keepdim=True)
        # `dist_an` means distance(anchor, negative)
        dist_an, relative_n_inds = torch.min(dist_mat + is_pos.to(torch.float32) * 1e3, 1, keepdim=True)
        # shape [N]
        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)

        return dist_ap, dist_an


class MMDLoss(object):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul

    def guassian_kernel(self, source, target, fix_sigma=None):
        n_samples = source.size(0) + target.size(0)
        concat = torch.cat([source, target], dim=0)
        l2_distance = euclidean_dist(concat, concat)

        bandwidth = fix_sigma if fix_sigma else torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def __call__(self, source, target, fix_sigma=None):
        batch_size = source.size(0)
        kernels = self.guassian_kernel(source, target, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class MMDLoss2(object):
    import functools

    def __call__(self, x, y, sigmas=(1, 5, 10), normalize=False):
        """Maximum Mean Discrepancy with several Gaussian kernels."""
        # Flatten:
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)

        if len(sigmas) == 0:
            mean_dist = torch.mean(torch.pow(torch.pairwise_distance(x, y, p=2), 2))
            factors = (-1 / (2 * mean_dist)).view(1, 1, 1)
        else:
            factors = self._get_mmd_factor(sigmas, x.device)

        if normalize:
            x = F.normalize(x, p=2, dim=1)
            y = F.normalize(y, p=2, dim=1)

        xx = torch.pairwise_distance(x, x, p=2) ** 2
        yy = torch.pairwise_distance(y, y, p=2) ** 2
        xy = torch.pairwise_distance(x, y, p=2) ** 2

        div = 1 / (x.shape[1] ** 2)

        k_xx = div * torch.exp(factors * xx).sum(0).squeeze()
        k_yy = div * torch.exp(factors * yy).sum(0).squeeze()
        k_xy = div * torch.exp(factors * xy).sum(0).squeeze()

        mmd_sq = torch.sum(k_xx) - 2 * torch.sum(k_xy) + torch.sum(k_yy)
        return torch.sqrt(mmd_sq)

    @functools.lru_cache(maxsize=1, typed=False)
    def _get_mmd_factor(self, sigmas, device):
        sigmas = torch.tensor(sigmas)[:, None, None].to(device).float()
        sigmas = -1 / (2 * sigmas)
        return sigmas
