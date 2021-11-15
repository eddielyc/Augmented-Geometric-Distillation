import time
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import numpy as np

from reid.evaluation import cmc, mean_ap, map_cmc
from reid.utils.meters import AverageMeter, AverageMeters
from reid.loss import euclidean_dist
from reid.loss.utils import cos


def extract_cnn_feature(model, inputs, output_feature='embedding', device=None, **kwargs):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    inputs = inputs.to(device)
    outputs_origin = model(inputs, **kwargs)[output_feature]
    # outputs_flip = model(inputs.flip(dims=[3]), output_feature=output_feature, **kwargs)
    # return [(output_origin + output_flip) / 2 for output_origin, output_flip in zip(outputs_origin, outputs_flip)]
    return outputs_origin


def extract_features(model, data_loader, output_feature="embedding", print_freq=1, device=None,  **kwargs):
    model.eval()
    meters = AverageMeters(AverageMeter("Batch Time"),
                           AverageMeter("Data Time"),
                           )

    features = OrderedDict()

    start = time.time()

    with torch.no_grad():
        for i, (imgs, fnames, pids, *_) in enumerate(data_loader):
            data_time = time.time() - start

            outputs = extract_cnn_feature(model, imgs, output_feature, device=device, **kwargs).detach().cpu()
            for index, (fname, pid) in enumerate(zip(fnames, pids)):
                features[fname] = outputs[index]

            batch_time = time.time() - start
            start = time.time()
            meters.update([batch_time, data_time])

            if (i + 1) % print_freq == 0:
                print(f'Extract Features: [{i + 1}/{len(data_loader)}], ' + meters())

    return features


def pairwise_distance(query_features, gallery_features, query=None, gallery=None, re_ranking=False, dist_type='cos'):
    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _, *_ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _, *_ in gallery], 0)

    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    if dist_type.lower() == 'euc':
        print("compute euc dist")
        dist = euclidean_dist(x, y)
    elif dist_type.lower() == 'cos':
        print("compute cos dist")
        dist = cos(x, y)
        # because argsort function takes ascending order as default
        dist = 1.0 / (dist + 1.0)
    else:
        raise RuntimeError("dist type must be one of 'euc' or 'cos'.")

    if re_ranking:
        dist = re_ranking_dist(x, y, k1=20, k2=6, lambda_value=0.3)
    else:
        dist = dist.numpy()

    return dist


def re_ranking_dist(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea,galFea])
        print('using GPU to compute original distance')
        distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
                      torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(1,-2,feat,feat.t())
        original_dist = distmat.cpu().numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


class Evaluator(object):
    def __init__(self, model):
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery, output_feature='embedding', re_ranking=False,
                 dist_func=pairwise_distance, dist_type='cos', print_freq=1, step=1024, cmc_topk=(1, 5, 10, 20),
                 device=None, **kwargs):
        query_features = extract_features(self.model, query_loader, output_feature, print_freq=print_freq, device=device, **kwargs)

        print('Finish extracting feature of queries.')
        gallery_features = extract_features(self.model, gallery_loader, output_feature, print_freq=print_freq, device=device, **kwargs)
        print('Finish extracting feature of galleries.')
        print('Calculating pairwise distance...')

        mAP, all_cmc = self.step_evaluate(query_features, gallery_features,
                                          query=query, gallery=gallery, step=step, dist_func=dist_func,
                                          dist_type=dist_type, re_ranking=re_ranking)

        self.print_results(mAP, all_cmc, cmc_topk)

        return mAP, [all_cmc[k - 1] for k in cmc_topk]

    @staticmethod
    def step_evaluate(query_features, gallery_features, query, gallery,
                      step=1024, dist_func=pairwise_distance, dist_type='cos', re_ranking=False):
        aps_collecter, cumsum_collecter, total_valid_queries = [], [], 0
        for i in range(len(query_features) // step + 1):
            partial_query = query[i * step: (i + 1) * step]
            distmat = dist_func(query_features, gallery_features, partial_query, gallery, re_ranking,
                                dist_type=dist_type)

            aps, cumsum, num_valid_queries = map_cmc(distmat,
                                                     query_ids=[pid for _, pid, _, *_ in partial_query],
                                                     gallery_ids=[pid for _, pid, _, *_ in gallery],
                                                     query_cams=[cam for _, _, cam, *_ in partial_query],
                                                     gallery_cams=[cam for _, _, cam, *_ in gallery])

            aps_collecter.append(aps)
            cumsum_collecter.append(cumsum)
            total_valid_queries += num_valid_queries

        mAP = np.concatenate(aps_collecter, axis=0).mean()
        all_cmc = sum(cumsum_collecter) / total_valid_queries

        return mAP, all_cmc

    @staticmethod
    def print_results(mAP, all_cmc, cmc_topk):
        print('Mean AP: {:4.1%}'.format(mAP))
        print('CMC Scores')
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'.format(k, all_cmc[k - 1]))


class IncrementalEvaluator(Evaluator):
    def __init__(self, model):
        super(IncrementalEvaluator, self).__init__(model=model)

    def evaluate(self, query_loaders, gallery_loader, queries, gallery, output_feature='embedding', re_ranking=False,
                 dist_func=pairwise_distance, dist_type='cos', print_freq=1, step=1024, cmc_topk=(1, 5, 10, 20),
                 device=None, **kwargs):
        queries_features = {name: extract_features(self.model, query_loader, output_feature, print_freq=print_freq, device=device, **kwargs)
                            for name, query_loader in query_loaders.items()}
        print('Finish extracting feature of queries.')
        gallery_features = extract_features(self.model, gallery_loader, output_feature, print_freq=print_freq, device=device, **kwargs)
        print('Finish extracting feature of galleries.')
        print('Calculating pairwise distance...')

        mAPs, all_cmcs = OrderedDict(), OrderedDict()
        for name, query_features in queries_features.items():
            mAP, all_cmc = self.step_evaluate(query_features, gallery_features,
                                              query=queries[name], gallery=gallery, step=step, dist_func=dist_func,
                                              dist_type=dist_type, re_ranking=re_ranking)
            mAPs[name], all_cmcs[name] = mAP, all_cmc
            self.print_results(mAP, all_cmc, cmc_topk)

        mAP, all_cmc = sum(mAPs.values()) / len(mAPs), sum(all_cmcs.values()) / len(all_cmcs)
        mAPs['avg'], all_cmcs['avg'] = mAP, all_cmc
        self.print_results(mAP, all_cmc, cmc_topk)

        return mAPs, {name: [cmc[k - 1] for k in cmc_topk] for name, cmc in all_cmcs.items()}
