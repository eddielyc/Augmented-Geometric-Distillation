# -*- coding: utf-8 -*-
# Time    : 2020/7/31 16:54
# Author  : Yichen Lu

import torch
from scipy.linalg import null_space
from torch.nn import functional as F
from .utils import euclidean_dist


def cosine_criterion(embeddings_a, embeddings_b, *args):
    return F.cosine_embedding_loss(embeddings_a, embeddings_b,
                                   torch.ones(embeddings_a.size(0)).to(embeddings_a.device))


class CosineLoss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, outputs_a, outputs_b, *args):
        return cosine_criterion(outputs_a["embedding"], outputs_b["embedding"])


class L1Loss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, outputs_a, outputs_b, *args):
        return (outputs_a["embedding"] - outputs_b['embedding']).abs().sum(dim=1).mean()


class L2Loss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, outputs_a, outputs_b, *args):
        return (outputs_a["embedding"] - outputs_b['embedding']).pow(2).sum(dim=1).mean()


def pairwise_residual(embeddings_a, embeddings_b, targets, return_flags=False):
    B, *_ = embeddings_a.size()

    residual_a = embeddings_a.unsqueeze(dim=0) - embeddings_a.unsqueeze(dim=1)
    residual_b = embeddings_b.unsqueeze(dim=0) - embeddings_b.unsqueeze(dim=1)

    is_pos = targets.expand(B, B).t().eq(targets.expand(B, B))
    assert is_pos.sum(dim=1).to(torch.int).unique().size(0) == 1, "No CK sampler."
    is_pos = is_pos.triu_(diagonal=1).to(torch.bool)

    if return_flags:
        return residual_a[is_pos], residual_b[is_pos], is_pos
    return residual_a[is_pos], residual_b[is_pos]


def triangle_criterion(embeddings_a, embeddings_b, targets, *args):
    embedding_distillation_loss = cosine_criterion(embeddings_a, embeddings_b)

    residual_a, residual_b = pairwise_residual(embeddings_a, embeddings_b, targets)
    residual_cos = (F.normalize(residual_a, p=2, dim=1) * F.normalize(residual_b, p=2, dim=1)).sum(dim=1)

    residual_distillation_loss = 1. - residual_cos.mean()

    return embedding_distillation_loss + 0.5 * residual_distillation_loss


class NoDistillationLoss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, *args, **kwargs):
        return 0


class ClassificationDistillationLoss(object):
    def __init__(self, config):
        self.T = config["T"]

    def __call__(self, outputs, outputs_target, *args):
        logits, logits_target = outputs['preds'], outputs_target['preds']
        B, C = logits_target.size()
        logits = logits[:, -C:]
        logits_hat = logits / self.T
        logits_target_hat = logits_target.detach() / self.T
        p_hat = F.softmax(logits_hat, dim=1)
        p_target_hat = F.softmax(logits_target_hat, dim=1)
        distillation_loss = F.kl_div(p_hat.log(), p_target_hat, reduction="batchmean")
        return distillation_loss


class MetricDistillationLoss(object):
    def __init__(self, config):
        self.T = config["T"]

    def __call__(self, outputs, outdated_outputs, memory_bank=None, outdated_memory_bank=None):
        embeddings, outdated_embeddings = outputs["embedding"], outdated_outputs["embedding"]
        memory_bank = embeddings if memory_bank is None else torch.cat([embeddings, memory_bank], dim=0)
        outdated_memory_bank = outdated_embeddings if outdated_memory_bank is None else torch.cat(
            [outdated_embeddings, outdated_memory_bank], dim=0)

        dist = euclidean_dist(embeddings, memory_bank)
        dist_hat = dist / self.T
        dist_hat = dist_hat + torch.zeros_like(dist_hat).fill_diagonal_(float('inf'))
        outdated_dist = euclidean_dist(outdated_embeddings, outdated_memory_bank)
        outdated_dist_hat = outdated_dist / self.T
        outdated_dist_hat = outdated_dist_hat + torch.zeros_like(outdated_dist_hat).fill_diagonal_(float('inf'))

        distribution = F.softmin(dist_hat, dim=1).clamp_min(1e-8)
        outdated_distribution = F.softmin(outdated_dist_hat, dim=1).clamp_min(1e-8)
        distillation_loss = (-1. * outdated_distribution * distribution.log()).sum(dim=1).mean()
        return distillation_loss


class PoolingDistillationLoss(object):
    def __init__(self, config):
        collapse, criterion = config["collapse"], config["criterion"]
        self.embedding_factor, self.attention_factor = config["embedding_factor"], config["attention_factor"]

        self.collapse_func = {"spatial": self.spatial_pooling,
                              "channel": self.channel_pooling,
                              "vertical": self.vertical_pooling,
                              "horizon": self.horizon_pooling,
                              "global": self.global_pooling,
                              "none": self.no_pooling}[collapse]

        self.criterion = {"cosine": cosine_criterion,
                          "triangle": TriangleDistillationLoss(),
                          "res-triangle": ResTriangleDistillationLoss(),
                          "sim-res-triangle": SimResTriangleDistillationLoss()}[criterion]

    def __call__(self, outputs_a, outputs_b, targets=None):
        embedding_a, embedding_b = outputs_a['embedding'], outputs_b['embedding'].detach()
        embedding_distillation_loss = F.cosine_embedding_loss(embedding_a, embedding_b,
                                                              torch.ones(embedding_a.size(0)).to(embedding_a.device))

        feature_maps_a, feature_maps_b = [attention.pow(2) for attention in outputs_a['maps']], \
                                         [activation_map.pow(2) for activation_map in outputs_b['maps']]
        feature_map_distillation_losses = [self.pooling_loss(feature_map_a, feature_map_b, targets)
                                           for feature_map_a, feature_map_b in zip(feature_maps_a, feature_maps_b)]
        feature_map_distillation_loss = sum(feature_map_distillation_losses) / len(feature_map_distillation_losses)

        return self.embedding_factor * embedding_distillation_loss + \
               self.attention_factor * feature_map_distillation_loss

    def pooling_loss(self, map_a, map_b, targets=None):
        pooling_vector_a = F.normalize(self.collapse_func(map_a), p=2, dim=1)
        pooling_vector_b = F.normalize(self.collapse_func(map_b), p=2, dim=1)
        return self.criterion(pooling_vector_a, pooling_vector_b, targets)

    @staticmethod
    def channel_pooling(feature_map):
        assert len(feature_map.size()) == 4
        bs = feature_map.size(0)
        return feature_map.sum(dim=1).view(bs, -1)

    @staticmethod
    def vertical_pooling(feature_map):
        assert len(feature_map.size()) == 4
        bs = feature_map.size(0)
        return feature_map.sum(dim=2).view(bs, -1)

    @staticmethod
    def horizon_pooling(feature_map):
        assert len(feature_map.size()) == 4
        bs = feature_map.size(0)
        return feature_map.sum(dim=3).view(bs, -1)

    @staticmethod
    def global_pooling(feature_map):
        assert len(feature_map.size()) == 4
        bs = feature_map.size(0)
        return feature_map.mean(dim=(2, 3)).view(bs, -1)

    @staticmethod
    def spatial_pooling(feature_map):
        assert len(feature_map.size()) == 4
        bs = feature_map.size(0)
        return torch.cat([feature_map.sum(dim=2).view(bs, -1), feature_map.sum(dim=3).view(bs, -1)], dim=1)

    @staticmethod
    def no_pooling(feature_map):
        assert len(feature_map.size()) == 4
        bs = feature_map.size(0)
        return feature_map.view(bs, -1)


class RelativeDistancesLoss(object):
    """Distillation loss between the teacher and the student comparing distances
        instead of embeddings.
        Reference:
            * Lu Yu et al.
              Learning Metrics from Teachers: Compact Networks for Image Embedding.
              CVPR 2019.
        :param embeddings_a: ConvNet embeddings of a model.
        :param embeddings_b: ConvNet embeddings of a model.
        :return: A float scalar loss.
        """

    def __init__(self, config):
        self.normalize = config.get("normalize", False)
        self.p = config.get("p", 2)

    def __call__(self, outputs_a, outputs_b, *args):
        embeddings_a, embeddings_b = outputs_a["embedding"], outputs_b["embedding"]
        if self.normalize:
            embeddings_a = F.normalize(embeddings_a, dim=-1, p=2)
            embeddings_b = F.normalize(embeddings_b, dim=-1, p=2)

        pairwise_distances_a = torch.pdist(embeddings_a, p=self.p)
        pairwise_distances_b = torch.pdist(embeddings_b, p=self.p)

        return (pairwise_distances_a - pairwise_distances_b).abs().mean()


class GradCAMLoss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, activation_s, grads_s, activation_t, grads_t):
        attention_s = F.adaptive_avg_pool2d(grads_s, (1, 1)) * activation_s
        # attention_s = activation_s
        # attention_s = F.relu(attention_s.sum(dim=1))
        attention_s = F.relu(attention_s)
        attention_t = F.adaptive_avg_pool2d(grads_t, (1, 1)) * activation_t
        # attention_t = activation_t
        # attention_t = F.relu(attention_t.sum(dim=1))
        attention_t = F.relu(attention_t)

        normalized_s = F.normalize(attention_s.view(attention_s.size(0), -1), p=2, dim=1)
        normalized_t = F.normalize(attention_t.view(attention_t.size(0), -1), p=2, dim=1)

        loss = torch.abs(normalized_s - normalized_t).sum(dim=1).mean()
        return loss


class TriangleDistillationLoss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, outputs_a, outputs_b, targets):
        embeddings_a, embeddings_b = outputs_a["embedding"], outputs_b["embedding"]
        return triangle_criterion(embeddings_a, embeddings_b, targets)


class ResTriangleDistillationLoss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, outputs_a, outputs_b, targets):
        embeddings_a, embeddings_b = outputs_a["embedding"], outputs_b["embedding"]
        residual_a, residual_b, is_pos = pairwise_residual(embeddings_a, embeddings_b, targets, return_flags=True)

        return triangle_criterion(residual_a,
                                  residual_b,
                                  targets.expand(targets.size(0), targets.size(0))[is_pos]
                                  )


class SimResTriangleDistillationLoss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, outputs_a, outputs_b, targets):
        embeddings_a, embeddings_b = outputs_a["embedding"], outputs_b["embedding"]
        return cosine_criterion(*pairwise_residual(embeddings_a, embeddings_b, targets))


class SimilitudeDistillationLoss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, outputs_a, outputs_b, pids):
        embeddings_a, embeddings_b = outputs_a["embedding"], outputs_b["embedding"]
        B, *_ = embeddings_a.size()
        is_pos = pids.expand(B, B).t().eq(pids.expand(B, B))
        dist_s = euclidean_dist(embeddings_a, embeddings_a)
        dist_t = euclidean_dist(embeddings_b, embeddings_b)
        scales = (dist_s / dist_t)[is_pos.triu_(diagonal=1).to(torch.bool)].view(pids.unique().size(0), -1)

        # scales_prob = scales.softmax(dim=1)
        # loss = (scales_prob * scales_prob.log()).sum(dim=1).mean()

        loss = scales.var(dim=1).mean()
        return loss


def sqrt_newton_schulz_autograd(A, numIters=2):
    # batchSize = A.data.shape[0]
    dim = A.data.shape[0]
    normA = A.mul(A).sum(dim=0).sum(dim=0).sqrt()
    Y = A.div(normA.view(1, 1).expand_as(A))
    I = torch.eye(dim, dim).double().cuda()
    Z = torch.eye(dim, dim).double().cuda()

    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)
    sA = Y * torch.sqrt(normA).expand_as(A)
    return sA


class GFK(object):
    def __init__(self, config):
        self.dim = config["dim"]
        self.eps = 1e-2

    def decompose(self, inputs, subspace_dim=2):
        square_mat = inputs.transpose(0, 1).mm(inputs)
        sA_minushalf = self.sqrt_newton_schulz_minus(square_mat, numIters=1)
        ortho_mat = inputs.double().mm(sA_minushalf)
        ortho_mat = ortho_mat.float()

        return ortho_mat[:, :subspace_dim]

    @staticmethod
    def train_pca_tall(data, subspace_dim):
        """
        Modified PCA function, different from the one in sklearn
        :param data: data matrix
        :param subspace_dim: dim
        :return: a wrapped machine object
        """

        data2 = data - data.mean(0)
        uu, ss, vv = torch.svd(data2.float())
        subspace = uu[:, :subspace_dim]

        return subspace

    @staticmethod
    def sqrt_newton_schulz_minus(A, numIters=1):
        # batchSize = A.data.shape[0]
        A = A.double()
        dim = A.data.shape[0]
        normA = A.mul(A).sum(dim=0).sum(dim=0).sqrt()
        Y = A.div(normA.view(1, 1).expand_as(A))
        I = torch.eye(dim, dim).double().cuda()
        Z = torch.eye(dim, dim).double().cuda()

        # A.register_hook(print)
        for i in range(numIters):
            T = 0.5 * (3.0 * I - Z.mm(Y))
            Y = Y.mm(T)
            Z = T.mm(Z)

        sZ = Z * 1. / torch.sqrt(normA).expand_as(A)
        return sZ

    def fit(self, input1, input2):
        """
        Obtain the kernel G
        :param input1: ns * n_feature, source feature
        :param input2: nt * n_feature, target feature
        :return: GFK kernel G
        """

        input1 = F.normalize(input1, dim=-1, p=2)
        input2 = F.normalize(input2, dim=-1, p=2)

        source_dim = input1.size(0) - 2  # min(64, input1.size(0)-2)#input1.size(0)//2#-2
        target_dim = input2.size(0) - 2  # min(64, input2.size(0)-2)#16#input2.size(0)//2#-2
        num_nullspacedim = 60

        # PSRS
        # source = supportset.contiguous().view(-1, supportset.size(-1))#[:source_dim]
        source = input1
        Ps = self.train_pca_tall(source.t(), subspace_dim=source_dim)  # .detach()
        Rs = torch.from_numpy(null_space(Ps.t().cpu().detach().numpy())[:, :num_nullspacedim]).cuda()
        # adding columns
        Ps = torch.cat([Ps, Rs], dim=1)
        N = Ps.shape[1]  # L = NxK shot - 1

        target = input2
        Pt = self.train_pca_tall(target.t(), subspace_dim=target_dim)

        # Pt.register_hook(print)

        G = self.gfk_G(Ps, Pt, N, source_dim, target_dim).detach().float()

        # G.register_hook(print)
        # sqrG = self.sqrt_newton_schulz_autograd(G.double(), numIters=1).float()
        # qq = query #- supportset[ii].mean(0)
        # meann = supportset[ii].mean(0)

        qq1 = input1 - input2
        # qq1_norm = input1/(torch.norm(input1, dim=-1, keepdim=True) + 1e-18)
        # qq2_norm = input2/(torch.norm(input2, dim=-1, keepdim=True) + 1e-18)
        projected_qq = G.t().mm(qq1.t()).t()

        projected_qq_norm = projected_qq  # /(torch.norm(projected_qq, dim=-1, keepdim=True) + 1e-18)
        # loss =  torch.sum((qq1_norm) * projected_qq_norm, dim=-1)
        loss = torch.sum((qq1) * projected_qq_norm, dim=-1)  # *1e-1
        # ones = torch.ones_like(loss).cuda()
        loss_kd = loss.mean()  # torch.mean(ones - loss)#loss.mean()#
        # new_query_dist =torch.sqrt(torch.sum(new_query_dist*new_query_dist, dim=-1) + 1e-10)#torch.sum(new_query_dist*new_query_dist, dim=-1)#*0.2#

        return loss_kd

    def gfk_G(self, Ps, Pt, N, source_dim, target_dim):
        A = Ps[:, :source_dim].t().mm(Pt)  # QPt[:source_dim, :]#.copy()
        B = Ps[:, source_dim:].t().mm(Pt)  # QPt[source_dim:, :]#.copy()

        ######## GPU #############

        UU, SS, VV = self.HOGSVD_fit([A, B])
        # SS.register_hook(print)
        V1, V2, V, Gam, Sig = UU[0], UU[1], VV, SS[0], SS[1]
        V2 = -V2

        Gam = Gam.clamp(min=-1., max=1.)
        theta = torch.acos(Gam)  # + 1e-5

        B1 = torch.diag(0.5 * (1 + (torch.sin(2 * theta) / (2. * theta + 1e-12))))
        B2 = torch.diag(0.5 * (torch.cos(2 * theta) - 1) / (2 * theta + 1e-12))
        B3 = B2
        B4 = torch.diag(0.5 * (1. - (torch.sin(2. * theta) / (2. * theta + 1e-12))))

        delta1_1 = torch.cat((V1, torch.zeros((N - source_dim, target_dim)).cuda()),
                             dim=0)  # np.hstack((V1, torch.zeros((dim, N - dim))))
        delta1_2 = torch.cat((torch.zeros((source_dim, target_dim)).cuda(), V2),
                             dim=0)  # np.hstack((np.zeros(shape=(N - dim, dim)), V2))

        delta1 = torch.cat((delta1_1, delta1_2), dim=1)

        delta2_1 = torch.cat((B1, B3), dim=0)  # c
        delta2_2 = torch.cat((B2, B4), dim=0)  #
        delta2 = torch.cat((delta2_1, delta2_2), dim=1)

        delta3_1 = torch.cat((V1.t(), torch.zeros((target_dim, source_dim)).cuda()),
                             dim=0)  # np.hstack((V1, np.zeros(shape=(dim, N - dim))))
        delta3_2 = torch.cat((torch.zeros((target_dim, N - source_dim)).cuda(), V2.t()),
                             dim=0)  # np.hstack((np.zeros(shape=(N - dim, dim)), V2))
        delta3 = torch.cat((delta3_1, delta3_2), dim=1)  # .t()  # np.vstack((delta3_1, delta3_2)).T

        mm_delta = torch.matmul(delta1, delta2)

        delta = torch.matmul(mm_delta, delta3)
        G = torch.matmul(torch.matmul(Ps, delta), Ps.t()).float()

        return G

    ############################## HOGSVD #########################
    def inverse(self, X):
        eye = torch.diag(torch.randn(X.shape[0]).cuda()).double() * self.eps
        # X = X + eye
        # A = torch.inverse(X)
        Z = self.sqrt_newton_schulz_minus(X.double(), numIters=1).float()
        A = Z.mm(Z)  ## inverse
        # A[0].register_hook(A)
        return A.float()

    def HOGSVD_fit_S(self, X):
        N = len(X)
        data_shape = X[0].shape
        # eye = torch.diag(torch.randn(data_shape[1]).cuda()) * self.eps
        # A = [x.T.dot(x) for x in X]
        A = [torch.matmul(x.transpose(0, 1), x).float().cuda() for x in X]
        A_inv = [self.inverse(a.double()).float().cuda() for a in A]
        S = torch.zeros((data_shape[1], data_shape[1])).float().cuda()
        for i in range(N):
            for j in range(i + 1, N):
                S = S + (torch.matmul(A[i], A_inv[j]) + torch.matmul(A[j], A_inv[i]))
        S = S / (N * (N - 1))
        # S.register_hook(print)
        return S

    def _eigen_decompostion(self, X, subspace_dim):
        V, eigen_values, V_t = torch.svd(X.double())
        # V = self.decompose(X.t(), subspace_dim=subspace_dim)
        return V.float()

    def HOGSVD_fit_B(self, X, V):
        X = [x.float().cuda() for x in X]
        # V.register_hook(print)
        V_inv = V.t()  # V_inv is its transpose #torch.inverse(V).float()#self.inverse(V).float()  # torch.inverse(V)
        # V_inv.register_hook(print)
        B = [torch.matmul(V_inv, x.transpose(0, 1)).transpose(0, 1) for x in X]
        # B[0].register_hook(print)
        return B

    def HOGSVD_fit_U_Sigma(self, B):
        B = [b for b in B]
        sigmas = torch.stack([torch.norm(b, dim=0) for b in B])
        # B[0].register_hook(print)
        U = [b / (sigma) for b, sigma in zip(B, sigmas)]

        return sigmas, U

    def HOGSVD_fit(self, X):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : array-like, shape (n_samples, (n_rows_i, n_cols)
            List of training input samples. Eah input element has
            the same numbe of columns but can have unequal number of rows.
        Returns
        -------
        self : object
            Returns self.
        """

        X = [x for x in X]

        # Step 1: Calculate normalized S
        S = self.HOGSVD_fit_S(X).float()
        # S.register_hook(print)

        V = self._eigen_decompostion(S, S.size(0))

        B = self.HOGSVD_fit_B(X, V)

        sigmas, U = self.HOGSVD_fit_U_Sigma(B)

        return U, sigmas, V

    def __call__(self, outputs_a, outputs_b, *args):
        embeddings_a, embeddings_b = outputs_a["embedding"], outputs_b["embedding"]
        return self.fit(embeddings_a, embeddings_b)


class GeoDLxLUCIR(object):
    def __init__(self, config):
        self.config = config
        self.geoDL_criterion = GFK(config)

    def __call__(self, outputs_a, outputs_b, *args):
        embeddings_a, embeddings_b = outputs_a["embedding"], outputs_b["embedding"]
        geoDL_loss = self.geoDL_criterion.fit(embeddings_a, embeddings_b)
        cosine_loss = cosine_criterion(embeddings_a, embeddings_b)
        loss = self.config["GeoDL_factor"] * geoDL_loss + self.config["cosine_factor"] * cosine_loss
        return loss


class AlwaysBeDreaming(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, outputs_s, outputs_t, *args):
        outdated_preds, preds = outputs_s["outdated_preds"], outputs_t["preds"]
        loss = F.mse_loss(outdated_preds, preds)
        return loss


factory = {
    "cosine": CosineLoss,
    "triangle": TriangleDistillationLoss,
    "res-triangle": ResTriangleDistillationLoss,
    "sim-res": SimResTriangleDistillationLoss,
    "no": NoDistillationLoss,
    "hinton": ClassificationDistillationLoss,
    "metric": MetricDistillationLoss,
    "podnet": PoolingDistillationLoss,
    "relation": RelativeDistancesLoss,
    "GradCAM": GradCAMLoss,
    "similitude": SimilitudeDistillationLoss,
    "GeoDL": GFK,
    "l1": L1Loss,
    "l2": L2Loss,
    "GeoDLxLUCIR": GeoDLxLUCIR,
    "always-be-dreaming": AlwaysBeDreaming,
}
