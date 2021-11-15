# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2020 paper
# Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion
# Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek
# Hoiem, Niraj K. Jha, and Jan Kautz
# --------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import collections
import random
import torch
import numpy as np
import pickle
import time

from .utils import lr_cosine_policy, lr_policy, clip, denormalize, InversionSampler
from .loss import prior_losses, negative_js_divergence, GravityCriterion, CrossEntropyCriterion
from reid import TriHardPlusLoss
from reid.utils.meters import AverageMeter, AverageMeters


class DeepInversionFeatureHook(object):
    """
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """

    def __init__(self, module):
        self.r_feature = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = inputs[0].shape[1]
        mean = inputs[0].mean([0, 2, 3])
        var = inputs[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


class DeepInversionFeatureHook1D(object):
    """
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """

    def __init__(self, module):
        self.r_feature = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = inputs[0].shape[1]
        mean = inputs[0].mean([0])
        var = inputs[0].permute(1, 0).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


class DeepInversionClass(object):
    def __init__(self, bs=64,
                 net_teacher=None,
                 net_student=None,
                 coefficients=None,
                 network_output_function=lambda x: x,
                 hook_for_display=None):
        """
        :param bs: batch size per GPU for image generation
        :parameter net_teacher: Pytorch model to be inverted
        :param coefficients: dictionary with parameters and coefficients for optimization.
            keys:
            "r_feature" - coefficient for feature distribution regularization
            "tv_l1" - coefficient for total variation L1 loss
            "tv_l2" - coefficient for total variation L1 loss
            "l2" - l2 penalization weight
            "lr" - learning rate for optimization
            "main_loss_multiplier" - coefficient for the main loss optimization
            "adi_scale" - coefficient for Adaptive DeepInversion, competition, def =0 means no competition
        network_output_function: function to be applied to the output of the network to get the output
        hook_for_display: function to be executed at every print/save call, useful to check accuracy of verifier
        """

        self.net_teacher = net_teacher
        self.net_student = net_student

        self.bs = bs  # batch size
        self.print_every = 10
        self.network_output_function = network_output_function

        self.bn_reg_scale = coefficients.r_feature
        self.first_bn_multiplier = coefficients.first_bn_multiplier
        self.var_scale_l1 = coefficients.tv_l1
        self.var_scale_l2 = coefficients.tv_l2
        self.l2_scale = coefficients.l2
        self.lr = coefficients.lr
        self.main_loss_multiplier = coefficients.main_loss_multiplier
        self.adi_scale = coefficients.adi_scale

        self.num_generations = 0

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = CrossEntropyCriterion()
        self.trip_criterion = TriHardPlusLoss()

        # Create hooks for feature statistics
        self.w = self.net_teacher.classifier.W
        self.bn_mean = self.net_teacher.feature_extractor.bn.running_mean
        self.bn_var = self.net_teacher.feature_extractor.bn.running_var
        self.bn_weight = self.net_teacher.feature_extractor.bn.weight
        self.bn_bias = self.net_teacher.feature_extractor.bn.bias

        self.loss_r_feature_layers = []
        self.register_hooks()

        self.hook_for_display = None
        if hook_for_display is not None:
            self.hook_for_display = hook_for_display

        self.meters = AverageMeters(AverageMeter("Total Loss"),
                                    AverageMeter("R Loss"),
                                    AverageMeter("CE Loss"),
                                    AverageMeter("Prior Loss"),
                                    AverageMeter("Accuracy"))
        if self.net_student and self.adi_scale:
            self.meters.append(AverageMeter("JS Loss"))

    def get_images(self, targets=None, preprocessor=None, starts=None):
        self.register_hooks()
        self.net_teacher.requires_grad_(True)
        best_cost = np.inf

        inputs = torch.randn((targets.size(0), 3, 256, 128), requires_grad=True,
                             device='cuda') if starts is None else torch.tensor(starts.cuda(), requires_grad=True,
                                                                                device='cuda')
        targets = targets.to('cuda')
        pooling_function = nn.AvgPool2d(kernel_size=2)
        optimizer = optim.Adam([inputs], lr=self.lr, betas=(0.5, 0.9), eps=1e-8)

        for iterations_per_layer, downsampling_ratio in zip([2000, 1000], [2, 1]):

            lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

            for iteration_loc in range(iterations_per_layer):
                # learning rate scheduling
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                # perform downsampling if needed
                inputs_jit = pooling_function(inputs) if downsampling_ratio != 1 else inputs

                inputs_jit = self.augment(inputs_jit, preprocessor)

                # forward pass
                optimizer.zero_grad()
                self.net_teacher.zero_grad()

                preds, outputs = self.net_teacher(inputs_jit)
                preds = self.network_output_function(preds)
                # print(outputs)

                # R_cross classification loss
                ce_loss = self.criterion(preds, targets)

                acc = torch.mean((torch.argmax(preds, dim=1) == targets).float())

                loss_image = self.image_loss(inputs_jit)

                loss_r_feature = self.bn_loss()

                # combining losses
                loss_aux = loss_r_feature + loss_image

                if self.net_student and self.adi_scale:
                    loss_verifier_cig = self.adi_loss(inputs_jit, preds)
                    loss_aux = loss_aux + loss_verifier_cig
                    self.meters.update({"JS Loss": loss_verifier_cig.item()})

                loss = self.main_loss_multiplier * ce_loss + loss_aux
                self.meters.update({"Total Loss": loss.item(),
                                    "R Loss": loss_r_feature.item(),
                                    "CE Loss": ce_loss.item(),
                                    "Prior Loss": loss_image.item(),
                                    "Accuracy": acc.item()})

                if iteration_loc % self.print_every == 0:
                    print(f"[{self.num_generations}][{iteration_loc}/{iterations_per_layer}]: " + self.meters())

                    if self.hook_for_display is not None:
                        self.hook_for_display(inputs, targets)

                # do image update
                loss.backward()
                optimizer.step()

                # clip color outlayers
                inputs.data = clip(inputs.data)

                if best_cost > loss.item() and downsampling_ratio == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
        return denormalize(best_inputs)

    def generate_batch(self, targets=None, preprocessor=None, starts=None):
        generated = self.get_images(targets=targets, preprocessor=preprocessor, starts=starts)
        self.num_generations += 1
        print(f"==> Finish Generation No. {self.num_generations}")
        return generated

    def register_hooks(self):
        if not self.loss_r_feature_layers:
            for module in self.net_teacher.modules():
                if isinstance(module, nn.BatchNorm2d):
                    self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))
            self.loss_r_feature_layers.append(DeepInversionFeatureHook1D(self.net_teacher.feature_extractor.bn))

    def remove_hooks(self):
        for hook in self.loss_r_feature_layers:
            hook.close()
        self.loss_r_feature_layers = []

    @staticmethod
    def augment(inputs, preprocessor=None):
        # Flipping
        flip = random.random() > 0.5
        if flip:
            inputs_jit = torch.flip(inputs, dims=(3,))
        else:
            inputs_jit = inputs

        if preprocessor is not None:
            inputs_jit = preprocessor(inputs_jit)

        return inputs_jit

    def image_loss(self, inputs):
        # R_prior losses
        loss_var_l1, loss_var_l2 = prior_losses(inputs)

        # l2 loss on images
        loss_l2 = torch.norm(inputs.view(self.bs, -1), dim=1).mean()

        return self.var_scale_l2 * loss_var_l2 + \
               self.var_scale_l1 * loss_var_l1 + \
               self.l2_scale * loss_l2

    def bn_loss(self):
        # R_feature loss
        rescales = [self.first_bn_multiplier] + [1. for _ in range(len(self.loss_r_feature_layers) - 1)]
        loss_r_feature = sum([mod.r_feature * rescale for (rescale, mod) in zip(rescales, self.loss_r_feature_layers)])

        return self.bn_reg_scale * loss_r_feature

    def adi_loss(self, inputs, preds_teacher):
        preds_student, outputs_student = self.net_student(inputs)
        preds_student = self.network_output_function(preds_student)
        preds_student = preds_student[..., -preds_teacher.size(1):]

        preds_student = preds_student.softmax(dim=1)
        # entropy = (-preds_student * preds_student.log()).sum(dim=1).mean()
        # loss_verifier_cig = -1. * entropy
        loss_verifier_cig = negative_js_divergence(preds_teacher, preds_student, T=3.)

        return self.adi_scale * loss_verifier_cig


class DeepInversionClassBeta(DeepInversionClass):
    def __init__(self, bs=64, net_teacher=None, net_student=None, coefficients=None,
                 network_output_function=lambda x: x, hook_for_display=None):
        """
        :param bs: batch size per GPU for image generation
        :parameter net_teacher: Pytorch model to be inverted
        :param coefficients: dictionary with parameters and coefficients for optimization.
            keys:
            "r_feature" - coefficient for feature distribution regularization
            "tv_l1" - coefficient for total variation L1 loss
            "tv_l2" - coefficient for total variation L1 loss
            "l2" - l2 penalization weight
            "lr" - learning rate for optimization
            "main_loss_multiplier" - coefficient for the main loss optimization
            "adi_scale" - coefficient for Adaptive DeepInversion, competition, def =0 means no competition
        network_output_function: function to be applied to the output of the network to get the output
        hook_for_display: function to be executed at every print/save call, useful to check accuracy of verifier
        """

        super().__init__(bs, net_teacher, net_student, coefficients, network_output_function, hook_for_display)

        with open('./deep_inversion/distribution.pkl', 'rb') as file:
            H_target = pickle.load(file)
            H_target = torch.tensor(H_target, dtype=torch.float, device='cuda')
        self.hist_cretirion = GravityCriterion(lb=-1., ub=1., R=100, H_target=H_target)

        self.meters = AverageMeters(AverageMeter("Total Loss"),
                                    AverageMeter("R Loss"),
                                    AverageMeter("CE Loss"),
                                    AverageMeter("Prior Loss"),
                                    AverageMeter("Accuracy"),
                                    # AverageMeter("Hist Loss")
                                    )

    def get_images(self, targets=None, preprocessor=None, starts=None):
        self.register_hooks()
        self.net_teacher.requires_grad_(True)
        best_cost = np.inf

        inputs = torch.randn((targets.size(0), 3, 256, 128), requires_grad=True,
                             device='cuda') if starts is None else torch.tensor(starts.cuda(), requires_grad=True,
                                                                                device='cuda')
        targets = targets.to('cuda')
        sampler = iter(InversionSampler(inputs, targets, self.bs))
        pooling_function = nn.AvgPool2d(kernel_size=2)
        optimizer = optim.Adam([inputs], lr=self.lr, betas=(0.5, 0.9), eps=1e-8)
        num_batch = targets.size(0) // self.bs

        for iterations_per_layer, downsampling_ratio in zip([2000 * num_batch, 1000 * num_batch], [2, 1]):

            lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

            for iteration_loc in range(iterations_per_layer):
                # learning rate scheduling
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                # batch_inputs, batch_targets = inputs, targets
                batch_inputs, batch_targets = next(sampler)
                # indices = torch.randint(targets.size(0), (self.bs, ))
                # batch_inputs, batch_targets = inputs[indices], targets[indices]

                inputs_jit = pooling_function(batch_inputs) if downsampling_ratio != 1 else batch_inputs

                inputs_jit = self.augment(inputs_jit, preprocessor)

                # forward pass
                optimizer.zero_grad()
                self.net_teacher.zero_grad()

                preds, outputs = self.net_teacher(inputs_jit)
                preds = self.network_output_function(preds)
                # print(outputs)

                # R_cross classification loss
                ce_loss = self.criterion(preds, batch_targets)

                acc = torch.mean((torch.argmax(preds, dim=1) == batch_targets).float())

                loss_image = self.image_loss(inputs_jit)
                loss_r_feature = self.bn_loss()

                # loss_hist = self.hist_cretirion(outputs['embedding'], batch_targets)

                # combining losses
                loss_aux = loss_r_feature + loss_image
                # loss_aux = loss_r_feature + loss_image + loss_hist

                loss = self.main_loss_multiplier * ce_loss + loss_aux
                self.meters.update({"Total Loss": loss.item(),
                                    "R Loss": loss_r_feature.item(),
                                    "CE Loss": ce_loss.item(),
                                    "Prior Loss": loss_image.item(),
                                    "Accuracy": acc.item(),
                                    # "Hist Loss": loss_hist.item(),
                                    })

                if iteration_loc % self.print_every == 0:
                    print(f"[{self.num_generations}][{iteration_loc}/{iterations_per_layer}]: " + self.meters())

                    if self.hook_for_display is not None:
                        self.hook_for_display(inputs, targets)

                # do image update
                loss.backward()
                optimizer.step()

                # clip color outlayers
                inputs.data = clip(inputs.data)

                if best_cost > loss.item() and downsampling_ratio == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
        return denormalize(best_inputs)


class DeepInversionClassMountain(DeepInversionClass):
    def __init__(self, bs, net_teacher, net_student, coefficients,
                 embeddings_teacher, embeddings_student):
        """
        :param bs: batch size per GPU for image generation
        :parameter net_teacher: Pytorch model to be inverted
        :param coefficients: dictionary with parameters and coefficients for optimization.
            keys:
            "r_feature" - coefficient for feature distribution regularization
            "tv_l1" - coefficient for total variation L1 loss
            "tv_l2" - coefficient for total variation L1 loss
            "l2" - l2 penalization weight
            "lr" - learning rate for optimization
            "main_loss_multiplier" - coefficient for the main loss optimization
            "adi_scale" - coefficient for Adaptive DeepInversion, competition, def =0 means no competition
        network_output_function: function to be applied to the output of the network to get the output
        hook_for_display: function to be executed at every print/save call, useful to check accuracy of verifier
        """

        super().__init__(bs, net_teacher, net_student, coefficients, lambda x: x, None)

        self.predecessor_teacher = embeddings_teacher
        self.predecessor_student = embeddings_student

        self.meters = AverageMeters(AverageMeter("Total Loss"),
                                    AverageMeter("R Loss"),
                                    AverageMeter("CE Loss"),
                                    AverageMeter("Prior Loss"),
                                    AverageMeter("Accuracy"),
                                    AverageMeter("Mountain Loss"),
                                    )

    def get_images(self, targets=None, preprocessor=None, starts=None):
        self.register_hooks()
        self.net_teacher.requires_grad_(True)
        best_cost = np.inf

        inputs = torch.randn((targets.size(0), 3, 256, 128), requires_grad=True,
                             device='cuda') if starts is None else torch.tensor(starts.cuda(), requires_grad=True,
                                                                                device='cuda')
        targets = targets.to('cuda')
        pooling_function = nn.AvgPool2d(kernel_size=2)
        optimizer = optim.Adam([inputs], lr=self.lr, betas=(0.5, 0.9), eps=1e-8)

        for iterations_per_layer, downsampling_ratio in zip([2000, 1000], [2, 1]):

            lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

            for iteration_loc in range(iterations_per_layer):
                # learning rate scheduling
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                batch_inputs, batch_targets = inputs, targets

                inputs_jit = pooling_function(batch_inputs) if downsampling_ratio != 1 else batch_inputs

                inputs_jit = self.augment(inputs_jit, preprocessor)

                # forward pass
                optimizer.zero_grad()
                self.net_teacher.zero_grad()

                preds, outputs = self.net_teacher(inputs_jit)
                preds = self.network_output_function(preds)

                _, outputs_s = self.net_student(inputs_jit)

                # R_cross classification loss
                ce_loss = self.criterion(preds, batch_targets)

                acc = torch.mean((torch.argmax(preds, dim=1) == batch_targets).float())

                loss_image = self.image_loss(inputs_jit)
                loss_r_feature = self.bn_loss()

                loss_mountain = self.mountain_loss(outputs['embedding'], outputs_s['embedding'].detach(), batch_targets)
                # combining losses
                # loss_aux = loss_r_feature + loss_image
                loss_aux = loss_r_feature + loss_image + loss_mountain

                loss = self.main_loss_multiplier * ce_loss + loss_aux
                self.meters.update({"Total Loss": loss.item(),
                                    "R Loss": loss_r_feature.item(),
                                    "CE Loss": ce_loss.item(),
                                    "Prior Loss": loss_image.item(),
                                    "Accuracy": acc.item(),
                                    "Mountain Loss": loss_mountain.item(),
                                    })

                if iteration_loc % self.print_every == 0:
                    print(f"[{self.num_generations}][{iteration_loc}/{iterations_per_layer}]: " + self.meters())

                    if self.hook_for_display is not None:
                        self.hook_for_display(inputs, targets)

                # do image update
                loss.backward()
                optimizer.step()

                # clip color outlayers
                inputs.data = clip(inputs.data)

                if best_cost > loss.item() and downsampling_ratio == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
        return denormalize(best_inputs) 

    def mountain_loss(self, embeddings_t, embeddings_s, targets):
        losses = []
        for embedding_t, embedding_s, target in zip(embeddings_t, embeddings_s, targets):
            predecessors_t = self.predecessor_teacher[int(target)]
            residual_t = predecessors_t - embedding_t.unsqueeze(dim=0)
            predecessors_s = self.predecessor_student[int(target)]
            residual_s = predecessors_s - embedding_s.unsqueeze(dim=0)
            residual_cosines = (F.normalize(residual_t, dim=1) * F.normalize(residual_s, dim=1)).sum(dim=1)
            losses.append(residual_cosines.mean())
            self_cosine = (F.normalize(embedding_t, dim=0) * F.normalize(embedding_s, dim=0)).sum()
            losses.append(residual_cosines.mean() + 1. - self_cosine)
        return self.adi_scale * (sum(losses) / len(losses))


class DeepInversionClassInc(DeepInversionClass):
    def __init__(self, bs, net_teacher, net_student, net_inc, coefficients):
        """
        :param bs: batch size per GPU for image generation
        :parameter net_teacher: Pytorch model to be inverted
        :param coefficients: dictionary with parameters and coefficients for optimization.
            keys:
            "r_feature" - coefficient for feature distribution regularization
            "tv_l1" - coefficient for total variation L1 loss
            "tv_l2" - coefficient for total variation L1 loss
            "l2" - l2 penalization weight
            "lr" - learning rate for optimization
            "main_loss_multiplier" - coefficient for the main loss optimization
            "adi_scale" - coefficient for Adaptive DeepInversion, competition, def =0 means no competition
        network_output_function: function to be applied to the output of the network to get the output
        hook_for_display: function to be executed at every print/save call, useful to check accuracy of verifier
        """

        super().__init__(bs, net_teacher, net_student, coefficients, lambda x: x, None)
        self.net_inc = net_inc
        self.meters = AverageMeters(AverageMeter("Total Loss"),
                                    AverageMeter("R Loss"),
                                    AverageMeter("CE Loss"),
                                    AverageMeter("Prior Loss"),
                                    AverageMeter("Accuracy"),
                                    AverageMeter("ADI Loss"),
                                    AverageMeter("Inc Loss"),
                                    )

    def get_images(self, targets=None, preprocessor=None, starts=None):
        self.register_hooks()
        self.net_teacher.requires_grad_(True)
        self.net_inc.requires_grad_(True)

        best_cost = np.inf

        inputs = torch.randn((targets.size(0), 3, 256, 128), requires_grad=True,
                             device='cuda') if starts is None else torch.tensor(starts.cuda(), requires_grad=True,
                                                                                device='cuda')
        targets = targets.to('cuda')
        pooling_function = nn.AvgPool2d(kernel_size=2)
        optimizer = optim.Adam([inputs], lr=self.lr, betas=(0.5, 0.9), eps=1e-8)

        for iterations_per_layer, downsampling_ratio in zip([2000, 1000], [2, 1]):

            lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

            for iteration_loc in range(iterations_per_layer):
                # learning rate scheduling
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                batch_inputs, batch_targets = inputs, targets

                inputs_jit = pooling_function(batch_inputs) if downsampling_ratio != 1 else batch_inputs

                inputs_jit = self.augment(inputs_jit, preprocessor)

                # forward pass
                optimizer.zero_grad()
                self.net_teacher.zero_grad()
                self.net_inc.zero_grad()
                self.net_student.zero_grad()

                preds, outputs = self.net_teacher(inputs_jit)
                preds = self.network_output_function(preds)

                # R_cross classification loss
                ce_loss = self.criterion(preds, batch_targets)

                acc = torch.mean((torch.argmax(preds, dim=1) == batch_targets).float())

                loss_image = self.image_loss(inputs_jit)
                loss_r_feature = self.bn_loss()
                loss_adi, loss_inc = self.adi_loss(inputs_jit, preds)

                loss_aux = loss_r_feature + loss_image + loss_adi + loss_inc

                loss = self.main_loss_multiplier * ce_loss + loss_aux
                self.meters.update({"Total Loss": loss.item(),
                                    "R Loss": loss_r_feature.item(),
                                    "CE Loss": ce_loss.item(),
                                    "Prior Loss": loss_image.item(),
                                    "Accuracy": acc.item(),
                                    "ADI Loss": loss_adi.item(),
                                    "Inc Loss": loss_inc.item(),
                                    })

                if iteration_loc % self.print_every == 0:
                    print(f"[{self.num_generations}][{iteration_loc}/{iterations_per_layer}]: " + self.meters())

                    if self.hook_for_display is not None:
                        self.hook_for_display(inputs, targets)

                # do image update
                loss.backward()
                optimizer.step()

                # clip color outlayers
                inputs.data = clip(inputs.data)

                if best_cost > loss.item() and downsampling_ratio == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
        return denormalize(best_inputs)

    def adi_loss(self, inputs, preds_teacher):
        preds_student, outputs_student = self.net_student(inputs)
        preds_student = self.network_output_function(preds_student)
        preds_student_base = preds_student[..., -preds_teacher.size(1):]
        preds_student_base = preds_student_base.softmax(dim=1)
        preds_student_inc = preds_student[..., :-preds_teacher.size(1)]
        preds_student_inc = preds_student_inc.softmax(dim=1)

        entropy = (-preds_student_base * preds_student_base.log()).sum(dim=1).mean()
        loss_verifier_cig = -entropy

        preds_inc, outputs_inc = self.net_inc(inputs)
        preds_inc = self.network_output_function(preds_inc)
        preds_inc = preds_inc[..., :-preds_teacher.size(1)]
        preds_inc = preds_inc.softmax(dim=1)

        loss_inc = F.kl_div(preds_student_inc.log(), preds_inc)

        return self.adi_scale * loss_verifier_cig, self.adi_scale * loss_inc
