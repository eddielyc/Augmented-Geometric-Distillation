# -*- coding: utf-8 -*-
# Time    : 2020/9/3 21:35
# Author  : Yichen Lu

import time

import torch
from torch import nn as nn
from torch.nn import functional as F
from reid.utils.meters import AverageMeters, AverageMeter
from .utils import save_as_images
from deep_inversion.deepinversion import DeepInversionFeatureHook
from deep_inversion.loss import prior_losses


class Trainer(object):
    def __init__(self, encoder, classifier, decoder, discriminator, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.decoder = decoder
        self.discriminator = discriminator
        self.classifier = classifier
        self.encoder = encoder
        self.batch_size = args.batch_size

        self.loss_r_feature_layers = [DeepInversionFeatureHook(module) for module in self.encoder.modules()
                                      if isinstance(module, nn.BatchNorm2d)]

        self.snapshot_every = args.snapshot_every
        self.save_folder = args.save_folder

        self.meters = AverageMeters(AverageMeter("Batch Time"),
                                    AverageMeter("Gen Loss"),
                                    AverageMeter("Dis Loss"),
                                    AverageMeter("Cond Loss"),
                                    # AverageMeter("BN Loss"),
                                    )

        # self.label_encoder = nn.Embedding(1041, 1041)
        # self.label_encoder.weight.data = torch.eye(1041)
        self.label_encoder = nn.Embedding(1041, 2048)
        self.label_encoder.weight.data = self.classifier.W.data.cpu()

    def train(self, epoch, training_loader, optimizer_gen, optimizer_dis, lr_scheduler):
        self.classifier.eval()
        self.encoder.eval()
        self.decoder.train()
        self.discriminator.train()

        lr_scheduler.step(epoch)
        for i in range(1, len(training_loader) + 1):
            start = time.time()

            loss_dis = self.optimize_discriminator(training_loader, optimizer_dis)

            # loss_gen = self.optimize_generator(optimizer_gen)
            loss_gen, loss_cond = self.optimize_conditional_generator(optimizer_gen)

            batch_time = time.time() - start

            self.meters.update([batch_time,
                                loss_gen.item(),
                                loss_dis.item(),
                                loss_cond.item(),
                                ])

            print(f"Epoch: [{epoch}][{i}/{len(training_loader)}], " + self.meters())

        if epoch % self.snapshot_every == 0:
            self.generate_batch(epoch)

    def generate_batch(self, iteration):
        centers, labels = torch.randn((self.batch_size, 2304)), torch.randint(1041, size=(self.batch_size, ), dtype=torch.long)
        centers, labels = centers.to(self.device), labels.to(self.device)
        generated = self.decoder(centers)

        save_as_images(generated, labels, self.save_folder, iteration)

    def _parse_data(self, inputs):
        imgs, _, pids, _, *_ = inputs
        inputs = imgs.to(self.device)
        inputs = F.avg_pool2d(inputs, kernel_size=4)
        pids = pids.to(self.device)
        return inputs, pids

    def __gradient_penalty(self, real_samps, fake_samps, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """
        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = torch.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = epsilon * real_samps + ((1 - epsilon) * fake_samps)
        merged.requires_grad_(True)

        # forward pass
        op = self.discriminator(merged)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(outputs=op, inputs=merged,
                                       grad_outputs=torch.ones_like(op), create_graph=True,
                                       retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def optimize_discriminator(self, training_loader, optimizer_dis):
        for _ in range(5):
            inputs = next(training_loader)

            reals, _ = self._parse_data(inputs)

            targets = torch.randint(1041, size=(self.batch_size,), dtype=torch.long)
            embedded_targets = self.label_encoder(targets)
            noise = torch.randn((self.batch_size, 256))
            merged = torch.cat([embedded_targets, noise], dim=-1)

            # merged = torch.randn((self.batch_size, 512))
            fakes = self.decoder(merged).detach()

            real_values = self.discriminator(reals)
            fake_values = self.discriminator(fakes)

            # loss_dis = torch.mean(fake_values) - torch.mean(real_values) + 0.001 * torch.mean(real_values ** 2)
            loss_dis = torch.mean(fake_values) - torch.mean(real_values)
            gradient_penalty = self.__gradient_penalty(reals, fakes)
            loss_dis = loss_dis + gradient_penalty

            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()

        return loss_dis

    def optimize_generator(self, optimizer_gen):

        merged = torch.randn((self.batch_size, 512))
        fakes = self.decoder(merged)

        loss_gen = - torch.mean(self.discriminator(fakes))

        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        return loss_gen

    def optimize_conditional_generator(self, optimizer_gen):

        targets = torch.randint(1041, size=(self.batch_size,), dtype=torch.long)
        embedded_targets = self.label_encoder(targets)
        noise = torch.randn((self.batch_size, 256))
        merged = torch.cat([embedded_targets, noise], dim=-1).cuda()

        fakes = self.decoder(merged)

        optimizer_gen.zero_grad()

        loss_gen = - torch.mean(self.discriminator(fakes))
        loss_gen.backward(retain_graph=True)

        features = self.encoder(fakes)['embedding']
        preds = self.classifier(features)
        loss_cond = F.cross_entropy(preds, targets.cuda())
        acc = torch.mean((torch.argmax(preds, dim=1) == targets.cuda()).float())
        loss_cond.backward()

        # rescales = [10.] + [1. for _ in range(len(self.loss_r_feature_layers) - 1)]
        # loss_r_feature = sum([mod.r_feature * rescale for idx, (mod, rescale) in enumerate(zip(self.loss_r_feature_layers, rescales))])
        # (loss_r_feature + loss_cond).backward()

        optimizer_gen.step()

        return loss_gen, loss_cond
