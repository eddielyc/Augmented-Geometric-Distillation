# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Pavlo Molchanov and Hongxu Yin
# --------------------------------------------------------

import torch
import os
from torch import distributed, nn
from collections import defaultdict
import random
import numpy as np
from PIL import Image
import torchvision.transforms as T
import imageio
from collections.abc import Iterable


def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


def clip(image_tensor, use_fp16=False):
    """
    adjust the input based on mean and variance
    """
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor, use_fp16=False):
    """
    convert floats back to input
    """
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def batch_images_as_tensors(paths):
    normalizer = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transformer = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        normalizer
    ])

    images = [transformer(Image.open(path).convert('RGB')) for path in paths]
    batched = torch.cat([image.unsqueeze(dim=0) for image in images], dim=0)
    return batched


class GIFGenerator(object):
    def __init__(self, fig):
        self.fig = fig
        self.snapshots = []

    def reset_fig(self, fig):
        self.fig = fig

    def snapshot(self):
        self.fig.canvas.draw()
        image_b, (w, h) = self.fig.canvas.print_to_buffer()
        self.snapshots.append(Image.frombuffer("RGBA", size=(w, h), data=image_b).convert('RGB'))

    def merge(self, path, duration=0.2):
        assert len(self.snapshots) > 0, "Empty snapshots."
        imageio.mimsave(path, self.snapshots, 'GIF', duration=duration)
        del self.snapshots
        self.snapshots = []


class InversionMetricSampler(object):
    def __init__(self, inputs, targets, bs=64, identities_per_batch=8):
        assert inputs.size(0) == targets.size(0), "Size of inputs must be equal to that of targets."
        assert bs % identities_per_batch == 0, "Bs should be divisible by identities_per_batch."
        self.inputs = inputs
        self.targets = targets
        self.bs = bs
        self.ids_per_batch = identities_per_batch
        self.samples_per_id = self.bs // self.ids_per_batch
        self.ids = self.targets.squeeze().unique().tolist()
        self.ids2index = defaultdict(list)
        for i, identity in enumerate(self.targets):
            self.ids2index[identity.item()].append(i)

    def __iter__(self):
        return self

    def __next__(self):
        batch_ids = random.sample(self.ids, self.ids_per_batch)
        indices = [torch.tensor(random.sample(self.ids2index[batch_id], self.samples_per_id), dtype=torch.long)
                   for batch_id in batch_ids]
        indices = torch.cat(indices, dim=0)
        return self.inputs[indices], self.targets[indices]


class InversionSampler(object):
    def __init__(self, inputs, targets, bs=64):
        assert inputs.size(0) == targets.size(0), "Size of inputs must be equal to that of targets."
        self.inputs = inputs
        self.targets = targets
        self.bs = bs
        self.pool = list(range(len(self.targets)))
        random.shuffle(self.pool)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.pool) < self.bs:
            self.pool = list(range(len(self.targets)))
            random.shuffle(self.pool)
        indices = self.pool[:self.bs]
        self.pool = self.pool[self.bs:]
        return self.inputs[indices], self.targets[indices]


class EmbeddingsContainer(object):
    def __init__(self, dataloader, model):
        self.model = model
        self.container = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for i, inputs in enumerate(dataloader, 1):
                inputs, pids = self._parse_data(inputs)
                embeddings = self.model(inputs)['embedding']
                for embedding, pid in zip(embeddings, pids):
                    self.container[int(pid)].append(embedding.unsqueeze(dim=0).detach())
                print(f"Loading embeddings [{i}] / [{len(dataloader)}] ...")

        self.container = {pid: torch.cat(embeddings, dim=0) for pid, embeddings in self.container.items()}

    def update(self, embeddings, pids):
        print(f"Updating {embeddings.size(0)} embeddings ...")
        for embedding, pid in zip(embeddings, pids):
            self.container[int(pid)] = torch.cat([self.container[int(pid)], embedding.unsqueeze(dim=0).detach()], dim=0)

    def _parse_data(self, inputs):
        imgs, _, pids, _, *_ = inputs
        inputs = imgs.cuda()
        pids = pids.cuda()
        return inputs, pids

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.container[key]
        elif isinstance(key, Iterable):
            return [self.container[int(individual_key)] for individual_key in key]
        else:
            raise RuntimeError("Only Int or Iterable instance is allowed for subscript.")


if __name__ == '__main__':
    p = ['/home/luyichen/datasets/Market-1501-v15.09.15/bounding_box_train/0002_c1s1_000551_01.jpg']
    b = batch_images_as_tensors(p)
    pass