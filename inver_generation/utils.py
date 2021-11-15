# -*- coding: utf-8 -*-
# Time    : 2020/9/24 9:46
# Author  : Yichen Lu

import os
import os.path as osp
from PIL import Image
import numpy as np
import torch


def save_as_images(tensors, labels, save_folder, iteration):
    if not osp.exists(osp.join(save_folder, str(iteration))):
        os.mkdir(osp.join(save_folder, str(iteration)))
    samples = tensors.detach().cpu().permute(0, 2, 3, 1)
    samples = denormalize(samples)
    samples = samples.numpy()

    for sample, label in zip(samples, labels):
        image = Image.fromarray((sample * 255).astype(np.uint8))
        digest = hash(image.tobytes())
        image.save(osp.join(save_folder, str(iteration), f"{label.item()}_{digest}.png"))
    print(f"All images are saved in {osp.join(save_folder, str(iteration))}.")


def denormalize(image_tensor):

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


class InfiniteLoader(object):
    def __init__(self, loader):
        self.loader = loader
        self.iter_obj = iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        try:
            ret = next(self.iter_obj)
        except StopIteration:
            self.iter_obj = iter(self.loader)
            ret = next(self.iter_obj)
        yield ret

    def __next__(self):
        try:
            ret = next(self.iter_obj)
        except StopIteration:
            self.iter_obj = iter(self.loader)
            ret = next(self.iter_obj)
        return ret

