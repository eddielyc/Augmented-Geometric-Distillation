# -*- coding: utf-8 -*-
# Time    : 2020/9/3 22:15
# Author  : Yichen Lu

import torch


class NoiseLoader(object):
    def __init__(self, classifier, latent_size, batch_size=128):
        self.classifier = classifier
        self.w = self.classifier.W.detach().cpu().clone()
        self.latent_size = latent_size
        self.batch_size = batch_size

        self.std, self.mean = torch.std_mean(self.w, dim=0, keepdim=True)

    def __iter__(self):
        return self

    def __next__(self):
        indices = torch.randint(0, self.w.size(0), size=(self.batch_size, ), dtype=torch.long, device=self.w.device)
        centers = self.w[indices]
        centers = (centers - self.mean) / self.std
        # print(torch.std_mean(centers, dim=0))
        salt = torch.randn(self.batch_size, self.latent_size)
        salted = torch.cat([centers, salt], dim=1)
        return salted, indices


if __name__ == '__main__':
    from reid.models.classifier import Linear
    l = Linear(2048, 751, torch.device('cpu'))
    l.load_state_dict(torch.load('../logs/baselines/market/checkpoint.pth.tar')['classifier'])

    nl = NoiseLoader(l, 1024)
    for _ in nl:
        pass


