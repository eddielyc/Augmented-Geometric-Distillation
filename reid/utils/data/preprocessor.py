import os.path as osp
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from collections.abc import Iterable
from tqdm import tqdm


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, preload=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.preload = preload
        if self.preload:
            print(f"Preloading dataset: {len(self.dataset)} samples into Preprocessor.")
            self.preloaded = [self._load(i) for i in tqdm(range(len(self.dataset)))]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, Iterable):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        if self.preload:
            img, fpath, *other = self.preloaded[index]
        else:
            img, fpath, *other = self._load(index)
        if self.transform is not None:
            img = self.transform(img)
        return [img, fpath, *other]

    def _load(self, index):
        fname, *other = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        return [img, fpath, *other]


class ContrastPreprocessor(Preprocessor):
    def __init__(self, dataset, root=None, transform=None, peers=2, preload=False):
        super(ContrastPreprocessor, self).__init__(dataset, root, transform, preload)
        self.peers = peers

    def _get_single_item(self, index):
        if self.preload:
            img, fpath, *other = self.preloaded[index]
        else:
            img, fpath, *other = self._load(index)
        img_list = [img] * self.peers
        if self.transform is not None:
            img_list = [self.transform(img) for _ in range(self.peers)]
        return [img_list, fpath, *other]
