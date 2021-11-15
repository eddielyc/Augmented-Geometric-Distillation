import os.path as osp
from pathlib import Path
from glob import glob
import re
from collections import defaultdict
from random import sample
from itertools import chain


class Dataset(object):
    def __init__(self, data_root, dataset, exemplars=0):
        self.data_root = data_root
        self.dataset = dataset
        self.exemplars = exemplars
        self.train_path = "bounding_box_train"
        self.gallery_path = "bounding_box_test"
        self.query_path = "query"

        self.train, self.query, self.gallery = [], [], []
        self.train_ids, self.query_ids,  self.gallery_ids, self.cams = set(), set(), set(), set()
        self.pid_samples = defaultdict(list)

        self.load()
        print(self)

    def preprocess(self, subpath, relabel=True, exemplars=0, seed=1):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        pids, ret = {}, []
        folder = osp.join(self.data_root, self.dataset, subpath)
        fpaths = sorted(glob(osp.join(folder, "*.jpg")) + glob(osp.join(folder, "*.png")))
        # fpaths = sorted(glob(osp.join(self.data_root, self.dataset, subpath, '*.(png|jpg)')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored
            if relabel:
                if pid not in pids:
                    pids[pid] = len(pids)
            else:
                if pid not in pids:
                    pids[pid] = pid
            pid = pids[pid]
            self.pid_samples[pid].append((fpath, pid, cam))
            self.cams.add(cam)
            ret.append((fpath, pid, cam))

        if exemplars:
            print(f"Sampling training set to {len(pids.values())} ways {exemplars} shot.")
            if seed:
                import random
                random.seed(seed)
                print(f"Set seed {seed}.")

            for pid, samples in self.pid_samples.items():
                if len(samples) > exemplars:
                    self.pid_samples[pid] = sample(samples, exemplars)
            ret = list(chain(*self.pid_samples.values()))
        return ret, set(pids.values())

    def load(self):
        self.train, self.train_ids = self.preprocess(self.train_path, relabel=True, exemplars=self.exemplars)
        self.gallery, self.gallery_ids = self.preprocess(self.gallery_path, relabel=False)
        self.query, self.query_ids = self.preprocess(self.query_path, relabel=False)

    def __add__(self, other):
        return MixedDataset(self, other)

    def __repr__(self):
        return f"{self.__class__.__name__} dataset loaded \n" \
               f"  subset   | # ids | # images \n" \
               f"  --------------------------- \n" \
               f"  train    | {len(self.train_ids):5d} | {len(self.train):8d} \n" \
               f"  query    | {len(self.query_ids):5d} | {len(self.query):8d} \n" \
               f"  gallery  | {len(self.gallery_ids):5d} | {len(self.gallery):8d} \n"


class MixedDataset(object):
    def __init__(self, *datasets, mode="+"):
        assert mode and mode in "+U", "Invalid mode."
        self.mode = mode
        self.dataset = mode.join([dataset.dataset for dataset in datasets])
        self.datasets = datasets

        self.train, self.train_ids, self.cams = [], set(), set()
        train_id_mapping = self.non_overlapping({dataset.dataset: dataset.train_ids for dataset in self.datasets})
        cam_id_mapping = self.non_overlapping({dataset.dataset: dataset.cams for dataset in self.datasets})

        self.query, self.query_ids = defaultdict(list) if mode == '+' else [], set()
        # query_id_mapping = self.non_overlapping({dataset.dataset: dataset.query_ids for dataset in self.datasets})
        self.gallery, self.gallery_ids = [], set()
        gallery_id_mapping = self.non_overlapping({dataset.dataset: dataset.gallery_ids for dataset in self.datasets})

        for i, dataset in enumerate(datasets):
            for fpath, pid, cam, *_ in dataset.train:
                self.train.append((fpath, train_id_mapping[dataset.dataset][pid], cam_id_mapping[dataset.dataset][cam], i))
                self.cams.add(cam_id_mapping[dataset.dataset][cam])
                self.train_ids.add(train_id_mapping[dataset.dataset][pid])

            # query and gallery share the same mapping
            for fpath, pid, cam, *_ in dataset.query:
                if mode == '+':
                    self.query[dataset.dataset].append((fpath, gallery_id_mapping[dataset.dataset][pid], cam_id_mapping[dataset.dataset][cam], i))
                else:
                    self.query.append((fpath, gallery_id_mapping[dataset.dataset][pid], cam_id_mapping[dataset.dataset][cam], i))
                self.query_ids.add(gallery_id_mapping[dataset.dataset][pid])

            for fpath, pid, cam, *_ in dataset.gallery:
                self.gallery.append((fpath, gallery_id_mapping[dataset.dataset][pid], cam_id_mapping[dataset.dataset][cam], i))
                self.gallery_ids.add(gallery_id_mapping[dataset.dataset][pid])

        print(self)

    @staticmethod
    def non_overlapping(dataset2ids):
        id_pool = []
        mapping = {}
        for dataset, ids in dataset2ids.items():
            mapping[dataset] = {}
            for identity in ids:
                mapping[dataset][identity] = len(id_pool)
                id_pool.append(identity)
        return mapping

    def __repr__(self):
        return f"{self.__class__.__name__} dataset loaded \n" \
               f"  subset   | # ids | # images \n" \
               f"  --------------------------- \n" \
               f"  train    | {len(self.train_ids):5d} | {len(self.train):8d} \n" \
               f"  query    | {len(self.query_ids):5d} | {sum([len(query) for query in self.query.values()]) if self.mode == '+' else len(self.query):8d} \n" \
               f"  gallery  | {len(self.gallery_ids):5d} | {len(self.gallery):8d} \n"


class InversionDataset(object):
    def __init__(self, data_root, exemplars=40):
        self.dataset = "inversion"
        self.cams = {-1}
        self.data_root = data_root
        self.exemplars = exemplars

        self.train, self.train_ids = [], set()
        self.pid_samples = defaultdict(list)

        self.load()
        print(self)

    def preprocess(self, data_root):
        root = Path(data_root)
        pids = set()
        fpaths = sorted(list(root.glob("*.jpg")) + list(root.glob("*.png")))
        for fpath in fpaths:
            if len(fpath.stem.split("_")) == 2:
                pid = int(fpath.stem.split("_")[0])
                if len(self.pid_samples[pid]) < self.exemplars:
                    self.pid_samples[pid].append((str(fpath), pid, -1))
            else:
                raise RuntimeError("Invalid file name.")
            pids.add(pid)
        return list(chain(*self.pid_samples.values())), pids

    def load(self):
        self.train, self.train_ids = self.preprocess(self.data_root)

    def __repr__(self):
        return f"{self.__class__.__name__} dataset loaded \n" \
               f"  subset   | # ids | # images \n" \
               f"  --------------------------- \n" \
               f"  train    | {len(self.train_ids):5d} | {len(self.train):8d} \n"


if __name__ == '__main__':
    dataset = Dataset('/home/luyichen/projects/Red/data', 'cuhk03')
    pass
