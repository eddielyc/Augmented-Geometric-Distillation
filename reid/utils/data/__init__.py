from reid.utils.data.preprocessor import *
from reid.utils.data.transforms import *
from reid.utils.data.triplet_sampler import *
from reid.utils.data.dataset import *

import torchvision.transforms as T
from torch.utils.data import DataLoader


def build_train_loader(dataset, args, metric=True, contrast=False):
    normalizer = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    training_transformer = T.Compose([
        T.Resize((args.height, args.width)),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((args.height, args.width)),
        T.ToTensor(),
        normalizer,
        RandomErasingBeta(probability=args.re, sh=args.re_area, mean=[0.485, 0.456, 0.406])
    ])

    if not contrast:
        preprocessed = Preprocessor(dataset.train, transform=training_transformer, preload=args.preload)
    else:
        preprocessed = ContrastPreprocessor(dataset.train, transform=training_transformer, peers=args.peers, preload=args.preload)

    if metric:
        training_loader = DataLoader(
            preprocessed,
            batch_sampler=RandomIdentitySampler(dataset.train, args.batch_size, 4),
            # reproducibility
            num_workers=4 if not hasattr(args, "seed") or args.seed is None else 0,
            pin_memory=True,
        )
    else:
        training_loader = DataLoader(
            preprocessed,
            batch_size=args.batch_size,
            # reproducibility
            num_workers=4 if not hasattr(args, "seed") or args.seed is None else 0,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    return training_loader


def build_no_aug_loader(dataset, args):
    normalizer = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transformer = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        normalizer
    ])

    no_aug_loader = DataLoader(
        Preprocessor(dataset.train, transform=transformer),
        batch_size=args.batch_size,
        # reproducibility
        num_workers=4 if not hasattr(args, "seed") or args.seed is None else 0,
        shuffle=False,
        pin_memory=True
    )

    return no_aug_loader


def build_test_loader(dataset, args):
    normalizer = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    test_transformer = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        normalizer
    ])

    if isinstance(dataset, MixedDataset):
        query_loader = {name: DataLoader(Preprocessor(query, transform=test_transformer),
                                         batch_size=args.batch_size,
                                         # reproducibility
                                         num_workers=4 if not hasattr(args, "seed") or args.seed is None else 0,
                                         shuffle=False,
                                         pin_memory=True)
                        for name, query in dataset.query.items()}
    else:
        query_loader = DataLoader(Preprocessor(dataset.query, transform=test_transformer),
                                  batch_size=args.batch_size,
                                  # reproducibility
                                  num_workers=4 if not hasattr(args, "seed") or args.seed is None else 0,
                                  shuffle=False,
                                  pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, transform=test_transformer),
        batch_size=args.batch_size,
        # reproducibility
        num_workers=4 if not hasattr(args, "seed") or args.seed is None else 0,
        shuffle=False,
        pin_memory=True
    )

    return query_loader, gallery_loader
