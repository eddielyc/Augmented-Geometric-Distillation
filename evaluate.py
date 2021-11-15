import os
import argparse
import os.path as osp
import glob
import torch
import torch.nn as nn

from reid.utils.data.dataset import Dataset, MixedDataset
from reid.utils.data import build_test_loader
from reid.models.backbone import ResNet
from reid.evaluation.evaluators import Evaluator, IncrementalEvaluator
from reid.utils.serialization import load_checkpoint


def conclude(log_dir, ctx):
    def print_results(file, name, mAP, cmc_topk, cmcs):
        file.write(f"{name}:\n")
        file.write(f"    mAP: {mAP:4.1%}\n")
        for topk, cmc in zip(cmc_topk, cmcs):
            file.write(f"    CMC-{topk:<3}: {cmc:4.1%}\n")
        file.write("\n")

    with open(osp.join(log_dir, 'eval.txt'), 'a') as file:
        if isinstance(ctx['mAP'], dict):
            file.write(f"{ctx['dataset']}:\n")
            for (name, mAP), (_, cmcs) in zip(ctx['mAP'].items(), ctx['cmc'].items()):
                print_results(file, name, mAP, ctx['cmc_topk'], cmcs)
        else:
            print_results(file, ctx['dataset'], ctx['mAP'], ctx['cmc_topk'], ctx['cmc'])


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets, loaders = [], []
    for dataset_name in args.dataset:
        if "+" not in dataset_name:
            dataset = Dataset(args.data_root, dataset_name)
        else:
            subdatasets = []
            for name in dataset_name.split("+"):
                if 'U' in name:
                    subdatasets.append(MixedDataset(*[Dataset(args.data_root, subname) for subname in name.split("U")], mode='U'))
                else:
                    subdatasets.append(Dataset(args.data_root, name))
            dataset = MixedDataset(*subdatasets, mode='+')
        datasets.append(dataset)
        loaders.append(build_test_loader(dataset, args))

    if args.recursive:
        assert osp.isdir(args.ckpt), "ckpt must be a folder in 'recursive' mode."
        ckpts = glob.glob(osp.join(args.ckpt, '**/checkpoint*.pth.tar'), recursive=True)
    else:
        if osp.isdir(args.ckpt):
            ckpts = [osp.join(args.ckpt, 'checkpoint.pth.tar')]
        elif osp.isfile(args.ckpt):
            ckpts = [args.ckpt]
        else:
            raise FileExistsError("checkpoint doesn't exists.")

    for ckpt in ckpts:

        checkpoint = load_checkpoint(ckpt)

        if args.output:
            log_dir = osp.dirname(ckpt)
            with open(osp.join(log_dir, 'eval.txt'), 'a') as file:
                file.write(str(args) + '\n')
                file.write(f"{ckpt}" + '\n')
                file.write(f"Epoch: {checkpoint['epoch']}" + '\n')

        model = ResNet(depth=args.depth, last_stride=args.last_stride, last_pooling=args.last_pooling,
                       embedding=args.embedding)
        model.load_state_dict(checkpoint["backbone"], strict=False)
        model = nn.DataParallel(model).to(device)

        cmc_topk = [1, 5, 10, 20]
        for dataset, (query_loader, gallery_loader) in zip(datasets, loaders):
            evaluator = Evaluator(model) if not isinstance(dataset, MixedDataset) else IncrementalEvaluator(model)
            mAP, cmc = evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery,
                                          re_ranking=args.re_ranking, dist_type='cos', cmc_topk=cmc_topk,
                                          output_feature=args.output_feature
                                          )
            if args.output:
                conclude(osp.dirname(ckpt), {'mAP': mAP, 'cmc': cmc,
                                             'cmc_topk': cmc_topk, 'dataset': dataset.dataset,
                                             })


if __name__ == "__main__":
    working_dir = osp.dirname(osp.abspath(__file__))
    parser = argparse.ArgumentParser(description="Evaluate in IL-ReID.")
    parser.add_argument('--data-root', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument("--dataset", nargs='*', type=str, default=["market", "duke", "msmt17"])
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("-c", "--ckpt", type=str, default='./logs/checkpoint.pth.tar')
    parser.add_argument("-r", "--re-ranking", action='store_true')
    parser.add_argument("-o", '--output', action='store_true', default=False)
    parser.add_argument("--recursive", action='store_true', default=False)
    parser.add_argument("-g", "--gpu", nargs='*', type=str, default=['0', '1'])

    # model config
    parser.add_argument("-e", "--embedding", type=int, default=2048)
    parser.add_argument("-lp", "--last-pooling", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("-ls", "--last-stride", type=int, default=2, choices=[1, 2])
    parser.add_argument("--depth", type=int, default=50, choices=[34, 50])
    parser.add_argument("--output-feature", type=str, default='embedding', choices=['embedding', 'global'])

    args = parser.parse_args()

    # args.dataset = ["cuhk03"]
    # args.ckpt = './logs/r50/baselines/market'
    # args.gpu = '0'

    main(args)
