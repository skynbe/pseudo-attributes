import argparse
import os, copy

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# import utils
from modules.transform import *
from model import Model
import os, sys, logging, time, random, json, pdb
from pathlib import Path


from factory import ModelFactory, TrainerFactory, DataLoaderFactory, TransformFactory


import wandb

DATA_ROOT = Path('./data')
CHECKPOINT_ROOT = Path('./checkpoint')


def main():
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=512, type=int, help='Feature dim for latent vector')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning Rate')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--max_epoch', default=50, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--test_epoch', default=25, type=int, help='Test epoch')
    parser.add_argument('--save_epoch', default=25, type=int, help='Save epoch')
    parser.add_argument('--train_eval_epoch', default=50, type=int, help='Save epoch')
    parser.add_argument('--label_noise', default=0.0, type=float, help='Label noise ratio')
    
    parser.add_argument('--dataset', default='stl10', type=str, help='Dataset')
    parser.add_argument('--arch', default='TinySimCLR', type=str, help='Model architecture')
    parser.add_argument('--trainer', default='contrastive', type=str, help='Training scheme')
    parser.add_argument('--cluster_weight_type', default='scale', type=str, help='Training scheme')
    parser.add_argument('--clustering_type', default='whole', type=str, help='whole/class')
    parser.add_argument('--centroid', default='cosine', type=str, help='cosine or l2')
    
    parser.add_argument('--target_attr', default='', type=str, help='Target attributes')
    parser.add_argument('--bias_attrs', nargs='+', help='Bias attributes')
    
    parser.add_argument('--num_partitions', default=1, type=int, help='Test epoch')
    parser.add_argument('--k', default=1, type=int, help='# of clusters')
    parser.add_argument('--ks', nargs='+', help='# of clusters list (multi)')
    parser.add_argument('--update_cluster_iter', default=0, type=int, help='0 for every epoch')
    parser.add_argument('--feature_bank_init', action='store_true')
    parser.add_argument('--num_multi_centroids', default=1, type=int, help='# of centroids')
    
    parser.add_argument('--desc', default='test', type=str, help='Checkpoint folder name')
    parser.add_argument('--note', default='test', type=str, help='just for note')
    parser.add_argument('--version', default='', type=str, help='Version')
    parser.add_argument('--load_epoch', default=-1, type=int, help='Load model epoch')
    parser.add_argument('--weight_decay', default=2e-2, type=float, help='Weight decay')
    parser.add_argument('--momentum', default=0.3, type=float, help='Positive class priorx')
    parser.add_argument('--adj', default=2.0, type=float, help='Label noise ratio')
    parser.add_argument('--adj_type', default='', type=str, help='multiply or default')
    parser.add_argument('--exp_step', default=0.01, type=float, help='Exponential step size for weight averaging in AvgFixedCentroids')
    parser.add_argument('--avg_weight_type', default='expavg', type=str, help='avg type for weight averaging in AvgFixedCentroids')
    parser.add_argument('--overlap_type', default='exclusive', type=str, help='Channel overlap type for hetero clustering, [exclusive, half_exclusive]')
    parser.add_argument('--gamma_reverse', action='store_true')
    parser.add_argument('--scale', default=1.0, type=float, help='Dataset scale')
    parser.add_argument('--sampling', default='', type=str, help='class_subsampling/class_resampling')
    
    parser.add_argument('--use_base', default='', type=str, help='load {base_model_name}.pth')
    parser.add_argument('--load_base', default='', type=str, help='load {base_model_name}.pth')
    parser.add_argument('--load_path', default='', type=str, help='load {base_model_name}.pth')
    
    parser.add_argument('--scheduler', default='', type=str, help='cosine')
    parser.add_argument('--scheduler_param', default=0, type=int, help='cosine')
    
    parser.add_argument('--resume', default='', type=str, help='Run ID')
    parser.add_argument('--optim', default='adam', type=str, help='adam or sgd')
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--feature_fix', action='store_true')

    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()
    args.num_clusters = args.k
    args.num_multi_centroids = len(args.ks)
    
    args.wandb = not args.no_wandb
    
    # savings
    checkpoint_dir = CHECKPOINT_ROOT / args.dataset / args.target_attr / args.desc
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loaders, datasets = DataLoaderFactory.create(args.dataset, trainer=args.trainer, batch_size=args.batch_size,
                                                 num_workers=4, configs=args)
    
    num_classes = len(datasets['test'].classes)
    print('# Classes: {}'.format(num_classes))
    
    model_args = {
        "name": args.arch,
        "feature_dim": args.feature_dim,
        "num_classes": num_classes,
        "feature_fix": args.feature_fix,
    }
    
    # model setup and optimizer config
    model = ModelFactory.create(**model_args).cuda()
    model = nn.DataParallel(model)
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    scheduler = None
    if args.scheduler == 'cosine':
        assert args.scheduler_param != 0
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.scheduler_param)
        

    args.checkpoint_dir = checkpoint_dir
    trainer = TrainerFactory.create(args.trainer, args, model, loaders, optimizer, num_classes)
    trainer.set_checkpoint_dir(checkpoint_dir)
    start_epoch = 1

    if args.load_base:
        trainer.load_base_model(args.load_base)
    
    if args.use_base:
        trainer.use_base_model(args.use_base)
        
    if args.load_epoch >= 0:
        trainer.load_model(args.load_epoch)
        start_epoch = trainer.epoch + 1
        
    if args.load_path:
        if args.origin_attr:
            path = '{}/{}'.format(args.origin_attr, args.load_path)
        else:
            trainer.load_path(args.load_path)

    if args.eval:
        trainer.test_unbiased(epoch=start_epoch-1)
        return
    
    
    for epoch in range(start_epoch, args.max_epoch+1):
        trainer.train(epoch=epoch)
        
        if epoch % args.test_epoch == 0:
            trainer.save_model(epoch=epoch)
            trainer.test_unbiased(epoch=epoch)
    
        if scheduler is not None:
            scheduler.step()        
        
        
if __name__ == "__main__":
    main()

            
            