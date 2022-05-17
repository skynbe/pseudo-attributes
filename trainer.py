import argparse
import os, pdb
import logging
from PIL import Image

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.distributions import Categorical

from modules.transform import *
from model import Model
from utils.train_utils import *
from utils.io_utils import *
import importlib


import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import pyplot as plt
from kmeans_pytorch import kmeans
        
import wandb
import pickle


class Trainer():
    
    def __init__(self, args, model, loaders, optimizer, num_classes):
        print(self)
        self.args = args
        self.model = model 
        self.loaders = loaders
        self.optimizer = optimizer
        
        self.max_epoch = args.max_epoch
        self.batch_size = args.batch_size

        self.k = args.k
        self.num_classes = num_classes
        self.num_groups = np.power(self.num_classes, len(self.args.bias_attrs)+1)
        self.num_clusters = self.k
        
        self.logger = get_logger('')
        
        self.accs = WindowAvgMeter(name='accs', max_count=20)
        
    
    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        
    
    def save_model(self, epoch):
        if not self.args.no_save or epoch % self.args.save_epoch == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict(), 
                'optimizer' : self.optimizer.state_dict(),
            }, self.checkpoint_dir / 'e{:04d}.pth'.format(epoch))
        return
    
    def load_model(self, epoch=0):
        self.logger.info('Resume training')
        if epoch==0:
            checkpoint_path = max((f.stat().st_mtime, f) for f in self.checkpoint_dir.glob('*.pth'))[1]
            self.logger.info('Resume Latest from {}'.format(checkpoint_path))
        else:
            checkpoint_path = self.checkpoint_dir / 'e{:04d}.pth'.format(epoch)
            self.logger.info('Resume from {}'.format(checkpoint_path))
            
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict((checkpoint['state_dict'])) # Set CUDA before if error occurs.
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        
    
    def load_path(self, file_name):
        self.logger.info('Loading model at ({})'.format(file_name))
        checkpoint_path = self.checkpoint_dir / '..' / '..' / '{}.pth'.format(file_name)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict']) # Set CUDA before if error occurs.

    
    def finetune(self, epoch, iter):
        return
    
    
    def extract_sample(self, data, desc):
        data_ = data.permute(1,2,0).cpu().numpy().astype(np.uint8)
        img = Image.fromarray(data_)
        img.save(self.checkpoint_dir / '{}.png'.format(desc))
    
    
    
    def _extract_features_with_path(self, model, data_loader):
        features, targets = [], []
        ids = []
        paths = []

        for data, target, index, path in tqdm(data_loader, desc='Feature extraction for clustering..', ncols=5):
            data, target, index = data.cuda(), target.cuda(), index.cuda()
            results = model(data)
            features.append(results["feature"])
            targets.append(target)
            ids.append(index)
            paths.append(path)

        features = torch.cat(features)
        targets = torch.cat(targets)
        ids = torch.cat(ids)
        paths = np.concatenate(paths)
        return features, targets, ids, paths
    
    
        
    
class ClassifyTrainer(Trainer):
    
    def train(self, epoch):
        
        data_loader = self.loaders['train']
        total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, ncols=100)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
        
        for data, target, _, _, _, _, _ in train_bar:
            B = target.size(0)
                
            data, target = data.cuda(), target.cuda()
            
            results = self.model(data)
            loss = torch.mean(criterion(results["out"], target.long()))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_num += B
            total_loss += loss.item() * B

            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.max_epoch, total_loss / total_num))
       
        
        return total_loss / total_num
        
    
    def test(self, epoch):
        self.model.eval()
        test_loader = self.loaders['test']
        
        total_top1, total_top5, total_num, test_bar = 0.0, 0.0, 0, tqdm(test_loader, ncols=100)
        with torch.no_grad():

            for data, target, _, _, _, _, _ in test_bar:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                B = target.size(0)
                
                results = self.model(data)
                pred_labels = results["out"].argsort(dim=-1, descending=True)
                
                total_num += B
                total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                         .format(epoch, self.max_epoch, total_top1 / total_num * 100, total_top5 / total_num * 100))
                
                
        self.model.train()

        return

        
class BiasedClassifyTrainer(ClassifyTrainer):
    
    
    def test(self, epoch):
        self.model.eval()
    
        for desc in ['test']:
            loader = self.loaders[desc]
        
            total_top1, total_top5, total_num, test_bar = 0.0, 0.0, 0, tqdm(loader, ncols=100)
            
            with torch.no_grad():

                for data, target, bias, _, _, _, _ in test_bar:
                    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    
                    B = target.size(0)

                    results = self.model(data)             
                    pred_labels = results["out"].argsort(dim=-1, descending=True)

                    total_num += B
                    total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    test_bar.set_description('[{}] Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                             .format(desc, epoch, self.max_epoch, total_top1 / total_num * 100, total_top5 / total_num * 100))
                    
                log = self.logger.info if desc in ['train', 'train_eval'] else self.logger.warning
                log('Eval Epoch [{}/{}] ({}) Acc@1:{:.2f}% Acc@5:{:.2f}%'.format(epoch, self.max_epoch, desc, total_top1 / total_num * 100, total_top5 / total_num * 100))
                
        self.model.train()
        return
    
   

    
    def test_unbiased(self, epoch, train_eval=True):
        self.model.eval()
            
        test_envs = ['valid', 'test']
        for desc in test_envs:
            loader = self.loaders[desc]
            
            total_top1, total_top5, total_num, test_bar = 0.0, 0.0, 0, tqdm(loader, ncols=100)
            
            num_classes = len(loader.dataset.classes)
            num_groups = loader.dataset.num_groups
            
            bias_counts = torch.zeros(num_groups).cuda()
            bias_top1s = torch.zeros(num_groups).cuda()
            
            
            with torch.no_grad():
                
                features, labels = [], []
                logits = []
                corrects = []

                for data, target, biases, group, _, _, ids in test_bar:
                    data, target, biases, group = data.cuda(), target.cuda(), biases.cuda(), group.cuda()
                    
                    B = target.size(0)

                    results = self.model(data)
                    pred_labels = results["out"].argsort(dim=-1, descending=True)
                    features.append(results["feature"])
                    logits.append(results["out"])
                    labels.append(group)
                
            
                    top1s = (pred_labels[:, :1] == target.unsqueeze(dim=-1)).squeeze().unsqueeze(0)
                    group_indices = (group==torch.arange(num_groups).unsqueeze(1).long().cuda())
                    
                    bias_counts += group_indices.sum(1)
                    bias_top1s += (top1s * group_indices).sum(1)
                    
                    corrects.append(top1s)
                    
                    total_num += B
                    total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    acc1, acc5 = total_top1 / total_num * 100, total_top5 / total_num * 100
                    
                    bias_accs = bias_top1s / bias_counts * 100

                    avg_acc = np.nanmean(bias_accs.cpu().numpy())
                    worst_acc = np.nanmin(bias_accs.cpu().numpy())
                    std_acc = np.nanstd(bias_accs.cpu().numpy())
                    
                    acc_desc = '/'.join(['{:.1f}%'.format(acc) for acc in bias_accs])
                    
                    test_bar.set_description('Eval Epoch [{}/{}] [{}] Bias: {:.2f}%'.format(epoch, self.max_epoch, desc, avg_acc))
                
                features = torch.cat(features)
                logits = torch.cat(logits)
                labels = torch.cat(labels)
                corrects = torch.cat(corrects, 1)
                
            
            log = self.logger.info if desc in ['train', 'train_eval'] else self.logger.warning
            log('Eval Epoch [{}/{}] [{}] Unbiased: {:.2f}% (std: {:.2f}), Worst: {:.2f}% [{}] (Average: {:.2f}%)'.format(epoch, self.max_epoch, desc, avg_acc, std_acc, worst_acc, acc_desc))
            self.logger.info('Total [{}]: Acc@1:{:.2f}% Acc@5:{:.2f}%'.format(desc, acc1, acc5))
            print("               {} / {} / {}".format(self.args.desc, self.args.target_attr, self.args.bias_attrs))

                    
        self.model.train()
        
        return
    
    
    
    

