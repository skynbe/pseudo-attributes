import os, pdb
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
# from models import model_attributes
from torch.utils.data import Dataset, Subset
from pathlib import Path
import torchvision
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler



class CelebA(torchvision.datasets.CelebA):
    
  # Attributes : '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    
    def __init__(self, root, split="train", target_type="attr", transform=None,
                 target_transform=None, download=False,
                 target_attr='', aux_attrs=[], bias_attrs=[], domain_attr=None, domain_type=None,
                 pair=False, args=None, scale=1.0): 
        
        super().__init__(root, split=split, target_type=target_type, transform=transform,
                 target_transform=target_transform, download=download)
        
        self.target_attr = target_attr
        self.aux_attrs = aux_attrs
        self.bias_attrs = bias_attrs
        self.domain_attr = domain_attr
        
        self.target_idx = self.attr_names.index(target_attr)
        self.aux_indices = [self.attr_names.index(aux_att) for aux_att in aux_attrs] if aux_attrs else []
        self.domain_idx = self.attr_names.index(domain_attr) if domain_attr else None
        
        self.domain_type = domain_type
        
        self.bias_indices = [self.attr_names.index(bias_att) for bias_att in bias_attrs]
    
        self.cluster_ids = None
        self.clustering = False
        self.sample_weights = None
        
        self.pair = pair
        
        self.visualize_image = False
        
        self.args = args
        self.visualize = False
        self.scale = scale
        
    
    
    @property
    def class_elements(self):
        return self.attr[:, self.target_idx]
    
    @property
    def group_elements(self):
        group_attrs = self.attr[:, [self.target_idx]+self.bias_indices]
        weight = np.power(self.num_classes, np.arange(group_attrs.size(1)))
        group_elems = (group_attrs*weight).sum(1)
        return group_elems
    
    @property
    def group_counts(self):
        group_attrs = self.attr[:, [self.target_idx]+self.bias_indices]
        weight = np.power(self.num_classes, np.arange(group_attrs.size(1)))
        group_elems = (group_attrs*weight).sum(1)
        return group_elems.bincount()
    
    
    def group_counts_with_attr(self, attr):
        target_idx = self.attr_names.index(attr)
        group_attrs = self.attr[:, [target_idx]+self.bias_indices]
        weight = np.power(self.num_classes, np.arange(group_attrs.size(1)))
        group_elems = (group_attrs*weight).sum(1)
        return group_elems.bincount()
    
    def visualize(self):
        self.visualize_image = True
        print("For visualize images with t-SNE plots.")
    
    def clustering_on(self):
        self.clustering = True
        print("Enable clustering")
    
    def clustering_off(self):
        self.clustering = False
        print("Unable clustering")
    
    def update_clusters(self, cluster_ids):
        self.cluster_ids = cluster_ids   
        
    def update_weights(self, weights):
        self.sample_weights = weights 
        
        
    
    def __len__(self):
        len = super().__len__()
        if self.scale < 1.0:
            len = int(len*self.scale)
        return len
    
    
    def get_sample_index(self, index):
        return index
    
    def __getitem__(self, index_):
        
        index = self.get_sample_index(index_)
        
        img_path = os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index])
        img_ = Image.open(img_path)

        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None
        
        target_attr = target[self.target_idx]
        bias_attrs = np.array([target[bias_idx] for bias_idx in self.bias_indices])
        group_attrs = np.insert(bias_attrs, 0, target_attr)  # target first
        
        bit = np.power(self.num_classes, np.arange(len(group_attrs)))
        group = np.sum(bit * group_attrs)
            
            
        if self.cluster_ids is not None:
            cluster = self.cluster_ids[index]
        else:
            cluster = -1
            
        if self.sample_weights is not None:
            weight = self.sample_weights[index]
        else:
            weight = 1
            
            
        if self.transform is not None:
            transform = self.transform
            img = transform(img_)
            if self.pair:
                img2 = transform(img_)

                
        if self.visualize_image is True:
            return img, target_attr, index, img_path
    
        # for clustering
        if self.clustering is True:
            return img, target_attr, index
        
        
        if len(self.aux_indices) > 0:
            aux_attrs = np.array([target[aux_index] for aux_index in self.aux_indices])
            return img, target_attr, aux_attrs, bias_attrs, group, cluster, weight, index
            
        if self.visualize:
            return img, target_attr, bias_attrs, group, cluster, weight, index, img_path
            
        
        return img, target_attr, bias_attrs, group, cluster, weight, index
        
    
    @property
    def classes(self):
        return ['0', '1']
    
    @property
    def num_classes(self):
        return len(self.classes)
    
    @property
    def num_groups(self):
        return np.power(len(self.classes), len(self.bias_attrs)+1)
    
    @property
    def bias_attributes(self):
        return
    
    @property
    def attribute_names(self):
        return ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    
    


def get_celebA_dataloader(root, batch_size, split, target_attr, bias_attrs, aux_attrs=None, num_workers=4, pair=False, cluster_ids=None, args=None):

    from factory import TransformFactory
    
    ### Transform and scale
    if split in ['train', 'train_target']:
        celebA_transform = TransformFactory.create("celebA_train")

    elif split in ['valid', 'test', 'train_eval']:
        celebA_transform = TransformFactory.create("celebA_test")
    
    ### Dataset split
    celebDataset = CelebA
    if split in ['train', 'train_eval']:
        dataset_split = 'train'
    elif split in ['valid']:
        dataset_split = 'valid'
    elif split in ['test']:
        dataset_split = 'test'
    

    dataset = celebDataset(root, split=dataset_split, transform=celebA_transform, download=True,
                           target_attr=target_attr, bias_attrs=bias_attrs, aux_attrs=aux_attrs, args=args)
        

    dataloader = data.DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=num_workers,
                         pin_memory=True)

    return dataloader, dataset
    
    