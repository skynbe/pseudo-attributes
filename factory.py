from typing import Any, Callable, Dict, Iterable, List, Optional
from pathlib import Path
from dataset.celebA import CelebA, get_celebA_dataloader

import torchvision
from modules.transform import *
from models.classification import ResNet18

from trainer import BiasedClassifyTrainer
from bpa_trainer import BPATrainer
from torch.utils.data import DataLoader
import pdb



class Factory(object):

    PRODUCTS: Dict[str, Callable] = {}

    def __init__(self):
        raise ValueError(
            f"""Cannot instantiate {self.__class__.__name__} object, use
            `create` classmethod to create a product from this factory.
            """
        )

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Any:
        r"""Create an object by its name, args and kwargs."""
        if name not in cls.PRODUCTS:
            raise KeyError(f"{cls.__class__.__name__} cannot create {name}.")

        return cls.PRODUCTS[name](*args, **kwargs)
    
    
    
class ModelFactory(Factory):


    MODELS: Dict[str, Callable] = {
        "ResNet18": ResNet18,
    }

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Any:
        
        return cls.MODELS[name](*args, **kwargs)
    
    
    

class TransformFactory(Factory):
    PRODUCTS: Dict[str, Callable] = {
        
        "train": train_transform,
        "test": test_transform,

        "celebA_train": celebA_train_transform,
        "celebA_test": celebA_test_transform,
        
    }
        
    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Any:
        r"""Create an object by its name, args and kwargs."""
        if name not in cls.PRODUCTS:
            raise KeyError(f"{cls.__class__.__name__} cannot create {name}.")

        return cls.PRODUCTS[name]
        
    
    
        
        
class DataLoaderFactory(Factory):

    @classmethod
    def create(cls, name: str, trainer: str, batch_size: int, num_workers: int, configs: Any, cluster_ids: Any = None) -> Any:
        
        if name == 'celebA':
            
            train_loader, train_set = get_celebA_dataloader(
                root=Path('./data/celebA'), batch_size=batch_size, split='train',
                target_attr=configs.target_attr, bias_attrs=configs.bias_attrs, 
                cluster_ids=cluster_ids, args=configs)
            valid_loader, valid_set = get_celebA_dataloader(
                root=Path('./data/celebA'), batch_size=batch_size, split='valid',
                target_attr=configs.target_attr, bias_attrs=configs.bias_attrs,
                args=configs)
            test_loader, test_set = get_celebA_dataloader(
                root=Path('./data/celebA'), batch_size=batch_size, split='test',
                target_attr=configs.target_attr, bias_attrs=configs.bias_attrs,
                args=configs)
            train_eval_loader, train_eval_set = get_celebA_dataloader(
                root=Path('./data/celebA'), batch_size=batch_size, split='train_eval',
                target_attr=configs.target_attr, bias_attrs=configs.bias_attrs,
                args=configs)

            datasets = {
                'train': train_set,
                'valid': valid_set,
                'test': test_set,
                'train_eval': train_eval_set,
            }

            data_loaders = {
                'train': train_loader,
                'valid': valid_loader,
                'test': test_loader,
                'train_eval': train_eval_loader,
            }
                
            
        else:
            raise ValueError
        
        
        return data_loaders, datasets
        

    
class TrainerFactory(Factory):

    TRAINERS: Dict[str, Callable] = {
        "classify": BiasedClassifyTrainer,
        "bpa": BPATrainer,
    }

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Any:
        
        return cls.TRAINERS[name](*args, **kwargs)
    