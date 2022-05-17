import torch, pdb
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
from kmeans_pytorch import kmeans

from utils.train_utils import grad_mul_const


class Centroids(nn.Module):
    
    def __init__(self, args, num_classes, per_clusters, feature_dim=None):
        super(Centroids, self).__init__()
        self.momentum = args.momentum
        self.per_clusters = per_clusters
        self.num_classes = num_classes
        self.feature_dim = args.feature_dim if feature_dim is None else feature_dim
        
        # Cluster
        self.cluster_means = None
        self.cluster_vars = torch.zeros((self.num_classes, self.per_clusters))
        self.cluster_losses = torch.zeros((self.num_classes, self.per_clusters))
        self.cluster_accs = torch.zeros((self.num_classes, self.per_clusters))
        self.cluster_weights = torch.zeros((self.num_classes, self.per_clusters))
        
        # Sample
        self.feature_bank = None
        self.assigns = None
        self.corrects = None
        self.losses = None
        self.weights = None
        
        self.initialized = False
        self.weight_type = args.cluster_weight_type
        
        self.max_cluster_weights = 0. # 0 means no-limit

    def __repr__(self):
        return "{}(Y{}/K{}/dim{})".format(self.__class__.__name__, self.num_classes, self.per_clusters, self.feature_dim)
    
    @property
    def num_clusters(self):
        return self.num_classes * self.per_clusters
            
    @property 
    def cluster_counts(self):
        if self.assigns is None:
            return 0
        return self.assigns.bincount(minlength=self.num_clusters)
    
    
    def _clamp_weights(self, weights):
        if self.max_cluster_weights > 0:
            if weights.max() > self.max_cluster_weights:
                scale = np.log(self.max_cluster_weights)/torch.log(weights.cpu().max())
                scale = scale.cuda()
                print("> Weight : {:.4f}, scale : {:.4f}".format(weights.max(), scale))
                return weights ** scale
        return weights
        
    
    def get_cluster_weights(self, ids):
        if self.assigns is None:
            return 1
        
        cluster_counts = self.cluster_counts + (self.cluster_counts==0).float() # avoid nans
            
        cluster_weights = cluster_counts.sum()/(cluster_counts.float())
        assigns_id = self.assigns[ids]

        if (self.losses == -1).nonzero().size(0) == 0:
            cluster_losses_ = self.cluster_losses.view(-1)
            losses_weight = cluster_losses_.float()/cluster_losses_.sum()
            weights_ = cluster_weights[assigns_id] * losses_weight[assigns_id].cuda()
            weights_ /= weights_.mean()
        else:
            weights_ = cluster_weights[assigns_id]
            weights_ /= weights_.mean()
                 
        return self._clamp_weights(weights_)
    
    
    def initialize_(self, cluster_assigns, cluster_centers, sorted_features=None):
        cluster_means = cluster_centers.detach().cuda()
        cluster_means = F.normalize(cluster_means, 1)
        self.cluster_means = cluster_means.view(self.num_classes, self.per_clusters, -1)
        
        N = cluster_assigns.size(0)
        self.feature_bank = torch.zeros((N, self.feature_dim)).cuda() if sorted_features is None else sorted_features
        self.assigns = cluster_assigns
        self.corrects = torch.zeros((N)).long().cuda() - 1
        self.losses = torch.zeros((N)).cuda() - 1
        self.weights = torch.ones((N)).cuda()
        self.initialized = True
        
        
    def get_variances(self, x, y):
        return 1 - (y @ x).mean(0)
        
    def compute_centroids(self, verbose=False, split=False):
        for y in range(self.num_classes):
            for k in range(self.per_clusters): 
                l = y*self.per_clusters + k
                ids = (self.assigns==l).nonzero()
                if ids.size(0) == 0:
                    continue
                self.cluster_means[y, k] = self.feature_bank[ids].mean(0)
                self.cluster_vars[y, k] = self.get_variances(self.cluster_means[y, k], self.feature_bank[ids])

                corrs = self.corrects[ids]
                corrs_nz = (corrs[:, 0]>=0).nonzero()
                if corrs_nz.size(0) > 0:
                    self.cluster_accs[y, k] = corrs[corrs_nz].float().mean(0)

                losses = self.losses[ids]
                loss_nz = (losses[:, 0]>=0).nonzero()
                if loss_nz.size(0) > 0:
                    self.cluster_losses[y, k] = losses[loss_nz].float().mean(0)
            
        return 

                 
    def update(self, results, target, ids, features=None):
        assert self.initialized
        
        ### update feature and assigns
        feature = results["feature"] if features is None else features
        feature_ = F.normalize(feature, 1).detach()
       
        feature_new = (1-self.momentum) * self.feature_bank[ids] + self.momentum * feature_
        feature_new = F.normalize(feature_new, 1)
        
        self.feature_bank[ids] = feature_new

        sim_score = self.cluster_means @ feature_new.permute(1, 0) # YKC/CB => YKB
            
        for y in range(self.num_classes):
            sim_score[y, :, (target!=y).nonzero()] -= 1e4
            
        sim_score_ = sim_score.view(self.num_clusters, -1)
        new_assigns = sim_score_.argmax(0)
        self.assigns[ids] = new_assigns
        
        corrects = (results["out"].argmax(1) == target).long()
        self.corrects[ids] = corrects
        
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        losses = criterion(results["out"], target.long()).detach()
        self.losses[ids] = losses
        
        return
    

        
class FixedCentroids(Centroids):
        
    def compute_centroids(self, verbose='', split=False):
        
        for y in range(self.num_classes):
            for k in range(self.per_clusters): 
                l = y*self.per_clusters + k
                
                ids = (self.assigns==l).nonzero()
                if ids.size(0) == 0:
                    continue

                corrs = self.corrects[ids]
                corrs_nz = (corrs[:, 0]>=0).nonzero()
                if corrs_nz.size(0) > 0:
                    self.cluster_accs[y, k] = corrs[corrs_nz].float().mean(0)

                losses = self.losses[ids]
                loss_nz = (losses[:, 0]>=0).nonzero()
                if loss_nz.size(0) > 0:
                    self.cluster_losses[y, k] = losses[loss_nz].float().mean(0)
                    
                self.cluster_weights[y, k] = self.weights[ids].float().mean(0)
     
        return 
                
    def get_cluster_weights(self, ids):
        weights_ids = super().get_cluster_weights(ids)
        self.weights[ids] = weights_ids
        return weights_ids

    
    def update(self, results, target, ids, features=None, preds=None):
        assert self.initialized
        
        out = preds if preds is not None else results["out"]
        
        corrects = (out.argmax(1) == target).long()
        self.corrects[ids] = corrects
        
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        losses = criterion(out, target.long()).detach()
        self.losses[ids] = losses
        
        return
    
    
    
class AvgFixedCentroids(FixedCentroids):
    
    def __init__(self, args, num_classes, per_clusters, feature_dim=None):
        super(AvgFixedCentroids, self).__init__(args, num_classes, per_clusters, feature_dim)
        self.exp_step = args.exp_step
        self.avg_weight_type = args.avg_weight_type
        
    def compute_centroids(self, verbose='', split=False):
        
        for y in range(self.num_classes):
            for k in range(self.per_clusters): 
                l = y*self.per_clusters + k
                
                ids = (self.assigns==l).nonzero()
                if ids.size(0) == 0:
                    continue

                corrs = self.corrects[ids]
                corrs_nz = (corrs[:, 0]>=0).nonzero()
                if corrs_nz.size(0) > 0:
                    self.cluster_accs[y, k] = corrs[corrs_nz].float().mean(0)

                losses = self.losses[ids]
                loss_nz = (losses[:, 0]>=0).nonzero()
                if loss_nz.size(0) > 0:
                    self.cluster_losses[y, k] = losses[loss_nz].float().mean(0)
                    
                self.cluster_weights[y, k] = self.weights[ids].float().mean(0)
                    
        return 
        
        
    def get_cluster_weights(self, ids):
        
        weights_ids = super().get_cluster_weights(ids)
        
        if self.avg_weight_type == 'expavg':
            weights_ids_ = self.weights[ids] * torch.exp(self.exp_step*weights_ids.data)
        elif self.avg_weight_type == 'avg':
            weights_ids_ = (1-self.momentum) * self.weights[ids] + self.momentum * weights_ids
        elif self.avg_weight_type == 'expgrad':
            weights_ids_l1 = weights_ids / weights_ids.sum()
            prev_ids_l1 = self.weights[ids] / self.weights[ids].sum()
            weights_ids_ = prev_ids_l1 * torch.exp(self.exp_step*weights_ids_l1.data)
        else:
            raise ValueError
            
        self.weights[ids] = weights_ids_ / weights_ids_.mean()
        return self.weights[ids]  

    
    
class HeteroCentroids(nn.Module):

    def __init__(self, args, num_classes, num_hetero_clusters, centroids_type):
        super(HeteroCentroids, self).__init__()
        self.momentum = args.momentum
        self.num_classes = num_classes
        self.feature_dim = args.feature_dim
        self.initialized = False
        
        self.num_hetero_clusters = num_hetero_clusters
        self.num_multi_centroids = len(num_hetero_clusters)
        self.centroids_list = [centroids_type(args, num_classes, per_clusters=num_hetero_clusters[m], feature_dim=self.feature_dim) for m in range(self.num_multi_centroids)]

    def __repr__(self):
        return self.__class__.__name__ + "(" +  ", ".join([centroids.__repr__() for centroids in self.centroids_list])+ ")"
    
    
    def initialize_multi(self, multi_cluster_assigns, multi_cluster_centers):
        for cluster_assigns, cluster_centers, centroids in zip(multi_cluster_assigns, multi_cluster_centers, self.centroids_list):
            centroids.initialize_(cluster_assigns, cluster_centers)
        self.initialized = True
        
    def compute_centroids(self, verbose=False, split=False):
        for m, centroids in enumerate(self.centroids_list):
            verbose_ = str(m) if verbose else ''
            centroids.compute_centroids(verbose=verbose_)

    
    def update(self, results, target, ids):
        for m, centroids in enumerate(self.centroids_list):
            features = results["feature"]
            centroids.update(results, target, ids)
        
        
    def get_cluster_weights(self, ids):
        
        weights_list = [centroids.get_cluster_weights(ids) for centroids in self.centroids_list]
        weights_ = torch.stack(weights_list).mean(0)
        return weights_
        
        
        
        
        
        
        