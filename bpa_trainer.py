
import copy
from trainer import *
from kmeans_pytorch import kmeans
from modules.centroids import AvgFixedCentroids   
from torch.autograd import Variable


class OnlineTrainer(BiasedClassifyTrainer):
    
    def __init__(self, args, model, loaders, optimizer, num_classes, per_clusters=0):
        super().__init__(args, model, loaders, optimizer, num_classes)
        if per_clusters == 0:
            per_clusters = args.k
        
        self.centroids = AvgFixedCentroids(args, num_classes, per_clusters=per_clusters)
        self.update_cluster_iter = args.update_cluster_iter
        self.checkpoint_dir = args.checkpoint_dir
        
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
            self.logger.info('Resume from {}'.format(epoch))
            checkpoint_path = self.checkpoint_dir / 'e{:04d}.pth'.format(epoch)
            
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict']) # Set CUDA before if error occurs.
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
    

    
class BPATrainer(OnlineTrainer):
    
    def __init__(self, args, model, loaders, optimizer, num_classes):
        super().__init__(args, model, loaders, optimizer, num_classes)
        self.class_weights = None
        self.base_model = copy.deepcopy(self.model)
        if not args.use_base:
            assert ValueError
        
    def save_model(self, epoch):
        if not self.args.no_save or epoch % self.args.save_epoch == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict(), 
                'optimizer' : self.optimizer.state_dict(),
            }, self.checkpoint_dir / 'e{:04d}.pth'.format(epoch))
            return
        
        
    def use_base_model(self, file_name):
        self.logger.info('Loading ({}) base model'.format(file_name))
        checkpoint_path = self.checkpoint_dir / '..' / '{}.pth'.format(file_name)
        checkpoint = torch.load(checkpoint_path)
        self.base_model.load_state_dict(checkpoint['state_dict']) # Set CUDA before if error occurs.
    
    
    def load_model(self, epoch=0):
        self.logger.info('Resume training')
        if epoch==0:
            checkpoint_path = max((f.stat().st_mtime, f) for f in self.checkpoint_dir.glob('*.pth'))[1]
            self.logger.info('Resume Latest from {}'.format(checkpoint_path))
        else:
            self.logger.info('Resume from {}'.format(epoch))
            checkpoint_path = self.checkpoint_dir / 'e{:04d}.pth'.format(epoch)
            
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict']) # Set CUDA before if error occurs.
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        
        
    def _extract_features(self, model, data_loader):
        features, targets = [], []
        ids = []

        for data, target, index in tqdm(data_loader, desc='Feature extraction for clustering..', ncols=5):
            data, target, index = data.cuda(), target.cuda(), index.cuda()
            results = model(data)
            features.append(results["feature"])
            targets.append(target)
            ids.append(index)

        features = torch.cat(features)
        targets = torch.cat(targets)
        ids = torch.cat(ids)
        return features, targets, ids
        
    
    def _cluster_features(self, data_loader, features, targets, ids, num_clusters):
        
        N = len(data_loader.dataset)
        num_classes = data_loader.dataset.num_classes
        sorted_target_clusters = torch.zeros(N).long().cuda() + num_clusters*num_classes

        target_clusters = torch.zeros_like(targets)-1
        cluster_centers = []

        for t in range(num_classes):
            target_assigns = (targets==t).nonzero().squeeze()
            feautre_assigns = features[target_assigns]

            cluster_ids, cluster_center = kmeans(X=feautre_assigns, num_clusters=num_clusters, distance='cosine', tqdm_flag=False, device=0)
            cluster_ids_ = cluster_ids + t*num_clusters

            target_clusters[target_assigns] = cluster_ids_.cuda()
            cluster_centers.append(cluster_center)

        sorted_target_clusters[ids] = target_clusters
        cluster_centers = torch.cat(cluster_centers, 0)
        return sorted_target_clusters, cluster_centers
        
        
    def inital_clustering(self):
        data_loader = self.loaders['train_eval']
        data_loader.dataset.clustering_on()
        self.base_model.eval()
        
        with torch.no_grad():

            features, targets, ids = self._extract_features(self.base_model, data_loader)
            num_clusters = self.args.num_clusters
            cluster_assigns, cluster_centers = self._cluster_features(data_loader, features, targets, ids, num_clusters)
            
            cluster_counts = cluster_assigns.bincount().float()
            print("Cluster counts : {}, len({})".format(cluster_counts, len(cluster_counts)))

            
        data_loader.dataset.clustering_off()                 
        return cluster_assigns, cluster_centers


    def get_metric_loss(self, data):
        return 0.

    
    def train(self, epoch):
        
        cluster_weights = None
#         if epoch > 1 and not self.centroids.initialized:
        if not self.centroids.initialized:
            cluster_assigns, cluster_centers = self.inital_clustering()
            self.centroids.initialize_(cluster_assigns, cluster_centers)
        
        data_loader = self.loaders['train']
        total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        total_metric_loss = 0.0
        
        i = 0
        for data, target, _, _, cluster, weight_pre, ids in train_bar:
            i += 1
            B = target.size(0)
                
            data, target = data.cuda(), target.cuda()
            
            results = self.model(data)
            weight = self.centroids.get_cluster_weights(ids)
            loss = torch.mean(criterion(results["out"], target.long()) * (weight))

            self.optimizer.zero_grad()
            (loss).backward()
            self.optimizer.step()

            total_num += B
            total_loss += loss.item() * B

            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.max_epoch, total_loss / total_num))
            
            if self.centroids.initialized:
                self.centroids.update(results, target, ids)
                if self.args.update_cluster_iter > 0 and i % self.args.update_cluster_iter == 0:
                    self.centroids.compute_centroids()
                    
        
        if self.centroids.initialized:
            self.centroids.compute_centroids(verbose=True)
            
            
        return total_loss / total_num

    
    def test_unbiased(self, epoch, train_eval=True):
        self.model.eval()
    
        test_envs = ['valid', 'test']
            
        for desc in test_envs:
            loader = self.loaders[desc]
            
            total_top1, total_top5, total_num, test_bar = 0.0, 0.0, 0, tqdm(loader)
            
            num_classes = len(loader.dataset.classes)
            num_groups = loader.dataset.num_groups
            
            bias_counts = torch.zeros(num_groups).cuda()
            bias_top1s = torch.zeros(num_groups).cuda()
            
            with torch.no_grad():
                
                features, labels = [], []
                corrects = []

                for data, target, biases, group, _, _, ids in test_bar:
                    data, target, biases, group = data.cuda(), target.cuda(), biases.cuda(), group.cuda()
                    
                    B = target.size(0)
                    num_groups = np.power(num_classes, biases.size(1)+1)

                    results = self.model(data)
                    pred_labels = results["out"].argsort(dim=-1, descending=True)
                    features.append(results["feature"])
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
                    
                    acc_desc = '/'.join(['{:.1f}%'.format(acc) for acc in bias_accs])
                    
                    test_bar.set_description('Eval Epoch [{}/{}] [{}] Bias: {:.2f}%'.format(epoch, self.max_epoch, desc, avg_acc))
                    
            
            log = self.logger.info if desc in ['train', 'train_eval'] else self.logger.warning
            log('Eval Epoch [{}/{}] [{}] Unbiased: {:.2f}% [{}]'.format(epoch, self.max_epoch, desc, avg_acc, acc_desc))
            self.logger.info('Total [{}]: Acc@1:{:.2f}% Acc@5:{:.2f}%'.format(desc, acc1, acc5))
            print("               {} / {} / {}".format(self.args.desc, self.args.target_attr, self.args.bias_attrs))

                    
        self.model.train()
        return
    