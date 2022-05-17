import torch, pdb
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def get_negative_class_mask(targets):
    batch_size = targets.size(0)
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        current_c = targets[i]
        same_indices = (targets == current_c).nonzero().squeeze()
        for s in same_indices:
            negative_mask[i, s] = 0
            negative_mask[i, s + batch_size] = 0
            
    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask
        


class GradMulConst(torch.autograd.Function):
    """
    This layer is used to create an adversarial loss.
    """
    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.const, None

def grad_mul_const(x, const):
    return GradMulConst.apply(x, const)



class AvgMeter(object):
    def __init__(self, name=''):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        if type(val) is torch.Tensor:
            val = val.detach()
            val = val.cpu()
            val = val.numpy()

        if n==len(val):
            self.val = val[-1]
            self.sum += np.sum(val)
            self.count += len(val)
        elif n==0: # array
            self.val = val[-1]
            self.sum += np.sum(val)
            self.count += len(val)
        else:
            self.val = val
            self.sum += val
            self.count += n
        
        self.avg = self.sum / self.count
        
    def __repr__(self):
        return self.name+":"+str(round(self.avg, 3))
    
    
class WindowAvgMeter(object):
    def __init__(self, name='', max_count=20):
        self.values = []
        self.name = name
        self.max_count = max_count

    @property
    def avg(self):
        if len(self.values) > 0:
            return np.sum(self.values)/len(self.values)
        else:
            return 0

    def update(self, val):
        if type(val) is torch.Tensor:
            val = val.detach()
            val = val.cpu()
            val = val.numpy()

        self.values.append(val)
        if len(self.values) > self.max_count:
            self.values.pop(0)
        
    def __repr__(self):
        return self.name+":"+str(round(self.avg, 3))
    
    
class EMA(object):
    # Exponential Moving Average
    
    def __init__(self, label, alpha=0.9):
        self.label = label
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))
        
    def update(self, data, index):
        self.parameter[index] = self.alpha * self.parameter[index] + (1-self.alpha*self.updated[index]) * data
        self.updated[index] = 1
        
    def max_loss(self, label):
        label_index = np.where(self.label == label)[0]
        return self.parameter[label_index].max()
    
    
    

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, 1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

        
        
def KL_u_p_loss(outputs):
    # KL(u||p)
    num_classes = outputs.size(1)
    uniform_tensors = torch.ones(outputs.size())
    uniform_dists = Variable(uniform_tensors / num_classes).cuda()
    instance_losses = F.kl_div(F.log_softmax(outputs, dim=1), uniform_dists, reduction='none').sum(dim=1)
    return instance_losses
