import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import pdb
from utils.train_utils import *

    
    
class ResNet(nn.Module):
    def __init__(self, feature_dim, num_classes, arch='', feature_fix=False):
        super(ResNet, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.arch = arch
        resnet = self.get_backbone()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 64
        self.res3 = resnet.layer2 # 1/8, 128
        self.res4 = resnet.layer3 # 1/16, 256
        self.res5 = resnet.layer4 # 1/32, 512
        
        self.f = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                               self.res2, self.res3, self.res4, self.res5)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # classifier
        self.fc = self.get_fc(num_classes)
    
    
        self.feature_fix = feature_fix
        if feature_fix:
            print("Fix parameters except fc layer")
            for param in self.parameters():
                param.requires_grad = False

            self.fc.weight.requires_grad = True
            self.fc.bias.requires_grad = True


    def get_backbone(self):
        raise NotImplementedError
        
    def get_fc(self, num_classes):
        raise NotImplementedError
        
    def forward(self, x):

        feature = self.f(x)
        feature = torch.flatten(self.avgpool(feature), start_dim=1)
        
        logits = self.fc(feature)

        results = {
            "out": logits,
            "feature": feature,
        }
        return results
    
    
class ResNet18(ResNet):
    
    def get_backbone(self):
        return torchvision.models.resnet18(pretrained=True)
    
    def get_fc(self, num_classes):
        return nn.Linear(512, num_classes, bias=True)
    
    
    
    