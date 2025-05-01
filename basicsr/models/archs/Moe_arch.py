import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass
from pdb import set_trace as stx
from torchvision import models

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# class Moe(nn.Module):
#     def __init__(self, opt, N=6, C=3):
#         super(Moe, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #感受野3，5，7 是不是变化太小了
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(128, N)
#         print("total parameter num of MOE: ", count_parameters(self), '-------------------------') # 

#     def forward(self, x):
#         x = self.conv_layers(x) 
#         x = self.global_avg_pool(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.fc(x) 
#         return F.softmax(x, dim=-1)

class Moe(nn.Module):
    def __init__(self, opt, N=6, C=3):
        super(Moe, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(C, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), #感受野3，5，7 是不是变化太小了
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) #？
        self.fc = nn.Linear(32, N)
        print("total parameter num of MOE: ", count_parameters(self), '-------------------------') # 

    def forward(self, x):
        x = self.conv_layers(x) 
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x) 
        return F.softmax(x, dim=-1)

# class Moe(nn.Module):
#     def __init__(self, output_dim=6, pretrained=True, freeze_backbone=False):
#         super().__init__()
#         # Backbone: ResNet-18 (移除非必要模块)
#         backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
#         self.feature_extractor = nn.Sequential(*list(backbone.children()))[:-2]  # 移除最后两层（全局池化和FC）
        
#         # 自适应全局平均池化（适应任意输入尺寸）
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # 全连接层（输出权重）
#         self.fc = nn.Linear(backbone.fc.in_features, output_dim)
        
#         # 可选：冻结骨干网络参数
#         if freeze_backbone:
#             for param in self.feature_extractor.parameters():
#                 param.requires_grad = False
    
#     def forward(self, x):
#             # 特征提取
#             features = self.feature_extractor(x)  # [B, 512, H/32, W/32]
            
#             # 全局平均池化
#             pooled = self.adaptive_pool(features)  # [B, 512, 1, 1]
#             flattened = torch.flatten(pooled, 1)   # [B, 512]
            
#             # 生成权重
#             weights = self.fc(flattened)           # [B, output_dim]
#             return F.softmax(weights, dim=-1)