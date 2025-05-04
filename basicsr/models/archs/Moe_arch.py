import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass
from pdb import set_trace as stx
from torchvision import models


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.ReLU(),
            nn.Linear(channel//reduction, channel),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CNNBlock, self).__init__()
        self.up_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        channels = out_channels
        self.conv = nn.Sequential(
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(channels)
        )
        self.se = SEBlock(channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.up_conv(x)
        x = F.relu(x + self.conv(x))
        x = self.se(x)
        x = self.pool(x)
        return x

    
class Moe(nn.Module):
    def __init__(self, opt):
        super(Moe, self).__init__()
        N = opt['num_models']
        C = opt['channels']
        self.conv_layers = nn.ModuleList([
            CNNBlock(3, 32),
            CNNBlock(32, 64),
            CNNBlock(64, 128)
        ])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, N)
        )
        print(f"total parameter num of MOE: {count_parameters(self)}, -------------------------", flush=True)

    def forward(self, x, xn=None):
        for layer in self.conv_layers:
            # print(x.shape)
            x = layer(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        w = F.softmax(x, dim=-1)
        if xn==None:
            return w # for ennet
        xn = xn * w[..., None, None, None]
        restored = torch.sum(xn, dim=1)
        return restored
    

if __name__ == '__main__':
    B, C, H, W = 4, 3, 384, 384
    N = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = {'num_models': N, 'channels': 3, 'block_size': 8, 
           'conv_dim': 64, 'kernal_sizes': [3,5,7,9], 'down_channels': [48, 96], # 18,48,96,192
           'emb_dim': 128, 'n_layers': 2, 'up_channels': [64, 16]} # 192, 64, 16, 3
    # opt = {'channels': 3, 'block_size': 16, 'kernal_sizes': [3,7,11,15], 'dim': 512, 'n_layers': 2, 'num_models': N}

    model = Moe(opt)
    x = torch.randn(B, C, H, W)
    xn = torch.randn(B, N, C, H, W)
    res = model(x, xn)
    print(res.shape)