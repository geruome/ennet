import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass
from pdb import set_trace as stx
# import cv2


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


class MOE(nn.Module):
    def __init__(self, N=6, C=3):
        super(MOE, self).__init__()
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
        print("total parameter num of MOE: ", count_parameters(self), '-------------------------')

    def forward(self, x):
        for layer in self.conv_layers:
            # print(x.shape)
            x = layer(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.softmax(x, dim=-1)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x): # B,T,C
        B,T,C = x.shape
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2) # split at dim 2 with length self.n_emb
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # B,T,nh,hs -> B,nh,T,hs
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # B,T,nh,hs -> B,nh,T,hs
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # B,T,nh,hs -> B,nh,T,hs
        attn = q @ k.transpose(-2, -1) * (1./math.sqrt(k.shape[-1])) # B,nh,T,T
        attn = F.softmax(attn, dim=-1) 
        x = attn @ v # B,nh,T,hs
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)        

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class AttentionConfig:
    n_embd: int
    n_head: int = 1


class MRFFI(nn.Module):
    def __init__(self, opt):
        super(MRFFI, self).__init__()
        self.N = opt['num_models']
        # self.linear = nn.Linear(self.N, 1, bias=False)

        self.channels = opt['channels']
        self.emb_dim = opt['emb_dim']
        self.P = opt['block_size']
        assert self.P%4==0, 'block_size must be multiple of 4 !!!'

        stride = self.P//2
        padding = stride//2
        in_channels = self.channels*self.N
        attn_config = AttentionConfig(**dict(n_embd=self.emb_dim))

        # self.linear = nn.Conv2d(in_channels, self.channels, 1)
        self.linear = nn.Linear(self.N, 1, bias=False)

        self.conv = nn.Conv2d(in_channels, self.emb_dim, kernel_size=self.P, stride=stride, padding=padding)
        self.up_conv = nn.ConvTranspose2d(self.emb_dim, self.channels, kernel_size=self.P, stride=stride, padding=padding)
        self.attn_layers = nn.ModuleList( Block(attn_config) for _ in range(opt['n_layers']))

        self.ksizes = opt['kernal_sizes']
        assert all(i%2==1 for i in self.ksizes)
        ch = self.emb_dim
        assert ch%len(self.ksizes)==0
        ch = ch//len(self.ksizes)
        self.convs = nn.ModuleList([
            nn.Conv2d(ch, ch, kernel_size=k, padding=k//2) for k in self.ksizes
        ])


    def forward(self, x: torch.Tensor):
        '''
        x: (B,N,c,h,w), [0,1]
        '''
        B, N, c, h, w = x.shape
        x0 = x.permute(0, 2, 3, 4, 1)
        x0 = self.linear(x0).squeeze(-1) # B,c,h,w

        x = x.reshape(B, N*c, h, w)
        x = self.conv(x) # (B,dim,hp,wp)
        B, dim, hp, wp = x.shape
        x1 = x.permute(0, 2, 3, 1).reshape(B, hp*wp, dim)
        for layer in self.attn_layers:
            x1 = layer(x1)
        x1 = x1.permute(0, 2, 1).reshape(B, dim, hp, wp)

        splits = torch.chunk(x, len(self.ksizes), dim=1)  # (B,dim/4,hp,wp) * 4
        x2 = [conv(s) for conv, s in zip(self.convs, splits)]
        x2 = torch.cat(x2, dim=1) # (B,dim,hp,wp)
        res = self.up_conv(x1+x2) + x0 # B,c,h,w
        # res = torch.clamp(res, 0, 1)
        return res


class Ennet(nn.Module):
    def __init__(self, opt):
        super(Ennet, self).__init__()
        self.N = opt['num_models']
        self.moe = MOE(self.N)
        self.mrffi = MRFFI(opt)
        print("total parameter num of Ennet: ", count_parameters(self), '-------------------------')

    def forward(self, x, xn):
        '''
        x.shape = B,C,H,W
        xn.shape = B,N,C,H,W
        '''
        moe_weight = self.moe(x) # B,N
        B, N, C, H, W = xn.shape
        xn = xn * moe_weight.view(B, self.N, 1, 1, 1)
        res = self.mrffi(xn)
        return moe_weight, res


if __name__ == '__main__':
    B, C, H, W = 4, 3, 512, 960
    N = 6 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = {'channels': 3, 'block_size': 16, 'kernal_sizes': [3,7,11,15], 'emb_dim': 256, 'n_layers': 2, 'num_models': 6}
    model = Ennet(opt)
    x = torch.randn(B, C, H, W)
    xn = torch.randn(B, N, C, H, W)
    w, res = model(x, xn)
    print(res.shape)