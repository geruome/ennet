import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass
from pdb import set_trace as stx
# import cv2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MOE(nn.Module):
    def __init__(self, N=6, C=3):
        super(MOE, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, N)
        print("total parameter num of MOE: ", count_parameters(self), '-------------------------') # 

    def forward(self, x):
        x = self.conv_layers(x) 
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x) 
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


class Ennet(nn.Module):
    def __init__(self, opt):
        super(Ennet, self).__init__()
        self.N = opt['num_models']
        self.in_channels = opt.get('in_channels', 3)
        self.emb_dim = opt['emb_dim']
        self.P = opt['block_size']
        self.stride = opt['stride']
        hidden_dim = self.N * self.emb_dim
        attn_config = AttentionConfig(**dict(n_embd=hidden_dim))
        self.conv = nn.Conv2d(self.in_channels, self.emb_dim, kernel_size=self.P, stride=self.stride)
        self.up_conv = nn.ConvTranspose2d(self.emb_dim, self.in_channels, kernel_size=self.P, stride=self.stride)
        self.attn_layers = nn.ModuleList( Block(attn_config) for _ in range(opt['n_layers']))
        self.linear = nn.Linear(self.N, 1, bias=False)
        self.moe = MOE(self.N)
        # self.weight = nn.Parameter(torch.full((self.N,), 1/self.N))
        print("total parameter num of Ennet: ", count_parameters(self), '-------------------------')

    def forward(self, x, xn):
        '''
        x.shape = B,C,H,W
        xn.shape = B,N,C,H,W #hqs
        '''
        moe_weight = self.moe(x) # B,N
        B, N, C, H, W = xn.shape
        xn = xn * moe_weight.view(B, self.N, 1, 1, 1)
        # assert H%self.stride==0 and W%self.stride==0
        y = xn.reshape(B*N, C, H, W)
        y = self.conv(y) # (B*N, dim, hp, wp)
        BN, dim, hp, wp = y.shape
        y = y.reshape(B, N, dim, hp, wp)
        y = y.permute(0, 3, 4, 1, 2)
        # (B, N, dim*hp*wp)
        y = y.reshape(B, hp*wp, N*dim) # (B, hp*wp, N*dim)

        for layer in self.attn_layers: # block_atten
            y = layer(y)
        y = y.reshape(B, hp, wp, N, dim)
        y = y.permute(0, 3, 4, 1, 2) # (B, N, dim, hp, wp)
        y = y.reshape(BN, dim, hp, wp)
        y = self.up_conv(y)
        y = y.reshape(B, N, C, H, W)
        
        xn = xn + y # (B, N, C, H, W)
        xn = xn.permute(0, 2, 3, 4, 1)
        xn = self.linear(xn).squeeze(-1)
        return moe_weight, xn


if __name__ == '__main__':
    B, C, H, W = 4, 3, 384, 384
    N = 6 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MOE(N, C)
    model.to(device)
    x = torch.randn(B, C, H, W).to(device)
    output = model(x)
    print(output.shape)
    print(output.sum(dim=1))