import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import warnings
from dataclasses import dataclass
from pdb import set_trace as stx
# import cv2
       

class moe(nn.Module):
    def __init__(self, N):
        super(moe, self).__init__()
        self.N = N
        dropout_rate = 0.5

        #提取特征
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        #embedding
        self.layer1 = nn.Linear(16*3*3, self.N)#C*3*3
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(16*3*3, self.N)#C*3*3
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(16*3*3, self.N)#C*3*3
        self.dropout3 = nn.Dropout(dropout_rate)


    def forward(self, x):
        x = x.unsqueeze(0)
        B, N, C, H, W = x.shape #1,1,C,H,W
        x = x.view(B, N*C, H, W) # B  * N*C * H * W

        #卷积提取特征
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        #固定输出大小的pooling
        B, _, H, W = x.shape
        kernel_size = (math.ceil(H / 3), math.ceil(W / 3))
        stride = (math.ceil(H / 3), math.ceil(W / 3))
        padding = (math.floor((kernel_size[0] * 3 - H + 1) / 2), math.floor((kernel_size[1] * 3 - W + 1) / 2))
        x = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)#  B  * N*C * 3 * 3
        x = x.view(B, N * 16 * 3 * 3)#二维

        #embedding
        q = torch.relu(self.layer1(x))
        q = self.dropout1(q) # self.N

        k = torch.relu(self.layer2(x))
        k = self.dropout2(k) # self.N

        v = torch.relu(self.layer3(x))
        v = self.dropout3(v) # self.N

        router = torch.matmul(q.T, k)
        weights = torch.softmax(torch.matmul(router, v.T).T, dim=1) #self.N
        return weights


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


class ennet(nn.Module):
    def __init__(self, opt):
        super(ennet, self).__init__()
        self.N = opt['num_models']
        self.in_channels = opt.get('in_channels', 3)
        self.emb_dim = opt['emb_dim']
        self.P = opt['block_size']
        hidden_dim = self.N * self.emb_dim
        # self.moe = moe(self.N)
        attn_config = AttentionConfig(**dict(n_embd=hidden_dim))
        self.conv = nn.Conv2d(self.in_channels, self.emb_dim, kernel_size=self.P, stride=self.P)
        self.attn_layers = nn.ModuleList( Block(attn_config) for _ in range(opt['n_layers']))
        self.up_conv = nn.ConvTranspose2d(self.emb_dim, self.in_channels, kernel_size=self.P, stride=self.P)
        self.linear = nn.Linear(self.N, 1)
        # print("total parameter num: ", count_parameters(self), '-------------------------')

    def forward(self, x, xn):
        '''
        x.shape = B,C,H,W
        xn.shape = B,N,C,H,W
        '''
        y = xn
        B, N, C, H, W = y.shape
        # print(y.shape, '000000')
        assert H%self.P==0 and W%self.P==0
        hp = H // self.P
        wp = W // self.P

        y = y.reshape(B*N, C, H, W)
        y = self.conv(y) # (B*N, dim, hp, wp)
        BN, dim, hp, wp = y.shape
        y = y.reshape(B, N, dim, hp, wp)
        # print(y.shape, '1111111')
        y = y.permute(0, 3, 4, 1, 2)
        y = y.reshape(B, hp*wp, N*dim) # (B, hp*wp, N*dim)
        # print(y.shape, '2222222')
        for layer in self.attn_layers:
            y = layer(y)
        y = y.reshape(B, hp, wp, N, dim)
        y = y.permute(0, 3, 4, 1, 2) # (B, N, dim, hp, wp)
        y = y.reshape(BN, dim, hp, wp)
        y = self.up_conv(y)
        y = y.reshape(B, N, C, H, W)
        # print(y.shape, '33333333')
        xn = xn + y
        xn = xn.permute(0, 2, 3, 4, 1)
        xn = self.linear(xn).squeeze(-1)
        # print(xn.shape, '44444444')
        return xn
