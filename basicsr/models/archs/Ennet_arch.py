import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass
from pdb import set_trace as stx
import pywt


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
        print(f"total parameter num of MOE: {count_parameters(self)}, -------------------------", flush=True)

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


def lg2(factor):
    num = int(math.log2(factor))
    assert 2**num == factor
    return num


class DownSample(nn.Module):
    def __init__(self, channels, factor):
        super(DownSample, self).__init__()
        num = lg2(factor)
        assert len(channels) == num+1
        layers = []

        for i in range(num):
            next_channel = channels[i+1]
            layers.append(nn.Conv2d(channels[i], next_channel, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(next_channel))
            layers.append(nn.ReLU(inplace=True))

        self.patch_embed = nn.Sequential(*layers)

    def forward(self, x):
        return self.patch_embed(x)


class BlurPool(nn.Module):
    def __init__(self, channels):
        super(BlurPool, self).__init__()
        kernel = torch.tensor([1., 2., 1.])
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()  # 3x3 Gaussian blur kernel
        self.register_buffer('kernel', kernel[None, None, :, :].repeat(channels, 1, 1, 1))
        self.channels = channels
        self.padding = 1

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=self.padding, groups=self.channels)


class UpSample(nn.Module):
    # pixelshuffle upsample
    def __init__(self, channels, factor):
        super(UpSample, self).__init__()
        num = lg2(factor)
        assert len(channels) == num+1

        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(channels[i], channels[i], kernel_size=3, padding=1))
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1))
            layers.append(BlurPool(channels[i+1]))
            layers.append(nn.ReLU(inplace=True))
        self.upsampler = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.upsampler(x)


class Denoiser(nn.Module):
    def __init__(self, opt):
        super(Denoiser, self).__init__()
        # opt['channels'] = 3
        channels = opt['channels']
        features = 16
        num_layers = 4
        layers = [nn.Sequential(nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))]
        for i in range(num_layers):
            layers.append(nn.Sequential(nn.Conv2d(features, features, kernel_size=3, padding=1), nn.ReLU(inplace=True)))
        layers.append(nn.Sequential(nn.Conv2d(features, 3, 1, 1), nn.ReLU(inplace=True))) # 1*1
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # [B,C,H,W]
        res = self.layers(x)
        return x + res


class BlockMixer(nn.Module):
    def __init__(self, opt):
        super(BlockMixer, self).__init__()
        self.N = opt['num_models']

        self.out_channels = opt['channels']
        self.emb_dim = opt['emb_dim']
        self.conv_dim = opt['conv_dim']
        self.dim = self.emb_dim + self.conv_dim
        self.P = opt['block_size']
        in_channels = self.out_channels*self.N

        self.linear = nn.Linear(self.N, 1, bias=False)

        self.down_channels = [in_channels] + opt['down_channels'] + [self.dim]
        self.downsample = DownSample(channels=self.down_channels, factor=self.P)
        # self.downsample = nn.Conv2d(in_channels, self.dim, kernel_size=self.P, stride=self.P, )
        
        attn_config = AttentionConfig(**dict(n_embd=self.emb_dim))
        self.attn_layers = nn.ModuleList( Block(attn_config) for _ in range(opt['n_layers']))

        self.ksizes = opt['kernal_sizes']
        assert all(i%2==1 for i in self.ksizes)
        ch = self.conv_dim
        assert ch%len(self.ksizes)==0
        ch = ch//len(self.ksizes)
        self.convs = nn.ModuleList([
            nn.Conv2d(ch, ch, kernel_size=k, padding=k//2) for k in self.ksizes
        ])

        self.up_channels = [self.dim] + opt['up_channels'] + [self.out_channels]
        self.upsample = UpSample(channels=self.up_channels, factor=self.P)
        # self.upsample = nn.ConvTranspose2d(self.dim, self.out_channels, kernel_size=self.P, stride=self.P, )
        self.denoiser = Denoiser(opt)

    def forward(self, x: torch.Tensor):
        '''
        x: (B,N,c,h,w), [0,1]
        '''
        B, N, c, h, w = x.shape
        assert h%self.P==0 and w%self.P==0
        x0 = x.permute(0, 2, 3, 4, 1)
        x0 = self.linear(x0).squeeze(-1) # B,c,h,w

        x = x.reshape(B, N*c, h, w)
        x = self.downsample(x) # (B,dim,hp,wp)
        x_atten, x_conv = torch.split(x, [self.emb_dim, self.conv_dim], dim=1) # (B,emb_dim,h,w), (B,conv_dim,h,w) 
        _, emb_dim, hp, wp = x_atten.shape
        x1 = x_atten.permute(0, 2, 3, 1).reshape(B, hp*wp, emb_dim)
        for layer in self.attn_layers:
            x1 = layer(x1)
        x1 = x1.permute(0, 2, 1).reshape(B, emb_dim, hp, wp)

        splits = torch.chunk(x_conv, len(self.ksizes), dim=1)  # (B,conv_dim/4,hp,wp) * 4
        x2 = [conv(s) for conv, s in zip(self.convs, splits)]
        x2 = torch.cat(x2, dim=1) # (B,conv_dim,hp,wp)
        res = torch.cat([x1,x2], dim=1)
        # stx()
        res = self.upsample(res) + x0 # B,c,h,w
        res = self.denoiser(res)
        return res
    

def create_wavelet_filter(wave='haar', in_size=3, out_size=3, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    x = x[:, :, 0, :, :]
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class Ennet(nn.Module):
    def __init__(self, opt):
        super(Ennet, self).__init__()
        self.N = opt['num_models']
        self.moe = MOE(self.N)
        self.blockmixers = nn.ModuleList([BlockMixer(opt) for _ in range(2)])
        self.dec_filter, self.rec_filter = create_wavelet_filter(in_size=opt['channels'], out_size=opt['channels'])
        self.dec_filter = nn.Parameter(self.dec_filter, requires_grad=False)
        self.rec_filter = nn.Parameter(self.rec_filter, requires_grad=False)
        print(f"total parameter num of Ennet: {count_parameters(self)}-------------------------", flush=True)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def multi_layer_enhance(self, x, xn, dep):
        # moe_w = self.moe(x)
        # xn = xn * moe_w[..., None, None, None]
        res = self.blockmixers[dep-1](xn)
        if dep==1:
            return res
        x = wavelet_transform(x, self.dec_filter)
        B, N, C, H, W = xn.shape
        xn = xn.flatten(0, 1)
        xn = wavelet_transform(xn, self.dec_filter)
        xn = xn.reshape(B, N, *xn.shape[1:])
        nres = self.multi_layer_enhance(x, xn, dep-1)
        nres = F.interpolate(nres, scale_factor=2, mode='bilinear', align_corners=False)
        return res + nres

    def forward(self, x, xn):
        '''
        x.shape = B,C,H,W
        xn.shape = B,N,C,H,W
        '''
        moe_w0 = self.moe(x) # B,N
        res = self.multi_layer_enhance(x, xn, dep=2)
        return moe_w0, res


if __name__ == '__main__':
    

    B, C, H, W = 4, 3, 384, 384
    N = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = {'num_models': N, 'channels': 3, 'block_size': 8, 
           'conv_dim': 64, 'kernal_sizes': [3,5,7,9], 'down_channels': [48, 96], # 18,48,96,192
           'emb_dim': 128, 'n_layers': 2, 'up_channels': [64, 16]} # 192, 64, 16, 3
    # opt = {'channels': 3, 'block_size': 16, 'kernal_sizes': [3,7,11,15], 'dim': 512, 'n_layers': 2, 'num_models': N}

    model = Ennet(opt)
    x = torch.randn(B, C, H, W)
    xn = torch.randn(B, N, C, H, W)
    w, res = model(x, xn)
    print(res.shape)