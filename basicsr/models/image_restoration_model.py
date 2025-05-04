import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import glob

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

from pdb import set_trace as stx
from skimage import img_as_ubyte

# 训练时开启混合增强
class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)
        # print(r_index.device, target.device)
        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


def norm(tensor: torch.Tensor):
    mean = tensor.mean(dim=-1).unsqueeze(-1).expand_as(tensor)
    std = tensor.std(dim=-1, unbiased=False).unsqueeze(-1).expand_as(tensor)
    return (tensor - mean) / (std+1e-10)


def calculate_psnr_tensor(hqs, gt):
    """
    hqs: (B,N,C,H,W)
    gt: (B,C,H,W)
    """
    B, N, C, H, W = hqs.shape
    assert gt.shape == (B, C, H, W)
    gt = gt[:, None, :, :, :].expand(B, N, C, H, W)
    mse = torch.mean((hqs - gt) ** 2, dim=(2, 3, 4))  # (B,N)
    psnr = 10 * torch.log10(1 / mse)
    return psnr  # (B,N)


def dict_add(dic, key, val):
    dic[key] = dic.get(key, 0) + val


class ImageCleanModel(BaseModel):
    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # 训练中混合增强
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get(
                'mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get(
                'use_identity', False)
            self.mixing_augmentation = Mixing_Augment(
                mixup_beta, use_identity, self.device)

        # define network
        self.net_g = define_network(deepcopy(opt)) # 按配置文件中的 network.type实例化net
        self.net_g = self.model_to_device(self.net_g)  #放到GPU上
        # self.print_network(self.net_g)
        
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        # 初始化训练设置
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train() # Set the module in training mode.
        train_opt = self.opt['train']

        # 初始化一个用于测试的网络实例 net_g_ema，并根据需要加载预训练的权重，然而现在并没有启用？
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = define_network(self).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        if train_opt.get('losses'):
            self.loss_funcs = []
            self.loss_names = train_opt['losses'].keys()
            for type in train_opt['losses']:
                func = getattr(loss_module, type)
                self.loss_funcs.append(func(**train_opt['losses'][type]).to(self.device))
        else:
            raise ValueError('pixel loss are None.')
        self.lambda_moe = train_opt.get('lambda_moe')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad: #需要被优化的参数，默认都是true
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam( 
                optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        # print(data['hqs'].device, self.device) # cuda:0 cuda
        # assert data['hqs'].device == self.device
        self.hqs = data['hqs'].to(self.device)
        self.lq = data['lq'].to(self.device)
        # self.metrics = data['metrics'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        # if self.mixing_flag:
        #     self.gt, self.hqs = self.mixing_augmentation(self.gt, self.hqs)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.hqs = data['hqs'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter, tb_logger=None): 
        self.optimizer_g.zero_grad()
        moe_w, pred = self.net_g(self.lq, self.hqs) # 得到增强结果
        self.output = pred # 用于validation
        loss_dict = OrderedDict()
        # pixel loss
        losses = [loss_func(pred, self.gt) for loss_func in self.loss_funcs]
        l_pix = sum(losses)
        self.metrics = calculate_psnr_tensor(self.hqs, self.gt)
        l_moe = F.kl_div(moe_w.log(), F.softmax(self.metrics, dim=-1))
        loss = l_pix + self.lambda_moe*l_moe
        loss.backward()
        if self.opt['train'].get('grad_clip', None):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.opt['train']['grad_clip'])
        self.optimizer_g.step()

        if not hasattr(self, 'log_dict'):
            self.log_dict = {}
        for name, _loss in zip(self.loss_names, losses):
            dict_add(self.log_dict, name, _loss)
        dict_add(self.log_dict, 'l_moe', l_moe)
        dict_add(self.log_dict, 'loss', loss)
        dict_add(self.log_dict, 'cnt', 1)
        if tb_logger:
            tb_logger.add_scalar(f'Train/loss_pix', l_pix, current_iter)
        # self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
 
    # 填充
    def pad_test(self, window_size): #使图像尺寸能整除window_size，方便卷积之类的操作？
        raise ValueError("window size not designeded")

    def nonpad_test(self, img=None):
        if img is None:
            img = self.hqs
        if hasattr(self, 'net_g_ema'):
            raise ValueError('net_g_ema need design')
        else: #here
            self.net_g.eval()
            with torch.no_grad():
                _, pred = self.net_g(self.lq, self.hqs)
            self.output = pred
            self.net_g.train() # 设置模型回到训练模式

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        # 准备模型，初始化，
        self.net_g.eval()
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        save_prob = self.opt['val'].get('save_img_prob', 1)
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else: #here
            test = self.nonpad_test

        # val过程
        cnt = 0
        self.metric_results['l_pix'] = 0 #总损失

        for idx, val_data in enumerate(dataloader): # 遍历验证数据集中的所有样本
            
            # 获取图像名称，确保只有一张
            img_name = val_data['img_name']
            assert len(img_name)==1
            img_name = img_name[0]
            
            # test
            self.feed_data(val_data)
            test()
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.hqs
            del self.output
            torch.cuda.empty_cache()
            
            # save img
            if save_img and idx<save_prob*len(dataloader):
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name, f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name, f'{img_name}_gt.png') #_{current_iter}
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                # imwrite(cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR), save_img_path)
                # imwrite(cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR), save_gt_img_path)
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)
            
            sr_img = sr_img.astype(np.float64)/255.
            gt_img = gt_img.astype(np.float64)/255.
            # calculate metrics
            if with_metrics:
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        ret = getattr(metric_module, metric_type)(sr_img, gt_img, **opt_) # use_image：指定用tensor2img后的结果
                        self.metric_results[name] += ret
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        ret = getattr(metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_) 
                        self.metric_results[name] += ret
            cnt += 1
            # self.metric_results['l_pix'] += self.cri_pix(visuals['result'], visuals['gt']) # here , validate时输出loss

        # 计算平均指标 
        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt 
            
            # 记录指标值到 TensorBoard，并保存当前的指标
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            current_metric = self.metric_results
        
        self.net_g.train()
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'Val/{metric}', value, current_iter)

    def get_current_visuals(self): #hqs,output,gt
        out_dict = OrderedDict()
        out_dict['hqs'] = self.hqs.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, **kwargs):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter, **kwargs)

    def save_best(self, best_metric, param_key='params'): # 保存best_psnr时的net_g模型
        psnr = best_metric['psnr']
        cur_iter = best_metric['iter']
        save_filename = f'best_psnr_{psnr:.2f}_{cur_iter}.pth'
        exp_root = self.opt['path']['experiments_root']
        save_path = os.path.join(
            self.opt['path']['experiments_root'], save_filename)

        if not os.path.exists(save_path):
            for r_file in glob.glob(f'{exp_root}/best_*'):
                os.remove(r_file)
            net = self.net_g

            net = net if isinstance(net, list) else [net]
            param_key = param_key if isinstance(
                param_key, list) else [param_key]
            assert len(net) == len(
                param_key), 'The lengths of net and param_key should be the same.'

            save_dict = {}
            for net_, param_key_ in zip(net, param_key):
                net_ = self.get_bare_model(net_)
                state_dict = net_.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    state_dict[key] = param.cpu()
                save_dict[param_key_] = state_dict

            torch.save(save_dict, save_path)


