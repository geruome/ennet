import argparse
import datetime
import logging
import math
import os
import random
import time
import torch
from os import path as osp

import sys
sys.path.append(os.getcwd())
sys.path.append('..')

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename, set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.misc import mkdir_and_rename2
from basicsr.utils.options import parse
import numpy as np 
from pdb import set_trace as stx

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='test/test.yml', help='Path to option YAML file.')
    parser.add_argument('--gpu_id', type=str, default="0", help='GPU devices.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher') #默认不进行分布式训练
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)  # parse用的是basicsr的工具，填加了一些参数(root,experiment)
    # gpu_list = ','.join(str(x) for x in args.gpu_id)
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    # if opt.get('devices', ""):
    #     os.environ["CUDA_VISIBLE_DEVICES"] = opt['devices']
    # print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    
    # distributed settings 
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info() #0,1

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])
    return opt

def init_loggers(opt):
    # log记录器
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    
    # metric记录器
    log_file = osp.join(opt['path']['log'],
                        f"metric.csv")
    logger_metric = get_root_logger(logger_name='metric',
                                    log_level=logging.INFO, log_file=log_file)
    metric_str = f'iter ({get_time_str()})'
    for k, v in opt['val']['metrics'].items():
        metric_str += f',{k}'
    logger_metric.info(metric_str) 
    
    # 打印opt
    # logger.info(dict2str(opt)) 
    return logger

def create_train_val_dataloader(opt, logger):
    
    train_loader, train_sampler, val_loader, total_epochs, total_iters = None, None, None, None, None
    ensemble_models = opt['ensemble_models']
    dataset_name = opt['dataset_name']

    for phase, dataset_opt in opt['datasets'].items():
        
        dataset_opt['ensemble_models'] = ensemble_models
        dataset_opt['dataset_name'] = dataset_name 
        
        if phase == 'train':
            continue

        elif phase == 'val':
            dataset_opt['use_cache'] = False
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                # num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters

def main():
    opt = parse_options(is_train=True)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name']) #状态路径
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []
    resume_state = None
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    if resume_state is None: 
        make_exp_dirs(opt) # 建文件夹： ennet_tower / {models, training_states, visualization, _arch.py}
    
    logger = init_loggers(opt)
    
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    model = create_model(opt)    

    current_iter = 0
    start_epoch = 0
    # weight_path = 'experiments/05072338_LOLv2s_nomean/models/net_g_90000.pth'
    weight_path = 'experiments/05081551_LOLv1/models/net_g_40000.pth'
    checkpoint = torch.load(weight_path)
    model.net_g.load_state_dict(checkpoint['params'])
    best_metric = {'iter': 0}
    for k, v in opt['val']['metrics'].items():
        best_metric[k] = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, None)

    # training
    logger.info(f'Start training from iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()
    iters = opt['datasets']['train'].get('iters')
    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')

    epoch = start_epoch

    rgb2bgr = opt['val'].get('rgb2bgr', True)
    use_image = opt['val'].get('use_image', True)
    
    current_metric = model.validation(val_loader, current_iter, None,
                                        opt['val']['save_img'], rgb2bgr, use_image)
    
    logger_metric = get_root_logger(logger_name='metric')
    metric_str = f'{current_iter}'
    for metric, value in current_metric.items():
        metric_str += f', {value:.4f}'
    logger_metric.info(metric_str)
    exit(0)

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, None, opt['val']['save_img'])


if __name__ == '__main__':
    main()