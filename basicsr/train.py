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
    parser.add_argument('--opt', type=str, default='Options/LOLv1.yml', help='Path to option YAML file.')
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
    log_file = osp.join(opt['path']['log'], f"train.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    
    # metric记录器
    log_file = osp.join(opt['path']['log'], f"metric.csv")
    logger_metric = get_root_logger(logger_name='metric',
                                    log_level=logging.INFO, log_file=log_file)
    metric_str = f'iter ({get_time_str()})'
    for k, v in opt['val']['metrics'].items():
        metric_str += f',{k}'
    logger_metric.info(metric_str) 
    
    # 打印opt
    # logger.info(dict2str(opt)) 
    tb_logger = None
    if opt['logger'].get('use_tb_logger'):
        tb_logger = init_tb_logger(log_dir=opt['path']['log'])
    return logger, tb_logger

def create_train_val_dataloader(opt, logger):
    
    train_loader, train_sampler, val_loader, total_epochs, total_iters = None, None, None, None, None
    ensemble_models = opt['ensemble_models']
    dataset_name = opt['dataset_name']

    for phase, dataset_opt in opt['datasets'].items():
        
        dataset_opt['ensemble_models'] = ensemble_models
        dataset_opt['dataset_name'] = dataset_name 
        
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)   # pair_image_dataset类，里面包含文件列表，进程和moe?            
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)  #采样器，用于分布式训练?
            
            train_loader = create_dataloader( 
                train_set,
                dataset_opt,
                # num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])
            
            # 计算并记录迭代信息
            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))  #一个epoch遍历一次数据
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch)) #一个iteration就是一次 inference + backward，总的iteration是不变的
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
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

    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)
    # 加速
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name']) #状态路径
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []
    resume_state = None
    # if len(states) > 0:  #如果路径已存在 (已经训练到)
    #     max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
    #     resume_state = os.path.join(state_folder_path, max_state_file)
    #     opt['path']['resume_state'] = resume_state
    # load resume states if necessary，resume_state是重新训练的时候接上的吗？
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None: 
        make_exp_dirs(opt) # 建文件夹： ennet_tower / {models, training_states, visualization, _arch.py}
    
    # initialize loggers
    logger, tb_logger = init_loggers(opt)
    
    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # if resume_state:  # resume training
    #     check_resume(opt, resume_state['iter'])
    #     # print(resume_state)
    #     model = create_model(opt)
    #     model.resume_training(resume_state)  # handle optimizers and schedulers
    #     logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
    #                 f"iter: {resume_state['iter']}.")
    #     start_epoch = resume_state['epoch']
    #     current_iter = resume_state['iter']
    #     best_metric = resume_state['best_metric']
    #     # best_psnr = best_metric['psnr']
    #     # best_iter = best_metric['iter']
    #     # logger.info(f'best psnr: {best_psnr} from iteration {best_iter}')

    model = create_model(opt)    
    if False: # resume training
        current_iter = 100000
        for _ in range(current_iter):
            model.update_learning_rate(current_iter) # 学习率调到位
        start_epoch = 445
        weight_path = 'experiments/04101948_LOLv2s/models/net_g_100000.pth'
        checkpoint = torch.load(weight_path)
        model.net_g.load_state_dict(checkpoint['params'])
        best_metric = {'iter': 0}
        for k, v in opt['val']['metrics'].items():
            best_metric[k] = 0
    else:
        start_epoch = 0
        current_iter = 0
        best_metric = {'iter': 0}
        for k, v in opt['val']['metrics'].items():
            best_metric[k] = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, None)
    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')    
    if prefetch_mode is None or prefetch_mode == 'cpu':
        print('cpu')
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda': # here
        print('gpu')
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()
    iters = opt['datasets']['train'].get('iters')
    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    # gt_size = opt['datasets']['train'].get('gt_size')
    # mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    epoch = start_epoch

    # loss_val = 0; loss_cnt = 0
    while current_iter <= total_iters: #一个epoch
        # print('new epoch, current_iter {} , total_iters {}'.format(current_iter, total_iters))
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next() #lq,hqs(小波变换过)
        sum_data_time = 0; sum_iter_time = 0
        while train_data is not None: #一个batch
            sum_data_time += time.time() - data_time # 数据加载时长
            current_iter += 1
            if current_iter > total_iters: break
            
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # ------Progressive learning ---------------------
            j = ((current_iter > groups) != True).nonzero()[0]  # 根据当前的iter次数判断在哪个阶段, [0,1,2,...]
            pre_bs_j = bs_j if 'bs_j' in locals() else None
            bs_j = j[0] if len(j) else len(groups)-1

            # mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]

            if bs_j != pre_bs_j:
                logger.info('\n Updating Batch_Size to {} at iter {}\n'.format(mini_batch_size * torch.cuda.device_count(), current_iter))

            # lq = train_data['lq']
            # hqs = train_data['hqs']
            # gt = train_data['gt']
            if mini_batch_size < batch_size: 
                indices = random.sample(range(0, batch_size), k=mini_batch_size)
                train_data['lq'] = train_data['lq'][indices]
                train_data['gt'] = train_data['gt'][indices]
                train_data['hqs'] = train_data['hqs'][indices]
                # train_data['metrics'] = train_data['metrics'][indices]
            # if mini_gt_size < gt_size:
            #     x0 = int((gt_size - mini_gt_size) * random.random())
            #     y0 = int((gt_size - mini_gt_size) * random.random())
            #     x1 = x0 + mini_gt_size
            #     y1 = y0 + mini_gt_size
            #     lq = lq[:, :, x0:x1, y0:y1]
            #     gt = gt[:, :, x0:x1, y0:y1]
            #     hqs = hqs[:, :, :, x0:x1, y0:y1]
            # model.feed_train_data({'lq': lq, 'hqs': hqs, 'gt': gt,  })
            model.feed_train_data(train_data)
            model.optimize_parameters(current_iter, tb_logger) #

            sum_iter_time += time.time() - iter_time

            if current_iter % opt['logger']['print_freq'] == 0: 
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                # print(f"{model.get_current_learning_rate()[0]:.15f}")
                log_vars.update({'time': sum_iter_time/opt['logger']['print_freq'], 'data_time': sum_data_time/opt['logger']['print_freq']})
                sum_iter_time = sum_data_time = 0
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if (opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0)):  # or current_iter==1:
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                
                current_metric = model.validation(val_loader, current_iter, tb_logger,
                                                  opt['val']['save_img'], rgb2bgr, use_image)
                
                # log cur metric to csv file
                logger_metric = get_root_logger(logger_name='metric')
                metric_str = f'{current_iter}'
                for metric, value in current_metric.items():
                    metric_str += f', {value:.4f}'
                logger_metric.info(metric_str)
                
                # log best metric
                if best_metric['psnr'] < current_metric['psnr']:
                    best_metric = current_metric
                    # save best model
                    best_metric['iter'] = current_iter
                    model.save_best(best_metric)

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()

        # end of iter
        epoch += 1
    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, None, opt['val']['save_img'])


if __name__ == '__main__':
    main()
