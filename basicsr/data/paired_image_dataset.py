from torch.utils import data as data
from torchvision.transforms.functional import normalize
from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file,
                                    path_to_tensor)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP, tensor2img, imwrite

import random
import numpy as np
import torch
import cv2
from pdb import set_trace as stx
import os
import time
from glob import glob
from natsort import natsorted
import json
# from basicsr.data.generate_hq import Processes_Controller
# from basicsr.utils.img_util import DWTs, IWTs, padding_img
# from basicsr.models.archs.ennet_arch import ennet

class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.file_client = None 
        self.io_backend_opt = opt['io_backend'] # disk
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt: self.filename_tmpl = opt['filename_tmpl']
        else: self.filename_tmpl = '{}'

        self.paths = paired_paths_from_folder( 
            [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            self.filename_tmpl) # 一个包含dict的列表，每个dict都有两个键：<input_key>_path 和 <gt_key>_path

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        self.ensemble_models = opt['ensemble_models']
        self.dataset_name = opt['dataset_name']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.metrics = {}
        # def calculate(metrics):
        #     return metrics['psnr']

        # for model in self.ensemble_models:
        #     path = os.path.join('ensemble_models', model, 'ensemble_output', self.dataset_name, 'metrics.json')
        #     with open(path, 'r') as f:
        #         data = json.load(f)
        #     for key, val in data.items():
        #         score = calculate(val)
        #         if key not in self.metrics:
        #             self.metrics[key] = []
        #         self.metrics[key].append(score)
        # for key, val in self.metrics.items():
        #     self.metrics[key] = torch.tensor(val, device=self.device)

    def get_hqs(self, img_name):
        ensemble_models = self.ensemble_models
        dataset_name = self.dataset_name
        res = []
        for model in ensemble_models:
            path = os.path.join('ensemble_models', model, 'ensemble_output', dataset_name, img_name)
            img = path_to_tensor(path)
            img = img.to(self.device).unsqueeze(0)
            assert len(res)==0 or res[-1].shape==img.shape
            res.append(img)
        res = torch.cat(res, dim=0)
        return res

    def __getitem__(self, index):
        index = index % len(self.paths)    
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_name = os.path.split(gt_path)[-1]
        img_gt = path_to_tensor(gt_path)
        lq_path = self.paths[index]['lq_path']
        img_lq = path_to_tensor(lq_path)

        img_hqs = self.get_hqs(img_name)
        img_hqs = img_hqs.to(self.device)
        # stx()
        ret = {'lq': img_lq, 'hqs': img_hqs, 'gt': img_gt, 'img_name': img_name}
        # if img_name in self.metrics:
        #     ret['metrics'] = self.metrics[img_name]
            # print(ret['metrics'])
        # lq,gt:ndarray. hqs:tensor
        return ret
    
    def __len__(self):
        return len(self.paths)

    def close_env(self):
        self.model_controller.close_env()

