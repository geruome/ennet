from ast import arg
import numpy as np
import os

import argparse
from tqdm import tqdm
import cv2
import time
import sys
sys.path.append(os.getcwd())
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Enhancement using Retinexformer')
parser.add_argument('--input_dir', default='data/tower/test/low', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='results', type=str, help='Directory for results')
parser.add_argument('--opt', type=str, default='Options/ennet_tower.yml', help='Path to option YAML file.')
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')

args = parser.parse_args()
####### Load yaml #######
yaml_file = args.opt
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from basicsr.utils.options import parse

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
opt = parse(args.opt, is_train=False)
opt['dist'] = False
P = opt['network_g']['block_size']

import torch.nn as nn
import torch
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import utils
from skimage import img_as_ubyte, metrics
# from basicsr.metrics.psnr_ssim import calculate_lpips

# from basicsr.data import create_dataloader, create_dataset
from basicsr.utils.img_util import DWTs, IWTs, light_effects_seg, light_seg_recover, padding_img, ndarray_to_tensor, tensor_to_ndarray, gamma_correction, denoise
# from basicsr.utils.img_util import Histogram
from basicsr.utils.new_hist import histogram_equalization_luminance as Histogram # 
# from basicsr.models.archs.donoise_arch import denoise_net
# from basicsr.models.archs.RIDnet_arch import RIDNet
from basicsr.utils import tensor2img, imwrite
import gradio as gr
from os import path as osp

#########################
from basicsr.models import create_model
model = create_model(opt)

weight_path = opt['weight_path']
checkpoint = torch.load(weight_path)
model.net_g.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ", weight_path)
model.net_g.eval()

# denoise_model = RIDNet(3, 64, 3)
# ckpt = torch.load('pretrained_weights/RIDNet.pth')
# denoise_model.load_state_dict(ckpt)
# denoise_model.to(device)
# denoise_model.eval()
print('#########################')

models = opt['ensemble_models']

lq_dir = args.input_dir
output_dir = args.result_dir

from basicsr.data.generate_hq import Processes_Controller, path_to_tensor
model_controller = Processes_Controller(models)

tmp_path = os.path.join(os.getcwd(), output_dir, '__tmp.png')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def crop_img(tensor_img, coff):
    # print(tensor_img.shape, coff)
    lh, rh, lw, rw = coff
    C, H, W = tensor_img.shape
    tensor_img = tensor_img[:, lh:H-rh, lw:W-rw]
    return tensor_img


def calc(img, io_numpy=True): # H,W,C
    with torch.inference_mode():
        lightseg_time = 0; moe_time = 0; dwt_time = 0; submodel_time = 0; hist_time = 0; gamma_time = 0
        start_time = time.time()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        # print('origin shape: ', img.shape[:-1])
        if io_numpy:
            img = ndarray_to_tensor(img)
        img_lq = img.to(device)
        img_lq, coffe = padding_img(img_lq, P=P, return_edges=True)

        # torch.cuda.synchronize(); lightseg_start = time.time()
        # torch.cuda.synchronize(); lightseg_time += time.time() - lightseg_start

        # torch.cuda.synchronize(); dwt_start = time.time()
        img_lq, lq_xhs = DWTs(img_lq)
        img_lq_tmp = img_lq
        img_lq, mask = light_effects_seg(img_lq)
        # torch.cuda.synchronize(); dwt_time += time.time() - dwt_start
        # print('compressed shape: ', img_lq.shape[1:])

        imwrite(tensor2img(img_lq), tmp_path)

        # torch.cuda.synchronize(); submodel_start = time.time()
        img_hqs = model_controller.get_hqs(tmp_path)
        # torch.cuda.synchronize(); submodel_time += time.time() - submodel_start
        os.remove(tmp_path)
        if not torch.is_tensor(img_hqs):
            raise gr.Error("subprocesses went wrong")
            # return cv2.imread('data/error.jpg')
        
        img_hqs = img_hqs.unsqueeze(0)
        img_hqs = img_hqs.to(device)
        img_lq = img_lq.unsqueeze(0)
        # torch.cuda.synchronize(); moe_start = time.time()
        restored = model.calc(img_lq, img_hqs) # B,C,H,W
        restored = restored.squeeze(0)
        # torch.cuda.synchronize(); moe_time += time.time() - moe_start
        # torch.cuda.synchronize(); lightseg_start = time.time()
        restored = light_seg_recover(restored, img_lq_tmp, mask)
        # torch.cuda.synchronize(); lightseg_time += time.time() - lightseg_start
        # torch.cuda.synchronize(); iwt_start = time.time()
        restored = IWTs(restored, lq_xhs) #
        # torch.cuda.synchronize(); dwt_time += time.time() - iwt_start
        restored = crop_img(restored, coffe)
        # torch.cuda.synchronize(); gamma_start = time.time()
        restored = gamma_correction(restored)
        # torch.cuda.synchronize(); gamma_time = time.time() - gamma_start
        # torch.cuda.synchronize(); hist_start = time.time()
        restored = Histogram(restored)
        # torch.cuda.synchronize(); hist_time += time.time() - hist_start
        restored = denoise(restored)

        # imwrite(tensor2img(restored), output_path)
        if io_numpy:
            restored = tensor_to_ndarray(restored)

        all_time = time.time() - start_time
        # print(type(restored), restored.shape)
        # print(f"all time: {all_time}")
        # print(f"io + data processing time: \
        #       {all_time-dwt_time-gamma_time-lightseg_time-hist_time-submodel_time-moe_time}")
        # print(f"dwt and iwt time: {dwt_time:.4f}")
        # print(f"gamma time: {gamma_time:.4f}")
        # print(f"light_seg time: {lightseg_time:.4f}")
        # print(f"histogram time: {hist_time}")
        # print(f"submodel time: {submodel_time}")
        # print(f"moe model time: {moe_time}")
        return restored


demo = gr.Interface(
    calc,
    inputs='image',
    outputs='image'
)
demo.launch()