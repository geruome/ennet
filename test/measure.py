import os
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import utils
from pdb import set_trace as stx

input_dir = '/root/autodl-tmp/ennet/ensemble_models/pydiff/ensemble_output/LOLv1'
target_dir = '/root/autodl-tmp/ennet/datasets/LOLv1/Test/target'
input_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
target_paths = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)])

for input_path, target_path in zip(input_paths, target_paths):
    img_name = input_path.split('/')[-1]
    tmp = target_path.split('/')[-1]
    if img_name != tmp:
        raise ValueError(f"File name mismatch: {input_path} vs {target_path}")

psnr = []
ssim = []
lpips = []
for input_path, target_path in tqdm(zip(input_paths, target_paths), total=len(input_paths)):
    input = np.float32(utils.load_img(input_path)) / 255.
    target = np.float32(utils.load_img(target_path)) / 255.
    stx()
    psnr.append(utils.calculate_psnr(target, input))
    ssim.append(utils.calculate_ssim(target, input))
    lpips.append(utils.calculate_lpips(target, input))

psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))
lpips = np.mean(np.array(lpips))
print(f"psnr: {psnr:.4f}, ssim: {ssim:.4f}, lpips: {lpips:.4f}")