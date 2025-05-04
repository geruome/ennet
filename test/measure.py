import os
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import utils
from pdb import set_trace as stx

res_dir = '/root/autodl-tmp/ennet/ensemble_models/retinexformer/ensemble_output/LOLv2s'
target_dir = '/root/autodl-tmp/ennet/datasets/LOLv2/Synthetic/Test/Normal'

res_names = os.listdir(res_dir)
target_names = sorted(os.listdir(target_dir))
res_names = sorted(list(set(res_names) & set(target_names)))
assert res_names == target_names
res_paths = sorted([os.path.join(res_dir, f) for f in res_names])
target_paths = sorted([os.path.join(target_dir, f) for f in target_names])

psnr = []
ssim = []
lpips = []
for res_path, target_path in tqdm(zip(res_paths, target_paths), total=len(res_paths)):
    res = np.float32(utils.load_img(res_path)) / 255.
    target = np.float32(utils.load_img(target_path)) / 255.
    psnr.append(utils.calculate_psnr(target, res))
    ssim.append(utils.calculate_ssim(target, res))
    lpips.append(utils.calculate_lpips(target, res))

psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))
lpips = np.mean(np.array(lpips))
print(f"psnr: {psnr:.4f}, ssim: {ssim:.4f}, lpips: {lpips:.4f}")