import os
from os import path as osp
import numpy as np 
from tqdm import tqdm
import random
from pdb import set_trace as stx
import sys
sys.path.append(os.getcwd())
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table

import utils
from basicsr.models import create_model
from basicsr.utils.options import parse


def parse_options():
    yml_path = 'test/test.yml'
    opt = parse(yml_path, is_train=False)
    return opt

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    opt = parse_options()
    set_random_seed(opt['manual_seed'])
    model = create_model(opt)
    dataset_name = 'LOLv1'
    if dataset_name == 'LOLv1':
        lq_dir = '/root/autodl-tmp/ennet/datasets/LOLv1/Test/input'
        target_dir = '/root/autodl-tmp/ennet/datasets/LOLv1/Test/target'
    elif dataset_name == 'LOLv2s':
        lq_dir = '/root/autodl-tmp/ennet/datasets/LOLv2/Synthetic/Test/Low'
        target_dir = '/root/autodl-tmp/ennet/datasets/LOLv2/Synthetic/Test/Normal'
    else:
        lq_dir = f'/root/autodl-tmp/ennet/datasets/{dataset_name}/test/LQ'
        target_dir = f'/root/autodl-tmp/ennet/datasets/{dataset_name}/test/GT'
    ensemble_models = opt['ensemble_models']
    
    # weight_path = 'experiments/05072338_LOLv2s/models/net_g_90000.pth'
    weight_path = 'experiments/05081551_LOLv1/models/net_g_20000.pth'
    checkpoint = torch.load(weight_path)
    model.net_g.load_state_dict(checkpoint['params'])

    def get_hqs(img_name):
        res = []
        for model in ensemble_models:
            path = os.path.join('ensemble_models', model, 'ensemble_output', dataset_name, img_name)
            img = utils.path_to_tensor(path)
            img = img.unsqueeze(0)
            assert len(res)==0 or res[-1].shape==img.shape
            res.append(img)
        res = torch.cat(res, dim=0)
        return res

    lq_names = os.listdir(lq_dir)
    target_names = sorted(os.listdir(target_dir))
    lq_names = sorted(list(set(lq_names) & set(target_names)))
    assert lq_names == target_names
    lq_paths = sorted([os.path.join(lq_dir, f) for f in lq_names])
    target_paths = sorted([os.path.join(target_dir, f) for f in target_names])

    psnr = []
    ssim = []
    lpips = []
    save_dir = f'visualization/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)
    for lq_path, target_path in tqdm(zip(lq_paths, target_paths), total=len(lq_paths)):
        lq = utils.path_to_tensor(lq_path)
        img_name = lq_path.split('/')[-1]
        print(img_name, end=',')
        hqs = get_hqs(img_name)
        target = np.float32(utils.load_img(target_path)) / 255.
        lq = lq.to(model.device).unsqueeze(0)
        hqs = hqs.to(model.device).unsqueeze(0)

        print(parameter_count_table(model.net_g))
        flops = FlopCountAnalysis(model.net_g, (lq,hqs))
        print(f"FLOPs: {flops.total() / 1e9:.2f} G")
        exit(0)

        with torch.no_grad():
            res = model.net_g(lq, hqs).squeeze(0)
        res = torch.clamp(res, 0, 1).detach().permute(1, 2, 0).cpu().numpy()
        # stx()
        psnr.append(utils.calculate_psnr(target, res))
        ssim.append(utils.calculate_ssim(target, res))
        lpips.append(utils.calculate_lpips(target, res))
        # utils.save_img(osp.join(save_dir, img_name), (res*255).astype('uint8'))

    # print(psnr)
    psnr = np.mean(np.array(psnr))
    ssim = np.mean(np.array(ssim))
    lpips = np.mean(np.array(lpips))
    print(f"\npsnr: {psnr:.4f}, ssim: {ssim:.4f}, lpips: {lpips:.4f}")

if __name__ == '__main__':
    main()