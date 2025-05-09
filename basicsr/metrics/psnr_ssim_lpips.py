import numpy as np
import os
import cv2
import math
import lpips
import torch
from skimage import img_as_ubyte

Use_GT_mean = False

def calculate_lpips(img1, img2, gt_mean=Use_GT_mean):
    """
    img1/img2 (numpy.ndarray): Input image with shape (H, W, C) and pixel values in the range [0,1].
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if gt_mean:
        mean_1 = cv2.cvtColor(img1.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        mean_2 = cv2.cvtColor(img2.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        img2 = np.clip(img2 * (mean_1 / mean_2), 0, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global lpips_model
    if "lpips_model" not in globals():
        print("Initializing LPIPS model...")
        lpips_model = lpips.LPIPS(net='alex').to(device)
    

    img1 = torch.tensor(img1, device=device).permute(2, 0, 1).unsqueeze(0).float()
    img2 = torch.tensor(img2, device=device).permute(2, 0, 1).unsqueeze(0).float()
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1
    with torch.no_grad():
        lpips_score = lpips_model(img1, img2)
    return lpips_score.item()


def calculate_psnr(img1, img2, gt_mean=Use_GT_mean):
    """
    img1/img2 (numpy.ndarray): Input image with shape (H, W, C) and pixel values in the range [0,1].
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if gt_mean:
        mean_1 = cv2.cvtColor(img1.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        mean_2 = cv2.cvtColor(img2.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        img2 = np.clip(img2 * (mean_1 / mean_2), 0, 1)
    mse_ = np.mean((img1 - img2) ** 2)
    if mse_ == 0:
        return 100
    return 10 * math.log10(1 / mse_)


def calculate_ssim(img1, img2, gt_mean=Use_GT_mean):
    '''
    img1/img2 (numpy.ndarray): Input image with shape (H, W, C) and pixel values in the range [0,1].
    '''
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if gt_mean:
        mean_1 = cv2.cvtColor(img1.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        mean_2 = cv2.cvtColor(img2.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        img2 = np.clip(img2 * (mean_1 / mean_2), 0, 1)
    img1 = img_as_ubyte(img1)
    img2 = img_as_ubyte(img2)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def l1_loss(pred, target) -> float:
    return np.mean(np.abs(pred -target))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
