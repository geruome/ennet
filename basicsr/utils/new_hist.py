import torch
import torchvision.transforms as transforms
from PIL import Image
import time

def histogram_equalization(img_tensor: torch.Tensor) -> torch.Tensor:
    """对图像张量直方图均衡化

    Args:
        img_tensor (torch.Tensor): shape 为 [b, c, h, w] 的图像张量，dtype 为 torch.float32，数值范围为 [0, 1]

    Returns:
        torch.Tensor: 直方图均衡化后的张量
    """
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # 使用 GPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # img_tensor = img_tensor.to(device)

    b, c, h, w = img_tensor.shape

    for channel in range(c):
        flat = img_tensor[0, channel].flatten()
        # 计算直方图
        hist = torch.histc(flat, bins=256, min=0.0, max=1.0)
        # 计算累积分布函数（CDF）
        cdf = hist.cumsum(dim=0)
        cdf_min = cdf[cdf > 0].min()
        # 归一化
        cdf_normalized = (cdf - cdf_min) / (cdf.max() - cdf_min + 1e-5)
        cdf_normalized = (cdf_normalized * 255).byte()
        # 映射回原始图像
        img_tensor[0, channel] = cdf_normalized[(flat * 255).long()].view(h, w).float() / 255

    img_tensor = img_tensor.squeeze(0)
    return img_tensor


def histogram_equalization_luminance(img_tensor: torch.Tensor) -> torch.Tensor:
    """对图像张量直方图均衡化，只均衡化亮度通道

    Args:
        img_tensor (torch.Tensor): shape 为 [b, c, h, w] 的图像张量，dtype 为 torch.float32，数值范围为 [0, 1]

    Returns:
        torch.Tensor: 对亮度进行直方图均衡化后的张量
    """
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)

    img_tensor = torch.clamp(img_tensor, 0, 1)
    # print(img_tensor.shape, '111111111')
    # 使用 GPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # img_tensor = img_tensor.to(device)
    device = img_tensor.device

    # 将 RGB 图像转换为 YUV
    r, g, b = img_tensor[0, 0], img_tensor[0, 1], img_tensor[0, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b

    y = y.unsqueeze(0)  # 1 x h x w

    # 对亮度通道进行直方图均衡化
    flat = y.flatten()
    hist = torch.histc(flat, bins=256, min=0.0, max=1.0)
    cdf = hist.cumsum(dim=0)
    cdf_min = cdf[cdf > 0].min()
    cdf_normalized = (cdf - cdf_min) / (cdf.max() - cdf_min + 1e-5)
    cdf_normalized = (cdf_normalized * 255).byte()
    cdf_normalized = cdf_normalized.clamp(0, 200)
    y_equalized = cdf_normalized[(flat * 255).long()].float().view_as(y).to(device) / 255.0

    y = y_equalized.squeeze(0)
    # print(y.shape, '222222222222')
    
    # 将 YUV 转换回 RGB
    r_new = y + 1.13983 * v
    g_new = y - 0.39465 * u - 0.58060 * v
    b_new = y + 2.03211 * u

    img_tensor[0, 0] = r_new.clamp(0, 1)
    img_tensor[0, 1] = g_new.clamp(0, 1)
    img_tensor[0, 2] = b_new.clamp(0, 1)

    img_tensor = img_tensor.squeeze(0)
    return img_tensor

