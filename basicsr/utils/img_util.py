import cv2
import collections
import math
import numpy as np
import os
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage import img_as_ubyte
from pdb import set_trace as stx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    if img_np is None:
        raise Exception('None .. !!!')
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img

def imfrombytesDP(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    if img_np is None:
        raise Exception('None .. !!!')
    img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    if float32:
        img = img.astype(np.float32) / 65535.
    return img

def padding(img_lq, img_gt, gt_size):
    h, w, _ = img_lq.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    
    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    if img_gt.ndim == 2:
        img_gt = np.expand_dims(img_gt, axis=2)
    return img_lq, img_gt

def padding_DP(img_lqL, img_lqR, img_gt, gt_size):
    h, w, _ = img_gt.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    
    if h_pad == 0 and w_pad == 0:
        return img_lqL, img_lqR, img_gt

    img_lqL = cv2.copyMakeBorder(img_lqL, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_lqR = cv2.copyMakeBorder(img_lqR, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt  = cv2.copyMakeBorder(img_gt,  0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    return img_lqL, img_lqR, img_gt

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [
                v[crop_border:-crop_border, crop_border:-crop_border, ...]
                for v in imgs
            ]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border,
                        ...]


def DWT(x):
    assert torch.is_tensor(x) # CHW
    assert len(x.shape) in [3, 4]
    dim = 4
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        dim = 3
    x01 = x[:, :, 0::2, :]
    x02 = x[:, :, 1::2, :]
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = (x1 + x2 + x3 + x4)/4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    if dim == 3:
        x_LL = x_LL.squeeze(0)
        x_HL = x_HL.squeeze(0)
        x_LH = x_LH.squeeze(0)
        x_HH = x_HH.squeeze(0)
    x_HL = x_HL.unsqueeze(0)
    x_LH = x_LH.unsqueeze(0)
    x_HH = x_HH.unsqueeze(0)
    x_H = torch.cat((x_HL, x_LH, x_HH), dim = 0)
    return x_LL, x_H



def IWT(x_LL, x_H):
    assert torch.is_tensor(x_LL) and torch.is_tensor(x_H)
    x_H = x_H.to(x_LL.device)
    # print(x_LL.device, x_H.device, '??????????')
    assert len(x_LL.shape) in [3, 4]
    dim = 4
    if len(x_LL.shape) == 3:
        x_LL = x_LL.unsqueeze(0)
        x_H = x_H.unsqueeze(1)
        dim = 3 
    r = 2
    in_batch, in_channel, in_height, in_width = x_LL.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, in_channel , r * in_height, r * in_width
    x_LL *= 4
    x_HL = x_H[0, :, :, :]
    x_LH = x_H[1, :, :, :]
    x_HH = x_H[2, :, :, :]

    x1 = (x_LL - x_LH - x_HL + x_HH) / 4
    x2 = (x_LL - x_LH + x_HL - x_HH) / 4
    x3 = (x_LL + x_LH - x_HL - x_HH) / 4
    x4 = (x_LL + x_LH + x_HL + x_HH) / 4
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x1.device)

    h[:, :, 0::2, 0::2] = x1
    h[:, :, 1::2, 0::2] = x2
    h[:, :, 0::2, 1::2] = x3
    h[:, :, 1::2, 1::2] = x4
    if dim == 3:
        h = h.squeeze(0)
    return h


def cal(l, P):
    v = (l-1)//P + 1
    v = v*P - l
    return (v//2, v-v//2)


min_DWT_cnt = 1
HW_min = 1000000


def DWTs(x): # 输入输出的都是tensor
    # print(x.shape, '-------------') 
    C, H, W = x.shape
    # 多次DWT
    cnt = min_DWT_cnt
    tmp = H*W
    for _ in range(cnt):
        tmp //= 4
    while tmp>HW_min:
        tmp //= 4; cnt += 1
    # 非2^cnt倍数要补足
    P = 1 # 子模型要求输出至少为什么的倍数
    for _ in range(cnt): P*=2
    lh, rh = cal(H, P)
    lw, rw = cal(W, P)
    x = F.pad(x, (lw, rw, lh, rh), mode='replicate')
    xhs = []
    for _ in range(cnt):
        x_LL, x_H = DWT(x)
        x = x_LL; xhs.append(x_H)
    xhs.reverse()
    return x, xhs


def IWTs(x, xhs):
    # 多次IWT还原
    for x_H in xhs:    
        x = IWT(x, x_H)
    return x


# def light_effects_seg(image_pt, threshold = 192):
#     """
#     image_pt: torch.Tensor, shape (C, H, W)
#     """
#     mask_0 = torch.zeros_like(image_pt)
#     mask_1 = torch.ones_like(image_pt)
#     rgb_mask = torch.where(image_pt > threshold, mask_0, image_pt)
#     brightness_mask = rgb_mask.sum(dim=0)
#     mask = brightness_mask > 0
#     # expand mask to 3 channels
#     mask = torch.stack([mask]*3, dim=0)
#     light_effects = torch.zeros_like(image_pt)
#     light_effects[mask] = image_pt[mask]
#     masked_image = image_pt.clone()
#     masked_image[mask] = 0
#     return masked_image, mask


def light_effects_seg(image_pt, threshold = 200):
    """
    image_pt: torch.Tensor, shape (C, H, W)
    """
    threshold /= 255
    mask_0 = torch.zeros_like(image_pt, dtype=torch.bool)
    mask_1 = torch.ones_like(image_pt, dtype=torch.bool)
    r, g, b = image_pt[0], image_pt[1], image_pt[2]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Y = Y.unsqueeze(0).repeat(3, 1, 1)
    mask = torch.where(Y > threshold, mask_1, mask_0)
    # brightness_mask = rgb_mask.sum(dim=0)
    # mask = brightness_mask > 0
    # # expand mask to 3 channels
    # mask = torch.stack([mask]*3, dim=0)
    masked_image = image_pt.clone()
    masked_image[mask] = 0
    return masked_image, mask


def light_seg_recover(image_hq, image_lq, mask):
    image_hq[mask] = image_lq[mask]
    return image_hq

# def light_effects_seg(image_array: np.ndarray, threshold: int):   
#     mask = np.any(image_array > threshold, axis=-1)
#     light_effects = np.zeros_like(image_array)
#     light_effects[mask] = image_array[mask]
#     image_array[mask] = [0, 0, 0]
#     return image_array


# def light_seg_recover(image_hq, image_lq):
#     threshold = 80
#     mask = np.any(image_hq < threshold, axis=-1)
#     image_hq[mask] = image_lq[mask]
#     return image_hq


def draw_histogram(grayscale):
    # 对图像进行通道拆分
    hsi_i = grayscale[:, :, 0]
    color_key = []
    color_count = []
    color_result = []
    histogram_color = list(hsi_i.ravel())  # 将多维数组转换成一维数组
    color = dict(collections.Counter(histogram_color))  # 统计图像中每个亮度级出现的次数
    color = sorted(color.items(), key=lambda item: item[0])  # 根据亮度级大小排序
    for element in color:
        key = list(element)[0]
        count = list(element)[1]
        color_key.append(key)
        color_count.append(count)
    for i in range(0, 256):
        if i in color_key:
            num = color_key.index(i)
            color_result.append(color_count[num])
        else:
            color_result.append(0)
    color_result = np.array(color_result)
    return color_result


def histogram_equalization(histogram_e, lut_e, image_e):
    sum_temp = 0
    cf = []
    for i in histogram_e:
        sum_temp += i
        cf.append(sum_temp)
    for i, v in enumerate(lut_e):
        lut_e[i] = int(150.0 * (cf[i] / sum_temp) + 50) #[50,200]     255.0 * (cf[i] / sum_temp) + 0.5
    equalization_result = lut_e[image_e]
    return equalization_result
    

def Histogram(image):
    histogram_original = draw_histogram(image)
    lut = np.zeros(256, dtype=image.dtype)  # 创建空的查找表
    image = histogram_equalization(histogram_original, lut, image)  # 均衡化处理
    return image
    

def padding_img(img, P=16, return_edges=False):
    '''
    input: tensor, output: tensor
    P is block_size or mininum multiple for submodel, default 16
    '''
    C, H, W = img.shape
    # 多次DWT
    cnt = min_DWT_cnt
    tmp = H*W
    for _ in range(cnt):
        tmp //= 4
    while tmp>HW_min:
        tmp//=4; cnt+=1
    # 非2^cnt倍数要补足
    for _ in range(cnt): P*=2
    C, H, W = img.shape
    lh, rh = cal(H, P)
    lw, rw = cal(W, P)
    res = F.pad(img, (lw, rw, lh, rh), mode='replicate')
    if return_edges:
        res = (res, (lh, rh, lw, rw))
    return res


# def read_input(input_save_path):
#     input = cv2.imread(input_save_path) # H,W,C
#     input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
#     transf = transforms.ToTensor()
#     ip_tensor = transf(input)  # C,H,W
    
#     # padding
#     cnt = 1  # DWT次数 
#     # 非2^cnt倍数要补足
#     P = 8 # 子模型要求输出至少为什么的倍数
#     for _ in range(cnt): P*=2
#     C, H, W = ip_tensor.shape
#     lh, rh = cal(H, P)
#     lw, rw = cal(W, P)
#     ip_tensor = F.pad(ip_tensor, (lw, rw, lh, rh), mode='replicate')

#     input = torch.clamp(ip_tensor, 0, 1).detach().permute(1, 2, 0).cpu().numpy() # H,W,C RGB
    
#     input = img_as_ubyte(input)
#     return input


def ndarray_to_tensor(img):
    transf = transforms.ToTensor()
    img_tensor = transf(img)  # C,H,W
    return img_tensor


def tensor_to_ndarray(img): # convert tensor to ndarray (HWC,RGB)
    return torch.clamp(img, 0, 1).detach().permute(1, 2, 0).cpu().numpy()


# def resize_img(img): # tensor -> tensor
#     # img = ndarray_to_tensor(img, light_seg=False)
#     img = padding_img(img)
#     # img = tensor_to_ndarray(img)
#     # img = img_as_ubyte(img)
#     return img
    

def gamma_correction(img_tensor: torch.Tensor, gamma: float=1.7) -> torch.Tensor:
    """对图像gamma矫正

    Args:
        img_tensor (torch.Tensor): shape 为 [b, c, h, w] 的图像张量，dtype 为 torch.float32，数值范围为 [0, 1]
        gamma (float): gamma 值，默认值为 2.2

    Returns:
        torch.Tensor: 对亮度进行gamma 矫正后的张量
    """
    img_tensor = torch.clamp(img_tensor, 0, 1)
    # Gamma 矫正
    img_tensor = torch.pow(img_tensor, gamma)
    return img_tensor


# def denoise(img_tensor): # C,H,W
#     img = tensor_to_ndarray(img_tensor)
#     img = cv2.medianBlur(img, 5)
#     denoised_tensor = ndarray_to_tensor(img)
#     return denoised_tensor


def denoise(img_tensor): 
    kernel_size = 5
    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
    _, C, H, W = img_tensor.shape
    unfolded = F.unfold(img_tensor, kernel_size, padding=kernel_size // 2)  # [1, C * K * K, L]
    unfolded = unfolded.view(1, C, kernel_size * kernel_size, H * W)  # [1, C, K * K, H * W]
    median_values, _ = unfolded.median(dim=2)  # [1, C, H * W]
    denoised_tensor = median_values.view(1, C, H, W).squeeze(0)
    return denoised_tensor