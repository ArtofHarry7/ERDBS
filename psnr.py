import os
import re
import sys
import math
import torch
import numpy as np
# import cv2
from skimage import color, metrics

def updateMaxPsnr():
    models = os.listdir('parameters')
    models = [m.split('-') for m in models]
    models = [[float(m[0]), m[1][:-4]] for m in models]
    maxPsnr = [0 for _ in range(21)]
    for m in models:
        maxPsnr[len(m[1])] = max(maxPsnr[len(m[1])], m[0])
    return maxPsnr

def isIn(folder, genome):
    models = os.listdir(folder)
    models = [m.split('-') for m in models]
    models = [[float(m[0]), m[1][:-4]] for m in models]
    mx = 0
    seq = ''
    for m in models:
        if m[1] == ''.join(genome):
            mx = max(mx, m[0])
    if mx:
        seq = f'{mx}-{"".join(genome)}.pth'
    return seq

def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = 0
    y: torch.Tensor = 16+(65.738 * r)/256 + (129.057 * g)/256 + (25.064 * b)/256
    cb: torch.Tensor = 128+(b - y) * .564 + delta
    cr: torch.Tensor = 128+(r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        
        
        img1=rgb_to_ycbcr(img1)
        img2=rgb_to_ycbcr(img2)
        mse = torch.mean((img1 - img2) ** 2)
        
        
        
        return 10 * torch.log10(255.0 / torch.sqrt(mse))
        
        # psnr = metrics.peak_signal_noise_ratio(img1, img2)
        # return psnr
   

 # class SSIM:
#     """Structure Similarity
#     img1, img2: [0, 255]"""

#     def __init__(self):
#         self.name = "SSIM"

#     @staticmethod
#     def __call__(img1, img2):
#         if not img1.shape == img2.shape:
#             raise ValueError("Input images must have the same dimensions.")
#         if img1.ndim == 2:  # Grey or Y-channel image
#             return self._ssim(img1, img2)
#         elif img1.ndim == 3:
#             if img1.shape[2] == 3:
#                 ssims = []
#                 for i in range(3):
#                     ssims.append(self._ssim(img1, img2))
#                 return np.array(ssims).mean()
#             elif img1.shape[2] == 1:
#                 return self._ssim(np.squeeze(img1), np.squeeze(img2))
#         else:
#             raise ValueError("Wrong input image dimensions.")

#     @staticmethod
#     def _ssim(img1, img2):
#         C1 = (0.01 * 255) ** 2
#         C2 = (0.03 * 255) ** 2

#         img1 = img1.astype(np.float64)
#         img2 = img2.astype(np.float64)
#         kernel = cv2.getGaussianKernel(11, 1.5)
#         window = np.outer(kernel, kernel.transpose())

#         mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#         mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#         mu1_sq = mu1 ** 2
#         mu2_sq = mu2 ** 2
#         mu1_mu2 = mu1 * mu2
#         sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#         sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#         sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#         ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#         return ssim_map.mean()