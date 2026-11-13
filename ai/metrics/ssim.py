"""
Structural Similarity Index Measure (SSIM) Metric Module.

What it measures:
    Evaluates structural similarity, luminance, and contrast preservation between cover and stego images.

Direction:
    Higher is better. Range is [-1.0, 1.0], where 1.0 indicates perfect structural identity.

Steganographic Relevance:
    Unlike MSE/PSNR which only measure pixel-wise difference, SSIM captures human visual system (HVS)
    perception of structural degradation and texture distortion.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def _scipy_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    max_val: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
    sigma: float = 1.5
) -> float:
    """
    Computes Structural Similarity Index (SSIM) using standard Gaussian window filtering.
    """
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = gaussian_filter(img1, sigma)
    mu2 = gaussian_filter(img2, sigma)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 ** 2, sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(np.mean(ssim_map))


def calculate_ssim(
    img1: np.ndarray | torch.Tensor,
    img2: np.ndarray | torch.Tensor,
    max_val: float = 1.0
) -> float:
    """
    Calculates Structural Similarity Index Measure (SSIM).

    Args:
        img1: Original cover image (HWC array or CHW tensor).
        img2: Modified stego image.
        max_val: Dynamic range (1.0 for normalized float, 255.0 for uint8).

    Returns:
        SSIM index between -1.0 and 1.0.
    """
    if isinstance(img1, torch.Tensor):
        img1_arr = img1.detach().cpu().numpy()
    else:
        img1_arr = img1

    if isinstance(img2, torch.Tensor):
        img2_arr = img2.detach().cpu().numpy()
    else:
        img2_arr = img2

    # Transpose CHW -> HWC if PyTorch tensor shape detected
    if img1_arr.ndim == 3 and img1_arr.shape[0] in (1, 3):
        img1_arr = np.transpose(img1_arr, (1, 2, 0))
    if img2_arr.ndim == 3 and img2_arr.shape[0] in (1, 3):
        img2_arr = np.transpose(img2_arr, (1, 2, 0))

    if img1_arr.shape != img2_arr.shape:
        raise ValueError(f"Shape mismatch: {img1_arr.shape} vs {img2_arr.shape}")

    if HAS_SKIMAGE:
        if img1_arr.ndim == 3 and img1_arr.shape[2] in (3, 4):
            val = skimage_ssim(img1_arr, img2_arr, data_range=max_val, channel_axis=2)
        else:
            if img1_arr.ndim == 3 and img1_arr.shape[2] == 1:
                img1_arr = img1_arr[:, :, 0]
                img2_arr = img2_arr[:, :, 0]
            val = skimage_ssim(img1_arr, img2_arr, data_range=max_val)
        return float(val)

    # Fallback to scipy SSIM calculation
    if img1_arr.ndim == 3:
        ssims = []
        for c in range(img1_arr.shape[2]):
            ssims.append(_scipy_ssim(img1_arr[:, :, c], img2_arr[:, :, c], max_val=max_val))
        return float(np.mean(ssims))
    else:
        return _scipy_ssim(img1_arr, img2_arr, max_val=max_val)
