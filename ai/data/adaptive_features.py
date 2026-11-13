"""
Supporting Feature Extraction & Reference Suitability Map Generation Module.

Generates normalized supporting feature maps (Sobel edge, local variance, Laplacian texture)
and constructs a heuristic reference suitability map S_ref(x, y) in range [0.0, 1.0].

Formula:
S_ref = Normalize( w_edge * Edge + w_variance * Variance + w_texture * Texture - w_smooth * Penalty_smooth )

IMPORTANT SCIENTIFIC NOTICE:
This reference suitability map is a HEURISTIC representation derived from image signal processing.
It serves as target supervisory signal for initial training. It is NOT ground-truth security.
"""

from typing import Tuple, Union
import numpy as np
from scipy import ndimage
import torch

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def extract_supporting_features(
    image: Union[np.ndarray, torch.Tensor],
    variance_window: int = 7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts normalized Sobel edge map, local variance map, and Laplacian texture map.

    Args:
        image: HWC or CHW image array/tensor in range [0, 1].
        variance_window: Window size for local variance calculation.

    Returns:
        Tuple of (edge_map, variance_map, texture_map)
        Each has shape (H, W, 1) in float32 [0.0, 1.0].
    """
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))
    else:
        img_np = image

    # RGB to Gray
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        gray = 0.2989 * img_np[:, :, 0] + 0.5870 * img_np[:, :, 1] + 0.1140 * img_np[:, :, 2]
    elif img_np.ndim == 3 and img_np.shape[2] == 1:
        gray = img_np[:, :, 0]
    else:
        gray = img_np

    # 1. Edge Map (Sobel magnitude)
    if HAS_OPENCV:
        img_ubyte = (gray * 255.0).astype(np.uint8)
        sobelx = cv2.Sobel(img_ubyte, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_ubyte, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx**2 + sobely**2)
    else:
        sobelx = ndimage.sobel(gray, axis=1)
        sobely = ndimage.sobel(gray, axis=0)
        edge_mag = np.sqrt(sobelx**2 + sobely**2)

    max_edge = np.max(edge_mag)
    edge_map = (edge_mag / max_edge) if max_edge > 1e-6 else np.zeros_like(edge_mag)

    # 2. Local Variance Map
    if HAS_OPENCV:
        ksize = (variance_window, variance_window)
        mean = cv2.blur(gray, ksize)
        mean_sq = cv2.blur(gray**2, ksize)
    else:
        mean = ndimage.uniform_filter(gray, size=variance_window)
        mean_sq = ndimage.uniform_filter(gray**2, size=variance_window)

    local_var = np.maximum(0.0, mean_sq - mean**2)
    max_var = np.max(local_var)
    var_map = (local_var / max_var) if max_var > 1e-6 else np.zeros_like(local_var)

    # 3. Texture Map (Laplacian response)
    if HAS_OPENCV:
        img_ubyte = (gray * 255.0).astype(np.uint8)
        laplace = cv2.Laplacian(img_ubyte, cv2.CV_64F)
        texture_mag = np.abs(laplace)
    else:
        texture_mag = np.abs(ndimage.laplace(gray))

    max_tex = np.max(texture_mag)
    texture_map = (texture_mag / max_tex) if max_tex > 1e-6 else np.zeros_like(texture_mag)

    # Expand channel dimension -> (H, W, 1)
    edge_map = np.expand_dims(edge_map.astype(np.float32), axis=-1)
    var_map = np.expand_dims(var_map.astype(np.float32), axis=-1)
    texture_map = np.expand_dims(texture_map.astype(np.float32), axis=-1)

    return edge_map, var_map, texture_map


def generate_reference_suitability_map(
    image: Union[np.ndarray, torch.Tensor],
    edge_weight: float = 0.4,
    texture_weight: float = 0.3,
    variance_weight: float = 0.3,
    smooth_penalty: float = 0.2,
    variance_window: int = 7
) -> np.ndarray:
    """
    Generates a heuristic reference suitability map S_ref(x, y) in range [0.0, 1.0].

    Formulation:
        S_raw = w_edge * Edge + w_variance * Var + w_texture * Tex - w_smooth * SmoothPenalty
        where SmoothPenalty = 1.0 - (Edge + Var + Tex) / 3.0

    Args:
        image: HWC or CHW input image array/tensor in range [0, 1].
        edge_weight: Weight for edge map.
        texture_weight: Weight for texture map.
        variance_weight: Weight for variance map.
        smooth_penalty: Weight for smooth-region penalty.
        variance_window: Variance sliding window size.

    Returns:
        Reference suitability map array of shape (H, W, 1) in range float32 [0.0, 1.0].
    """
    edge_map, var_map, texture_map = extract_supporting_features(image, variance_window=variance_window)

    # Calculate smooth region penalty (regions with very low edge, variance, and texture)
    combined_activity = (edge_map + var_map + texture_map) / 3.0
    smooth_penalty_map = 1.0 - combined_activity

    fused = (
        edge_weight * edge_map +
        variance_weight * var_map +
        texture_weight * texture_map -
        smooth_penalty * smooth_penalty_map
    )

    # Shift and scale to range [0.0, 1.0]
    min_val = np.min(fused)
    max_val = np.max(fused)

    if (max_val - min_val) > 1e-6:
        suitability_map = (fused - min_val) / (max_val - min_val)
    else:
        suitability_map = np.clip(fused, 0.0, 1.0)

    return suitability_map.astype(np.float32)
