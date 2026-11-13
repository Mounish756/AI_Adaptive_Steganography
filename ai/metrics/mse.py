"""
Mean Squared Error (MSE) Metric Module.

What it measures:
    Calculates the average squared difference between pixel values of the original cover image
    and the modified stego image.

Direction:
    Lower is better. A value of 0.0 indicates identical images.

Steganographic Relevance:
    MSE provides a baseline mathematical measure of visual distortion introduced during embedding.
"""

import numpy as np
import torch


def calculate_mse(
    img1: np.ndarray | torch.Tensor,
    img2: np.ndarray | torch.Tensor
) -> float:
    """
    Calculates Mean Squared Error (MSE) between two images.

    Args:
        img1: Original cover image (NumPy array or PyTorch Tensor in range [0, 1] or [0, 255]).
        img2: Modified stego image (matching type and dimensions).

    Returns:
        Float MSE value >= 0.0.
    """
    if isinstance(img1, torch.Tensor):
        img1_arr = img1.detach().cpu().numpy()
    else:
        img1_arr = img1

    if isinstance(img2, torch.Tensor):
        img2_arr = img2.detach().cpu().numpy()
    else:
        img2_arr = img2

    if img1_arr.shape != img2_arr.shape:
        raise ValueError(f"Shape mismatch: {img1_arr.shape} vs {img2_arr.shape}")

    diff = img1_arr.astype(np.float64) - img2_arr.astype(np.float64)
    mse_val = np.mean(diff ** 2)
    return float(mse_val)
