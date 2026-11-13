"""
Peak Signal-to-Noise Ratio (PSNR) Metric Module.

What it measures:
    Evaluates visual imperceptibility in decibels (dB) between cover image and stego image.
    Formula: PSNR = 10 * log10(MAX^2 / MSE)

Direction:
    Higher is better. Values above 40 dB indicate high visual quality with imperceptible modifications.

Steganographic Relevance:
    PSNR is the standard academic benchmark for evaluating steganographic imperceptibility.
"""

import math
import numpy as np
import torch
from ai.metrics.mse import calculate_mse


def calculate_psnr(
    img1: np.ndarray | torch.Tensor,
    img2: np.ndarray | torch.Tensor,
    max_val: float = 1.0
) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Args:
        img1: Cover image array/tensor.
        img2: Stego image array/tensor.
        max_val: Maximum dynamic pixel range (1.0 for normalized float, 255.0 for uint8).

    Returns:
        PSNR value in decibels (dB). Returns float('inf') if MSE is 0.
    """
    mse = calculate_mse(img1, img2)
    if mse < 1e-12:
        return float("inf")

    psnr = 10.0 * math.log10((max_val ** 2) / mse)
    return float(psnr)
