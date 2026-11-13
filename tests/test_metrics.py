"""
Unit Tests for Evaluation Metrics (MSE, PSNR, SSIM, Capacity, BER).
"""

import numpy as np
import torch
from ai.metrics.mse import calculate_mse
from ai.metrics.psnr import calculate_psnr
from ai.metrics.ssim import calculate_ssim
from ai.metrics.payload import calculate_bpp, calculate_ber


def test_identical_images_metrics():
    """Tests that identical images yield MSE=0, PSNR=inf, SSIM=1.0."""
    img = np.random.rand(256, 256, 3).astype(np.float32)

    assert calculate_mse(img, img) == 0.0
    assert calculate_psnr(img, img) == float("inf")
    assert calculate_ssim(img, img) == 1.0


def test_modified_images_metrics():
    """Tests metrics for known slight modification."""
    img1 = np.ones((256, 256, 3), dtype=np.float32) * 0.5
    img2 = img1.copy()
    img2[0, 0, 0] += 0.1  # Slight pixel perturbation

    mse = calculate_mse(img1, img2)
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)

    assert mse > 0.0
    assert psnr < 100.0
    assert ssim < 1.0 and ssim > 0.99


def test_bpp_and_ber_metrics():
    """Tests Bits-Per-Pixel (bpp) and Bit Error Rate (BER) functions."""
    bpp = calculate_bpp(65536, (256, 256, 3))
    assert bpp == 1.0  # 65536 bits / (256*256 pixels) = 1.0 bpp

    bits_orig = [1, 0, 1, 1, 0, 0]
    bits_rec = [1, 0, 1, 0, 0, 0]  # 1 bit difference out of 6
    ber = calculate_ber(bits_orig, bits_rec)

    assert abs(ber - (1.0 / 6.0)) < 1e-5
