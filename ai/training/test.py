"""
Test Evaluation Module for Steganography Model.
Computes empirical metrics (PSNR, SSIM, MSE, BER, Bit Accuracy) over the test dataset split.
"""

from typing import Dict, Any
import torch
import numpy as np
from torch.utils.data import DataLoader

from ai.metrics.mse import calculate_mse
from ai.metrics.psnr import calculate_psnr
from ai.metrics.ssim import calculate_ssim


def evaluate_test_set(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates test set performance metrics on held-out images.

    Args:
        model: Trained AdaptiveStegoModel instance.
        test_loader: Test dataset DataLoader.
        device: Torch evaluation device.

    Returns:
        Dictionary containing average PSNR, SSIM, MSE, BER, and bit accuracy metrics.
    """
    model.eval()
    psnr_list = []
    ssim_list = []
    mse_list = []
    ber_list = []
    acc_list = []

    with torch.no_grad():
        for batch in test_loader:
            cover = batch["image"].to(device)
            b, c, h, w = cover.shape

            payload_map = torch.randint(0, 2, (b, 1, h, w), dtype=torch.float32, device=device)

            output = model(cover, payload_map)
            stego = output["stego_image"]
            extracted = output["extracted_payload"]

            pred_bits = (extracted > 0.5).float()

            for i in range(b):
                c_i = cover[i].cpu().numpy()
                s_i = stego[i].cpu().numpy()

                p_true = payload_map[i]
                p_pred = pred_bits[i]

                # Fast vectorized Bit Error Rate (BER) & accuracy
                errors = (p_pred != p_true).float().sum().item()
                total_bits = p_true.numel()
                ber_val = errors / max(total_bits, 1)
                acc_val = 1.0 - ber_val

                mse_val = calculate_mse(c_i, s_i)
                psnr_val = calculate_psnr(c_i, s_i, max_val=1.0)
                ssim_val = calculate_ssim(c_i, s_i, max_val=1.0)

                mse_list.append(mse_val)
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                ber_list.append(ber_val)
                acc_list.append(acc_val)

    if len(mse_list) == 0:
        return {
            "avg_mse": 0.0,
            "avg_psnr": 0.0,
            "avg_ssim": 0.0,
            "avg_ber": 0.0,
            "avg_bit_acc": 0.0
        }

    return {
        "avg_mse": float(np.mean(mse_list)),
        "avg_psnr": float(np.mean(psnr_list)),
        "avg_ssim": float(np.mean(ssim_list)),
        "avg_ber": float(np.mean(ber_list)),
        "avg_bit_acc": float(np.mean(acc_list))
    }
