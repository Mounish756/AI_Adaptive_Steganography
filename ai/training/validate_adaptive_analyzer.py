"""
Validation Module for Adaptive Region Analyzer.

Evaluates validation loss over validation dataset split, calculates MSE & SSIM similarity
between AI predicted maps and reference maps, and saves visual comparison artifacts.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Dict, Any
import torch
import numpy as np
from torch.utils.data import DataLoader

from ai.models.adaptive_analyzer import AdaptiveRegionAnalyzer
from ai.data.adaptive_features import generate_reference_suitability_map
from ai.data.preprocessing import to_tensor
from ai.training.losses import AdaptiveAnalyzerLoss
from ai.metrics.mse import calculate_mse
from ai.metrics.ssim import calculate_ssim


def validate_adaptive_analyzer_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: AdaptiveAnalyzerLoss,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates model over validation dataloader without computing gradients.

    Args:
        model: AdaptiveRegionAnalyzer PyTorch module.
        val_loader: Validation DataLoader.
        criterion: AdaptiveAnalyzerLoss criterion instance.
        device: Target evaluation device.

    Returns:
        Dictionary of averaged validation metrics (total_loss, map_loss, tv_loss, avg_mse, avg_ssim).
    """
    model.eval()
    running_total_loss = 0.0
    running_map_loss = 0.0
    running_tv_loss = 0.0
    mse_list = []
    ssim_list = []
    total_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)  # (B, 3, 256, 256)
            b = images.shape[0]

            # Generate reference suitability maps for batch
            ref_maps_list = []
            for i in range(b):
                img_np = images[i].cpu().numpy().transpose(1, 2, 0)
                ref_map_np = generate_reference_suitability_map(img_np)  # (256, 256, 1)
                ref_map_tensor = to_tensor(ref_map_np)  # (1, 256, 256)
                ref_maps_list.append(ref_map_tensor)

            ref_maps = torch.stack(ref_maps_list, dim=0).to(device)  # (B, 1, 256, 256)

            # Predict map
            pred_maps = model(images)  # (B, 1, 256, 256)

            losses = criterion(pred_maps, ref_maps)

            running_total_loss += losses["total_loss"].item()
            running_map_loss += losses["map_reconstruction_loss"].item()
            running_tv_loss += losses["smoothness_loss"].item()
            total_batches += 1

            for i in range(b):
                p_np = pred_maps[i, 0].cpu().numpy()
                r_np = ref_maps[i, 0].cpu().numpy()
                mse_list.append(calculate_mse(p_np, r_np))
                ssim_list.append(calculate_ssim(p_np, r_np, max_val=1.0))

    if total_batches == 0:
        return {
            "total_loss": 0.0,
            "map_reconstruction_loss": 0.0,
            "smoothness_loss": 0.0,
            "avg_mse": 0.0,
            "avg_ssim": 0.0
        }

    return {
        "total_loss": running_total_loss / total_batches,
        "map_reconstruction_loss": running_map_loss / total_batches,
        "smoothness_loss": running_tv_loss / total_batches,
        "avg_mse": float(np.mean(mse_list)),
        "avg_ssim": float(np.mean(ssim_list))
    }
