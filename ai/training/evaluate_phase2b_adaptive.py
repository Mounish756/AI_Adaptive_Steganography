"""
Phase 2B Evaluation Script: Quantitative Analysis of Adaptive Modulation.

Compares:
1. Adaptive Embedding: residual = perturbation * suitability_map * scaling_factor
2. Uniform (Non-Adaptive) Embedding: residual = perturbation * mean(suitability) * scaling_factor
3. Spatial Region Analysis: evaluates distortion and bit accuracy in high-suitability
   (textured/edge) vs low-suitability (smooth/flat) regions.
"""

import os
import sys
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
import yaml

from ai.models.full_model import AdaptiveStegoModel
from ai.data.dataset import SteganographyDataset
from ai.metrics.mse import calculate_mse
from ai.metrics.psnr import calculate_psnr
from ai.metrics.ssim import calculate_ssim


def evaluate_phase2b_adaptive_effect(
    model_path: str = "checkpoints/phase2a/best_model.pth",
    dataset_dir: str = "datasets/coco_subset",
    device_name: str = "cpu"
) -> Dict[str, Any]:
    device = torch.device(device_name)
    
    # Load model
    model = AdaptiveStegoModel(base_dim=32, scaling_factor=0.15).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    test_ds = SteganographyDataset(dataset_dir, split="test", seed=42)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    print(f"\n{'='*64}")
    print("PHASE 2B: QUANTITATIVE EVALUATION OF ADAPTIVE EMBEDDING")
    print(f"Checkpoint: {model_path}")
    print(f"Test Set Size: {len(test_ds)} held-out images")
    print(f"{'='*64}\n")

    adaptive_metrics = {"psnr": [], "ssim": [], "mse": [], "ber": []}
    uniform_metrics  = {"psnr": [], "ssim": [], "mse": [], "ber": []}

    high_suit_ber = []
    low_suit_ber  = []

    torch.manual_seed(42)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            cover = batch["image"].to(device)  # (1, 3, 256, 256)
            b, c, h, w = cover.shape
            payload = torch.randint(0, 2, (b, 1, h, w), dtype=torch.float32, device=device)

            # 1. Normal Adaptive Forward Pass
            out_adapt = model(cover, payload)
            stego_adapt = out_adapt["stego_image"]
            ext_adapt = out_adapt["extracted_payload"]
            suit_map = out_adapt["suitability_map"]  # (1, 1, H, W)

            # 2. Uniform (Non-Adaptive) Forward Pass using mean suitability as constant
            mean_suit = suit_map.mean()
            uniform_suit_map = torch.full_like(suit_map, mean_suit.item())
            stego_uniform = model.encoder(cover, payload, uniform_suit_map, scaling_factor=model.scaling_factor)
            ext_uniform = model.decoder(stego_uniform)

            c_np = cover[0].cpu().numpy()
            sa_np = stego_adapt[0].cpu().numpy()
            su_np = stego_uniform[0].cpu().numpy()

            # Adaptive metrics
            mse_a = calculate_mse(c_np, sa_np)
            psnr_a = calculate_psnr(c_np, sa_np, max_val=1.0)
            ssim_a = calculate_ssim(c_np, sa_np, max_val=1.0)
            pred_bits_a = (ext_adapt > 0.5).float()
            ber_a = (pred_bits_a != payload).float().mean().item()

            adaptive_metrics["mse"].append(mse_a)
            adaptive_metrics["psnr"].append(psnr_a)
            adaptive_metrics["ssim"].append(ssim_a)
            adaptive_metrics["ber"].append(ber_a)

            # Uniform metrics
            mse_u = calculate_mse(c_np, su_np)
            psnr_u = calculate_psnr(c_np, su_np, max_val=1.0)
            ssim_u = calculate_ssim(c_np, su_np, max_val=1.0)
            pred_bits_u = (ext_uniform > 0.5).float()
            ber_u = (pred_bits_u != payload).float().mean().item()

            uniform_metrics["mse"].append(mse_u)
            uniform_metrics["psnr"].append(psnr_u)
            uniform_metrics["ssim"].append(ssim_u)
            uniform_metrics["ber"].append(ber_u)

            # Regional BER analysis (High Suitability vs Low Suitability pixels)
            suit_thresh = suit_map.median().item()
            high_mask = (suit_map >= suit_thresh)
            low_mask  = (suit_map < suit_thresh)

            if high_mask.sum() > 0:
                high_ber_val = (pred_bits_a[high_mask] != payload[high_mask]).float().mean().item()
                high_suit_ber.append(high_ber_val)
            if low_mask.sum() > 0:
                low_ber_val = (pred_bits_a[low_mask] != payload[low_mask]).float().mean().item()
                low_suit_ber.append(low_ber_val)

            fname = os.path.basename(batch["image_path"][0])
            print(f"[{i+1}/{len(test_ds)}] {fname:<25} | Adapt PSNR: {psnr_a:.2f}dB (BER: {ber_a*100:.2f}%) | Uniform PSNR: {psnr_u:.2f}dB (BER: {ber_u*100:.2f}%)")

    results = {
        "adaptive": {
            "mean_psnr": float(np.mean(adaptive_metrics["psnr"])),
            "mean_ssim": float(np.mean(adaptive_metrics["ssim"])),
            "mean_mse": float(np.mean(adaptive_metrics["mse"])),
            "mean_ber": float(np.mean(adaptive_metrics["ber"])),
            "accuracy": float(1.0 - np.mean(adaptive_metrics["ber"]))
        },
        "uniform": {
            "mean_psnr": float(np.mean(uniform_metrics["psnr"])),
            "mean_ssim": float(np.mean(uniform_metrics["ssim"])),
            "mean_mse": float(np.mean(uniform_metrics["mse"])),
            "mean_ber": float(np.mean(uniform_metrics["ber"])),
            "accuracy": float(1.0 - np.mean(uniform_metrics["ber"]))
        },
        "regional_ber": {
            "high_suitability_ber": float(np.mean(high_suit_ber)),
            "low_suitability_ber": float(np.mean(low_suit_ber))
        }
    }

    print(f"\n{'-'*64}")
    print("PHASE 2B COMPARATIVE SUMMARY:")
    print(f"  Adaptive Embedding : PSNR = {results['adaptive']['mean_psnr']:.2f} dB | SSIM = {results['adaptive']['mean_ssim']:.4f} | BER = {results['adaptive']['mean_ber']*100:.3f}% (Acc: {results['adaptive']['accuracy']*100:.2f}%)")
    print(f"  Uniform Embedding  : PSNR = {results['uniform']['mean_psnr']:.2f} dB | SSIM = {results['uniform']['mean_ssim']:.4f} | BER = {results['uniform']['mean_ber']*100:.3f}% (Acc: {results['uniform']['accuracy']*100:.2f}%)")
    print(f"  Regional Modulation: High-Suitability BER = {results['regional_ber']['high_suitability_ber']*100:.3f}% | Low-Suitability BER = {results['regional_ber']['low_suitability_ber']*100:.3f}%")
    print(f"{'='*64}\n")

    return results


if __name__ == "__main__":
    evaluate_phase2b_adaptive_effect()
