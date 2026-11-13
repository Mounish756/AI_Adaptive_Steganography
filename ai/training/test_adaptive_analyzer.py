"""
Test Evaluation Script for Adaptive Region Analyzer.

Evaluates model exclusively on the test dataset split, reporting:
- Test Loss (Map Reconstruction MSE + TV Smoothness)
- Map Error & Similarity (MSE, SSIM between AI prediction and reference map)
- Per-image and batch inference latency (ms)
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
from typing import Dict, Any
import torch
import numpy as np
import yaml

from ai.models.adaptive_analyzer import AdaptiveRegionAnalyzer
from ai.data.dataset import SteganographyDataset, get_image_files
from ai.data.adaptive_features import generate_reference_suitability_map
from ai.data.preprocessing import preprocess_pipeline, to_tensor
from ai.training.losses import AdaptiveAnalyzerLoss
from ai.metrics.mse import calculate_mse
from ai.metrics.ssim import calculate_ssim


def evaluate_test_adaptive_analyzer(
    model_path: str = "models/adaptive_analyzer/best_model.pth",
    config_path: str = "configs/config.yaml"
) -> Dict[str, Any]:
    """
    Evaluates AdaptiveRegionAnalyzer model performance on test split images.

    Loads images directly from the 'test' subdirectory of dataset_root,
    bypassing the hash-based split filter so all images in that folder are used.

    Args:
        model_path: Path to saved PyTorch checkpoint file.
        config_path: Path to YAML config file.

    Returns:
        Dictionary of empirical evaluation metrics.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device_name = config.get("training", {}).get("device", "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    model = AdaptiveRegionAnalyzer(
        in_channels=config["adaptive_analyzer"]["input_channels"],
        out_channels=config["adaptive_analyzer"]["output_channels"],
        base_dim=config["adaptive_analyzer"].get("base_channels", 32)
    ).to(device)

    if not os.path.exists(model_path):
        print(f"\n[!] WARNING: Model checkpoint file '{model_path}' does not exist!")
        print(f"[*] Training is required before evaluating the test set. Run: python ai/training/train_adaptive_analyzer.py\n")
        return {
            "status": "UNTRAINED",
            "message": f"Checkpoint {model_path} not found. Please train model first."
        }

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Load test images directly from datasets/test/ directory (avoids hash-filter issue)
    dataset_root = config["paths"]["dataset_root"]
    test_dir = os.path.join(dataset_root, "test")

    if not os.path.exists(test_dir) or len(get_image_files(test_dir)) == 0:
        print(f"[!] Warning: Test directory '{test_dir}' is empty. Generating synthetic samples...")
        from scripts.generate_synthetic_data import generate_dataset_samples
        generate_dataset_samples(dataset_root=dataset_root)

    test_image_files = get_image_files(test_dir)

    if len(test_image_files) == 0:
        print("[!] No test images available even after generation. Aborting.")
        return {"status": "NO_DATA", "total_test_samples": 0}

    criterion = AdaptiveAnalyzerLoss(
        map_weight=config["adaptive_analyzer"]["loss"]["map_weight"],
        smoothness_weight=config["adaptive_analyzer"]["loss"]["smoothness_weight"]
    )

    running_loss = 0.0
    mse_list = []
    ssim_list = []
    inference_times_ms = []

    print(f"\n==================================================")
    print(f"ADAPTIVE REGION ANALYZER TEST SET EVALUATION")
    print(f"Model Checkpoint: {model_path}")
    print(f"Evaluation Device: {device}")
    print(f"Test Images Found: {len(test_image_files)}")
    print(f"==================================================")

    with torch.no_grad():
        for i, img_path in enumerate(test_image_files):
            img_name = os.path.basename(img_path)
            img_np, img_tensor = preprocess_pipeline(img_path)
            img_tensor = img_tensor.unsqueeze(0).to(device)   # (1, 3, 256, 256)

            img_np_hwc = img_np                                # (H, W, 3) float32 [0,1]
            ref_map_np = generate_reference_suitability_map(img_np_hwc)  # (H, W, 1)
            ref_map_tensor = to_tensor(ref_map_np).unsqueeze(0).to(device)  # (1, 1, H, W)

            # Benchmark inference latency
            start_t = time.perf_counter()
            pred_map = model(img_tensor)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            loss = criterion(pred_map, ref_map_tensor)["total_loss"].item()
            running_loss += loss

            p_np = pred_map[0, 0].cpu().numpy()       # (H, W)
            r_np = ref_map_np[:, :, 0]                # (H, W)

            mse_val = calculate_mse(p_np, r_np)
            ssim_val = calculate_ssim(p_np, r_np, max_val=1.0)

            mse_list.append(mse_val)
            ssim_list.append(ssim_val)
            inference_times_ms.append(elapsed_ms)

            print(f"[{i+1}/{len(test_image_files)}] {img_name:<30} | Loss: {loss:.6f} | MSE: {mse_val:.6f} | SSIM: {ssim_val:.4f} | Latency: {elapsed_ms:.2f} ms")

    n = len(test_image_files)
    results = {
        "status": "SUCCESS",
        "total_test_samples": n,
        "test_loss": float(running_loss / n),
        "mean_mse": float(np.mean(mse_list)),
        "mean_ssim": float(np.mean(ssim_list)),
        "mean_inference_time_ms": float(np.mean(inference_times_ms))
    }

    print(f"--------------------------------------------------")
    print(f"SUMMARY RESULTS:")
    print(f"  - Total Test Images: {results['total_test_samples']}")
    print(f"  - Average Test Loss: {results['test_loss']:.6f}")
    print(f"  - Mean Squared Error (MSE): {results['mean_mse']:.6f}")
    print(f"  - Structural Similarity (SSIM): {results['mean_ssim']:.4f}")
    print(f"  - Average Inference Latency: {results['mean_inference_time_ms']:.2f} ms")
    print(f"==================================================\n")

    return results


if __name__ == "__main__":
    evaluate_test_adaptive_analyzer()
