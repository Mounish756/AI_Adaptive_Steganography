"""
CLI Inference Script for AI-Based Adaptive Region Analyzer.

Usage:
    python ai/models/predict_adaptive_map.py --image path/to/image.png

Pipeline:
1. Loads & preprocesses input cover image (256x256 RGB).
2. Loads trained AdaptiveRegionAnalyzer checkpoint from models/adaptive_analyzer/best_model.pth.
3. Computes continuous spatial suitability map (256x256x1).
4. Saves visualization artifact to results/adaptive_analyzer/.
5. Prints statistical breakdown (Min, Max, Mean, Median, Std, % Low, % Medium, % High, Inference Latency).
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import time
import torch
import numpy as np
import yaml

from ai.models.adaptive_analyzer import AdaptiveRegionAnalyzer
from ai.data.preprocessing import preprocess_pipeline
from ai.metrics.adaptive_visualization import visualize_adaptive_analyzer_comparison, calculate_map_statistics


def predict_adaptive_map(
    image_path: str,
    model_path: str = "models/adaptive_analyzer/best_model.pth",
    config_path: str = "configs/config.yaml",
    output_dir: str = "results/adaptive_analyzer"
):
    """
    Executes inference for a single cover image.

    Args:
        image_path: Path to target cover image file.
        model_path: Path to PyTorch model checkpoint.
        config_path: Path to config YAML file.
        output_dir: Path to save visualization artifacts.
    """
    if not os.path.exists(image_path):
        print(f"[!] Error: Image file not found: {image_path}")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"\n[!] ERROR: Model checkpoint not found at: {model_path}")
        print(f"[*] A trained model checkpoint is required to run inference.")
        print(f"[*] Please run the training script first:\n")
        print(f"    python ai/training/train_adaptive_analyzer.py\n")
        sys.exit(1)

    # 1. Load Config & Select Device
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device_name = config.get("training", {}).get("device", "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"Using device: {device.type.upper()}")

    # 2. Load Model Architecture & Checkpoint
    cfg_aa = config["adaptive_analyzer"]
    model = AdaptiveRegionAnalyzer(
        in_channels=cfg_aa["input_channels"],
        out_channels=cfg_aa["output_channels"],
        base_dim=cfg_aa.get("base_channels", 32)
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 3. Preprocess Image & Benchmark Inference Latency
    norm_arr, tensor_img = preprocess_pipeline(image_path, target_size=(256, 256))
    inp_tensor = tensor_img.unsqueeze(0).to(device)  # (1, 3, 256, 256)

    start_t = time.perf_counter()
    with torch.no_grad():
        pred_map_tensor = model(inp_tensor)
    inference_ms = (time.perf_counter() - start_t) * 1000.0

    pred_map_np = pred_map_tensor[0, 0].cpu().numpy()  # (256, 256)

    # 4. Generate Visualization Plot
    out_file, stats = visualize_adaptive_analyzer_comparison(
        image_path=image_path,
        model=model,
        output_dir=output_dir,
        device=device
    )

    # 5. Report Map Statistics
    print(f"\n==================================================")
    print(f"AI ADAPTIVE REGION ANALYZER INFERENCE REPORT")
    print(f"Target Image: {image_path}")
    print(f"Model Checkpoint: {model_path}")
    print(f"==================================================")
    print(f"STATISTICAL METRICS:")
    print(f"  - Minimum Suitability:   {stats['min']:.4f}")
    print(f"  - Maximum Suitability:   {stats['max']:.4f}")
    print(f"  - Mean Suitability:      {stats['mean']:.4f}")
    print(f"  - Median Suitability:    {stats['median']:.4f}")
    print(f"  - Standard Deviation:    {stats['std']:.4f}")
    print(f"--------------------------------------------------")
    print(f"SUITABILITY DISTRIBUTION:")
    print(f"  - Low Suitability [0.00 - 0.33]:    {stats['pct_low']:6.2f} %")
    print(f"  - Medium Suitability [0.33 - 0.66]: {stats['pct_medium']:6.2f} %")
    print(f"  - High Suitability [0.66 - 1.00]:   {stats['pct_high']:6.2f} %")
    print(f"--------------------------------------------------")
    print(f"PERFORMANCE:")
    print(f"  - Inference Latency:     {inference_ms:.2f} ms")
    print(f"  - Saved Map Artifact:    {out_file}")
    print(f"==================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Adaptive Embedding Suitability Map for an Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input cover image")
    parser.add_argument("--model", type=str, default="models/adaptive_analyzer/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Configuration YAML path")
    parser.add_argument("--output_dir", type=str, default="results/adaptive_analyzer", help="Output directory")

    args = parser.parse_args()
    predict_adaptive_map(
        image_path=args.image,
        model_path=args.model,
        config_path=args.config,
        output_dir=args.output_dir
    )
