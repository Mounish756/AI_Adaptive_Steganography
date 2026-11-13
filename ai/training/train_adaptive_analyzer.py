"""
Multi-Epoch Training Pipeline for AI-Based Adaptive Region Analyzer.

Trains AdaptiveRegionAnalyzer PyTorch neural network to predict continuous spatial suitability maps
approximating reference feature representations.

Features:
- Reproducibility via seed=42
- Auto-device selection (CUDA / CPU)
- Loss logging: Map Reconstruction MSE + Total Variation Smoothness
- Checkpoint saving: models/adaptive_analyzer/best_model.pth and last_model.pth
- Experiment config report export: results/adaptive_analyzer/experiment_config.json
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import time
from typing import Dict, Any
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import numpy as np

from ai.models.adaptive_analyzer import AdaptiveRegionAnalyzer
from ai.data.dataset import SteganographyDataset
from ai.data.adaptive_features import generate_reference_suitability_map
from ai.data.preprocessing import to_tensor
from ai.training.losses import AdaptiveAnalyzerLoss
from ai.training.validate_adaptive_analyzer import validate_adaptive_analyzer_epoch


def set_seed(seed: int = 42):
    """Sets random seed across PyTorch, NumPy, and standard library for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_adaptive_analyzer_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: AdaptiveAnalyzerLoss,
    device: torch.device
) -> Dict[str, float]:
    """Executes single training epoch across train_loader batches."""
    model.train()
    running_total_loss = 0.0
    running_map_loss = 0.0
    running_tv_loss = 0.0
    total_batches = 0

    for batch in train_loader:
        images = batch["image"].to(device)  # (B, 3, 256, 256)
        b = images.shape[0]

        # Generate reference suitability maps for current batch
        ref_maps_list = []
        for i in range(b):
            img_np = images[i].cpu().numpy().transpose(1, 2, 0)
            ref_map_np = generate_reference_suitability_map(img_np)
            ref_map_tensor = to_tensor(ref_map_np)
            ref_maps_list.append(ref_map_tensor)

        ref_maps = torch.stack(ref_maps_list, dim=0).to(device)

        optimizer.zero_grad()

        pred_maps = model(images)
        losses = criterion(pred_maps, ref_maps)
        total_loss = losses["total_loss"]

        total_loss.backward()
        optimizer.step()

        running_total_loss += total_loss.item()
        running_map_loss += losses["map_reconstruction_loss"].item()
        running_tv_loss += losses["smoothness_loss"].item()
        total_batches += 1

    if total_batches == 0:
        return {"total_loss": 0.0, "map_reconstruction_loss": 0.0, "smoothness_loss": 0.0}

    return {
        "total_loss": running_total_loss / total_batches,
        "map_reconstruction_loss": running_map_loss / total_batches,
        "smoothness_loss": running_tv_loss / total_batches
    }


def run_training_pipeline(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Executes complete multi-epoch training and validation pipeline.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary of training results and metrics.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    # 1. Device Selection & Output Logging
    req_device = config.get("training", {}).get("device", "cpu")
    if req_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    cfg_aa = config["adaptive_analyzer"]
    dataset_root = config["paths"]["dataset_root"]
    checkpoint_dir = config["paths"]["adaptive_analyzer_models"]
    results_dir = config["paths"]["adaptive_analyzer_results"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 2. Check & Prepare Dataset
    train_ds = SteganographyDataset(data_dir=dataset_root, split="train", seed=seed)
    val_ds = SteganographyDataset(data_dir=dataset_root, split="validation", seed=seed)

    if len(train_ds) == 0:
        print(f"[!] Warning: No images found in dataset split. Generating synthetic sample images...")
        from scripts.generate_synthetic_data import generate_dataset_samples
        generate_dataset_samples(dataset_root=dataset_root, samples_per_split=4)
        train_ds = SteganographyDataset(data_dir=dataset_root, split="train", seed=seed)
        val_ds = SteganographyDataset(data_dir=dataset_root, split="validation", seed=seed)

    batch_size = cfg_aa["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Model & Loss Initialization
    model = AdaptiveRegionAnalyzer(
        in_channels=cfg_aa["input_channels"],
        out_channels=cfg_aa["output_channels"],
        base_dim=cfg_aa.get("base_channels", 32)
    ).to(device)

    param_count = model.count_parameters()
    print(f"[+] Initialized AdaptiveRegionAnalyzer Model:")
    print(f"    - Parameter Count: {param_count:,}")
    print(f"    - Input Shape: [B, {cfg_aa['input_channels']}, {cfg_aa['image_size']}, {cfg_aa['image_size']}]")
    print(f"    - Output Shape: [B, {cfg_aa['output_channels']}, {cfg_aa['image_size']}, {cfg_aa['image_size']}]")

    optimizer = optim.Adam(model.parameters(), lr=cfg_aa["learning_rate"])
    criterion = AdaptiveAnalyzerLoss(
        map_weight=cfg_aa["loss"]["map_weight"],
        smoothness_weight=cfg_aa["loss"]["smoothness_weight"]
    )

    epochs = cfg_aa["epochs"]
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_ssim": []}

    print(f"\n==================================================")
    print(f"STARTING ADAPTIVE REGION ANALYZER TRAINING")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {cfg_aa['learning_rate']}")
    print(f"==================================================")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t_metrics = train_adaptive_analyzer_epoch(model, train_loader, optimizer, criterion, device)
        v_metrics = validate_adaptive_analyzer_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(t_metrics["total_loss"])
        history["val_loss"].append(v_metrics["total_loss"])
        history["val_ssim"].append(v_metrics["avg_ssim"])

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {t_metrics['total_loss']:.6f} | "
            f"Val Loss: {v_metrics['total_loss']:.6f} | "
            f"Val SSIM: {v_metrics['avg_ssim']:.4f}"
        )

        # Checkpoint Management
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": t_metrics["total_loss"],
            "val_loss": v_metrics["total_loss"],
            "val_ssim": v_metrics["avg_ssim"],
            "config": config
        }

        # Save last model checkpoint
        last_ckpt_path = os.path.join(checkpoint_dir, "last_model.pth")
        torch.save(checkpoint_data, last_ckpt_path)

        # Save best model checkpoint
        if v_metrics["total_loss"] < best_val_loss:
            best_val_loss = v_metrics["total_loss"]
            best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(checkpoint_data, best_ckpt_path)
            print(f"  [+] Saved new best model checkpoint: {best_ckpt_path} (Val Loss: {best_val_loss:.6f})")

    elapsed_sec = time.time() - start_time
    print(f"\n[+] Training Complete in {elapsed_sec:.2f} seconds.")

    # 4. Export Reproducibility Experiment Config Report
    experiment_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "device": str(device),
        "parameter_count": param_count,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": cfg_aa["learning_rate"],
        "dataset_samples": {
            "train": len(train_ds),
            "validation": len(val_ds)
        },
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": best_val_loss,
        "final_val_ssim": history["val_ssim"][-1]
    }

    report_path = os.path.join(results_dir, "experiment_config.json")
    with open(report_path, "w") as f:
        json.dump(experiment_report, f, indent=4)
    print(f"[+] Saved experiment configuration report: {report_path}")

    return experiment_report


if __name__ == "__main__":
    run_training_pipeline()
