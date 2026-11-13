"""
Phase 2A Training Pipeline — UNet Encoder + Decoder Networks.

Trains AdaptiveStegoModel (NeuralAdaptiveAnalyzer + UNetEncoder + UNetDecoder)
on natural RGB cover images.

Training Objective:
    Total Loss = alpha * MSE(stego, cover) + beta * BCE(extracted_payload, target_payload)

Usage (smoke test, synthetic data):
    python ai/training/train.py --smoke-test

Usage (real images subset, e.g. COCO 2017 subset):
    python ai/training/train.py --dataset-dir datasets/coco_subset --epochs 15

Usage (full training with real dataset):
    python ai/training/train.py --config configs/config.yaml

Usage (resume from checkpoint):
    python ai/training/train.py --resume checkpoints/phase2a/training_state.pth
"""

import os
import sys
import csv
import json
import time
import argparse
import random
from typing import Dict, Any, Optional

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.optim as optim
import yaml
import numpy as np

from torch.utils.data import DataLoader

from ai.models.full_model import AdaptiveStegoModel
from ai.training.losses import SteganographyLoss
from ai.training.validate import validate_epoch
from ai.training.test import evaluate_test_set
from ai.data.dataset import SteganographyDataset, get_image_files


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    cfg: Dict[str, Any],
    smoke_test: bool = False,
    dataset_dir: Optional[str] = None
) -> tuple:
    """
    Build train / validation / test DataLoaders.

    Args:
        cfg: Full parsed config dict.
        smoke_test: If True, use synthetic images.
        dataset_dir: Optional directory override.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    p2a = cfg.get("phase2a", {})
    batch_size = p2a.get("batch_size", 2)
    num_workers = p2a.get("num_workers", 0)
    seed = p2a.get("seed", 42)

    # Determine dataset root
    if dataset_dir:
        dataset_root = dataset_dir
    elif p2a.get("dataset_dir"):
        dataset_root = p2a["dataset_dir"]
    else:
        dataset_root = cfg.get("paths", {}).get("dataset_root", "datasets")

    if smoke_test:
        train_dir = cfg.get("paths", {}).get("train", "datasets/train")
        val_dir   = cfg.get("paths", {}).get("validation", "datasets/validation")
        test_dir  = cfg.get("paths", {}).get("test", "datasets/test")

        all_files = (
            get_image_files(train_dir) +
            get_image_files(val_dir) +
            get_image_files(test_dir)
        )
        if len(all_files) == 0:
            raise FileNotFoundError(
                "No images found in datasets/. "
                "Run: python scripts/generate_synthetic_data.py first."
            )

        from torch.utils.data import Dataset
        from ai.data.preprocessing import preprocess_pipeline

        class DirectFolderDataset(Dataset):
            def __init__(self, directory: str, image_size=(256, 256)):
                self.files = get_image_files(directory)
                self.image_size = image_size

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                path = self.files[idx]
                try:
                    _, tensor = preprocess_pipeline(path, target_size=self.image_size)
                    return {"image": tensor, "image_name": os.path.basename(path), "image_path": path}
                except Exception as e:
                    print(f"[!] Skipping corrupt image {path}: {e}")
                    return {"image": torch.zeros(3, 256, 256), "image_name": "error", "image_path": path}

        train_ds = DirectFolderDataset(train_dir)
        val_ds   = DirectFolderDataset(val_dir)
        test_ds  = DirectFolderDataset(test_dir)

        print(f"[SMOKE TEST] Dataset sizes -- Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        print("[SMOKE TEST] Synthetic images only. Pipeline verification.")

    else:
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(
                f"Dataset root '{dataset_root}' not found.\n"
                "Please place natural RGB images (e.g. COCO 2017 or BOSSbase) in:\n"
                f"  {os.path.abspath(dataset_root)}/\n"
                "Or run: python scripts/download_coco_subset.py"
            )

        all_imgs = get_image_files(dataset_root)
        if len(all_imgs) == 0:
            raise FileNotFoundError(
                f"No images found under '{dataset_root}'. "
                "Add natural images and retry."
            )

        train_ds = SteganographyDataset(dataset_root, split="train", seed=seed)
        val_ds   = SteganographyDataset(dataset_root, split="validation", seed=seed)
        test_ds  = SteganographyDataset(dataset_root, split="test", seed=seed)

        print(f"[Dataset: {dataset_root}] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} images")

        if len(train_ds) == 0:
            raise RuntimeError(
                f"Training split is empty in '{dataset_root}'. "
                "Add more images to the dataset directory."
            )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Train epoch
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: SteganographyLoss,
    device: torch.device,
    scaling_factor: float = 0.15
) -> Dict[str, float]:
    """
    One training epoch over all batches.

    Returns:
        Dict of averaged loss, BER, and bit accuracy metrics for the epoch.
    """
    model.train()
    running = {"total_loss": 0.0, "img_mse_loss": 0.0, "payload_bce_loss": 0.0}
    bit_correct = 0
    bit_total = 0
    n_batches = 0

    for batch in dataloader:
        cover = batch["image"].to(device)
        b, _, h, w = cover.shape

        payload_map = torch.randint(0, 2, (b, 1, h, w), dtype=torch.float32, device=device)

        optimizer.zero_grad()

        output = model(cover, payload_map)
        stego = output["stego_image"]
        extracted = output["extracted_payload"]

        losses = criterion(cover, stego, payload_map, extracted)
        losses["total_loss"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pred_bits = (extracted > 0.5).float()
        bit_correct += (pred_bits == payload_map).sum().item()
        bit_total += payload_map.numel()

        for k in running:
            running[k] += losses[k].item()
        n_batches += 1

    if n_batches == 0:
        return {k: 0.0 for k in running}

    res = {k: v / n_batches for k, v in running.items()}
    res["bit_accuracy"] = bit_correct / max(bit_total, 1)
    res["ber"] = 1.0 - res["bit_accuracy"]
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    cfg: Dict[str, Any],
    ckpt_dir: str,
    is_best: bool = False
) -> None:
    """Save training state and best model checkpoints."""
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }
    last_path = os.path.join(ckpt_dir, "training_state.pth")
    torch.save(state, last_path)

    if is_best:
        best_path = os.path.join(ckpt_dir, "best_model.pth")
        torch.save(state, best_path)
        enc_path = os.path.join(ckpt_dir, "best_encoder.pth")
        dec_path = os.path.join(ckpt_dir, "best_decoder.pth")
        torch.save(model.encoder.state_dict(), enc_path)
        torch.save(model.decoder.state_dict(), dec_path)
        print(f"  [+] Best checkpoint saved -> {best_path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple:
    """Safely load a checkpoint for training resumption."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=device, weights_only=False)
    required_keys = {"epoch", "model_state_dict", "optimizer_state_dict"}
    missing = required_keys - set(state.keys())
    if missing:
        raise ValueError(f"Checkpoint missing required keys: {missing}")

    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    start_epoch = state["epoch"] + 1
    best_val_loss = state.get("best_val_loss", float("inf"))
    print(f"[+] Resumed from checkpoint: epoch {state['epoch']}, best_val_loss={best_val_loss:.6f}")
    return start_epoch, best_val_loss


# ──────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_training_pipeline(
    cfg: Dict[str, Any],
    smoke_test: bool = False,
    resume_path: Optional[str] = None,
    dataset_dir: Optional[str] = None
) -> torch.nn.Module:
    """
    Full Phase 2A / 2A.1 training pipeline.
    """
    p2a = cfg["phase2a"]
    seed = p2a.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    ckpt_dir   = p2a.get("checkpoint_dir", "checkpoints/phase2a")
    output_dir = p2a.get("output_dir", "outputs/phase2a")
    os.makedirs(ckpt_dir,   exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, smoke_test=smoke_test, dataset_dir=dataset_dir
    )

    # Model
    scaling_factor = p2a.get("scaling_factor", 0.15)
    model = AdaptiveStegoModel(
        base_dim=p2a.get("base_dim", 32),
        scaling_factor=scaling_factor
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] AdaptiveStegoModel -- {total_params:,} trainable parameters (analyzer frozen)")

    # Loss & optimizer
    alpha = p2a.get("loss", {}).get("alpha", 1.0)
    beta = p2a.get("loss", {}).get("beta", 10.0)
    criterion = SteganographyLoss(alpha=alpha, beta=beta)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=p2a.get("learning_rate", 0.001))

    total_epochs = p2a.get("epochs", 15)

    start_epoch = 1
    best_val_loss = float("inf")
    resume = resume_path or p2a.get("resume_checkpoint")
    if resume:
        start_epoch, best_val_loss = load_checkpoint(resume, model, optimizer, device)

    # CSV history log
    history_path = os.path.join(output_dir, "training_history.csv")
    write_header = not os.path.exists(history_path) or start_epoch == 1
    csv_file = open(history_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "train_total_loss", "train_img_mse", "train_payload_bce", "train_ber", "train_bit_acc",
        "val_total_loss", "val_img_mse", "val_payload_bce", "val_ber", "val_bit_acc", "val_psnr", "epoch_time_s"
    ])
    if write_header:
        writer.writeheader()

    print(f"\n{'='*64}")
    mode = "SMOKE TEST" if smoke_test else "REAL IMAGES VALIDATION"
    print(f"PHASE 2A - {mode}")
    print(f"Epochs: {start_epoch}->{total_epochs} | Batch: {p2a['batch_size']} | LR: {p2a['learning_rate']}")
    print(f"Loss weights - alpha(img)={alpha}  beta(payload)={beta} | Scale={scaling_factor}")
    print(f"{'='*64}")

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.perf_counter()

        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, scaling_factor
        )
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        elapsed = time.perf_counter() - t0

        is_best = val_metrics["total_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["total_loss"]

        save_checkpoint(model, optimizer, epoch, best_val_loss, cfg, ckpt_dir, is_best=is_best)

        row = {
            "epoch": epoch,
            "train_total_loss": round(train_metrics["total_loss"], 6),
            "train_img_mse": round(train_metrics["img_mse_loss"], 6),
            "train_payload_bce": round(train_metrics["payload_bce_loss"], 6),
            "train_ber": round(train_metrics.get("ber", 0.5), 6),
            "train_bit_acc": round(train_metrics.get("bit_accuracy", 0.5), 6),
            "val_total_loss": round(val_metrics["total_loss"], 6),
            "val_img_mse": round(val_metrics["img_mse_loss"], 6),
            "val_payload_bce": round(val_metrics["payload_bce_loss"], 6),
            "val_ber": round(val_metrics.get("ber", 0.5), 6),
            "val_bit_acc": round(val_metrics.get("bit_accuracy", 0.5), 6),
            "val_psnr": round(val_metrics.get("psnr_db", 0.0), 2),
            "epoch_time_s": round(elapsed, 2),
        }
        writer.writerow(row)
        csv_file.flush()

        best_marker = " <- best" if is_best else ""
        tr_ber = train_metrics.get('ber', 0.5) * 100.0
        val_ber = val_metrics.get('ber', 0.5) * 100.0
        val_psnr = val_metrics.get('psnr_db', 0.0)

        print(
            f"Epoch {epoch:03d}/{total_epochs:03d} | "
            f"Train {train_metrics['total_loss']:.4f} (BER={tr_ber:.1f}%) | "
            f"Val {val_metrics['total_loss']:.4f} (BER={val_ber:.1f}%, PSNR={val_psnr:.1f}dB) | "
            f"{elapsed:.1f}s{best_marker}"
        )

    csv_file.close()
    print(f"\n[+] Training complete. History -> {history_path}")

    # ── Test set evaluation ────────────────────────────────────────────────────
    print("\n[*] Evaluating on held-out test set using best checkpoint...")
    best_ckpt = os.path.join(ckpt_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])

    test_metrics = evaluate_test_set(model, test_loader, device)

    print(f"\n{'='*64}")
    print("HELD-OUT TEST SET EVALUATION RESULTS")
    print(f"  PSNR         : {test_metrics['avg_psnr']:.2f} dB")
    print(f"  SSIM         : {test_metrics['avg_ssim']:.4f}")
    print(f"  MSE          : {test_metrics['avg_mse']:.6f}")
    print(f"  BER          : {test_metrics['avg_ber']:.4f}  ({test_metrics['avg_ber']*100:.2f}%)")
    print(f"  Bit Accuracy : {test_metrics.get('avg_bit_acc', 1.0 - test_metrics['avg_ber']):.4f}  ({(test_metrics.get('avg_bit_acc', 1.0 - test_metrics['avg_ber']))*100:.2f}%)")
    print(f"{'='*64}\n")

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "smoke_test": smoke_test,
            "dataset_dir": dataset_dir or p2a.get("dataset_dir"),
            "best_val_loss": best_val_loss,
            "test_metrics": test_metrics,
            "config": {
                "epochs": total_epochs,
                "batch_size": p2a.get("batch_size"),
                "learning_rate": p2a.get("learning_rate"),
                "loss_alpha": alpha,
                "loss_beta": beta,
                "scaling_factor": scaling_factor
            }
        }, f, indent=2)
    print(f"[+] Metrics JSON -> {metrics_path}")

    _save_sample_images(model, val_loader, device, output_dir)
    return model


def _save_sample_images(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: str
) -> None:
    """Save cover, stego, and residual image artifacts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            cover = batch["image"][:1].to(device)
            payload = torch.randint(0, 2, (1, 1, 256, 256), dtype=torch.float32, device=device)
            out = model(cover, payload)
            stego = out["stego_image"]
            diff = (stego - cover).abs()

        def to_hwc(t):
            return t[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(to_hwc(cover)); axes[0].set_title("Original Cover Image"); axes[0].axis("off")
        axes[1].imshow(to_hwc(stego)); axes[1].set_title("Stego Image (Embedded)"); axes[1].axis("off")
        im_diff = axes[2].imshow(to_hwc(diff * 10.0)); axes[2].set_title("Residual Perturbation (x10)"); axes[2].axis("off")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "cover_vs_stego.png"), dpi=150)
        plt.close()

        Image.fromarray((to_hwc(cover) * 255).astype(np.uint8)).save(os.path.join(output_dir, "sample_cover.png"))
        Image.fromarray((to_hwc(stego) * 255).astype(np.uint8)).save(os.path.join(output_dir, "sample_stego.png"))
        print(f"[+] Sample image comparisons saved in {output_dir}/")
    except Exception as e:
        print(f"[!] Warning: Could not save sample comparison images: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2A / 2A.1: Train UNet Encoder + Decoder on Real Natural Images"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run smoke test using synthetic dataset"
    )
    parser.add_argument(
        "--dataset-dir", default=None,
        help="Path to dataset directory (e.g. datasets/coco_subset)"
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to checkpoint to resume training"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--beta", type=float, default=None,
        help="Override secret recovery loss weight"
    )
    parser.add_argument(
        "--scaling-factor", type=float, default=None,
        help="Override residual scaling factor"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["phase2a"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["phase2a"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["phase2a"]["learning_rate"] = args.lr
    if args.beta is not None:
        cfg["phase2a"]["loss"]["beta"] = args.beta
    if args.scaling_factor is not None:
        cfg["phase2a"]["scaling_factor"] = args.scaling_factor
    if args.dataset_dir is not None:
        cfg["phase2a"]["dataset_dir"] = args.dataset_dir

    run_training_pipeline(
        cfg,
        smoke_test=args.smoke_test,
        resume_path=args.resume,
        dataset_dir=args.dataset_dir
    )


if __name__ == "__main__":
    main()
