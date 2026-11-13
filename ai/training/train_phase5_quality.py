"""
Phase 5 — Visual Quality Improvement Fine-Tuning.

Fine-tunes the existing Phase 2B checkpoint using PerceptualSteganographyLoss
(MSE + SSIM + edge preservation + smooth-region penalty + payload BCE).

Key design decisions:
  - Loads Phase 2B checkpoint as starting point (NOT from scratch).
  - Uses a low learning rate (5e-5) to avoid disrupting the learned bit recovery.
  - Saves new checkpoints to checkpoints/phase5/ — does NOT overwrite Phase 2A/2B.
  - The AdaptiveRegionAnalyzer remains FROZEN throughout.
  - Fine-tunes for 20 epochs with early stopping based on validation SSIM+PSNR.

Usage:
    python ai/training/train_phase5_quality.py
    python ai/training/train_phase5_quality.py --epochs 30 --lr 3e-5
    python ai/training/train_phase5_quality.py --scaling-factor 0.12
"""

import os
import sys
import csv
import json
import time
import argparse
import random
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from ai.models.full_model import AdaptiveStegoModel
from ai.training.losses import PerceptualSteganographyLoss
from ai.data.dataset import SteganographyDataset
from ai.metrics.psnr import calculate_psnr
from ai.metrics.ssim import calculate_ssim
from ai.metrics.mse import calculate_mse


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(dataset_dir: str, batch_size: int = 2, seed: int = 42):
    train_ds = SteganographyDataset(dataset_dir, split="train",      seed=seed)
    val_ds   = SteganographyDataset(dataset_dir, split="validation",  seed=seed)
    test_ds  = SteganographyDataset(dataset_dir, split="test",        seed=seed)
    print(f"[Dataset] Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )


def load_model(
    ckpt_path: str,
    device: torch.device,
    scaling_factor: float = 0.15,
) -> AdaptiveStegoModel:
    model = AdaptiveStegoModel(
        base_dim=32,
        analyzer_checkpoint="models/adaptive_analyzer/best_model.pth",
        freeze_analyzer=True,
        scaling_factor=scaling_factor,
    ).to(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(sd)
    print(f"[+] Loaded Phase 2B checkpoint: {ckpt_path}")
    return model


def train_epoch(
    model: AdaptiveStegoModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PerceptualSteganographyLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    # Keep analyzer frozen explicitly
    model.adaptive_analyzer.eval()

    running = {k: 0.0 for k in [
        "total_loss", "img_mse_loss", "ssim_loss",
        "edge_loss", "smooth_region_loss", "payload_bce_loss"
    ]}
    bit_correct = bit_total = n = 0

    for batch in loader:
        cover = batch["image"].to(device)
        b, _, h, w = cover.shape
        payload = torch.randint(0, 2, (b, 1, h, w), dtype=torch.float32, device=device)

        optimizer.zero_grad()
        out = model(cover, payload)
        stego     = out["stego_image"]
        extracted = out["extracted_payload"]

        losses = criterion(cover, stego, payload, extracted)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in running:
            running[k] += losses[k].item()
        pred = (extracted > 0.5).float()
        bit_correct += (pred == payload).sum().item()
        bit_total   += payload.numel()
        n += 1

    if n == 0:
        return {k: 0.0 for k in running}

    res = {k: v / n for k, v in running.items()}
    res["bit_accuracy"] = bit_correct / max(bit_total, 1)
    res["ber"] = 1.0 - res["bit_accuracy"]
    return res


def validate_epoch(
    model: AdaptiveStegoModel,
    loader: DataLoader,
    criterion: PerceptualSteganographyLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running = {k: 0.0 for k in [
        "total_loss", "img_mse_loss", "ssim_loss",
        "edge_loss", "smooth_region_loss", "payload_bce_loss"
    ]}
    psnr_sum = ssim_sum = mse_sum = 0.0
    bit_correct = bit_total = n_imgs = n = 0

    with torch.no_grad():
        for batch in loader:
            cover = batch["image"].to(device)
            b, _, h, w = cover.shape
            payload = torch.randint(0, 2, (b, 1, h, w), dtype=torch.float32, device=device)

            out = model(cover, payload)
            stego     = out["stego_image"]
            extracted = out["extracted_payload"]

            losses = criterion(cover, stego, payload, extracted)
            for k in running:
                running[k] += losses[k].item()

            for i in range(b):
                c_np = cover[i].cpu().numpy()
                s_np = stego[i].cpu().numpy()
                psnr_sum += calculate_psnr(c_np, s_np, max_val=1.0)
                ssim_sum += calculate_ssim(c_np, s_np, max_val=1.0)
                mse_sum  += calculate_mse(c_np, s_np)
                n_imgs   += 1

            pred = (extracted > 0.5).float()
            bit_correct += (pred == payload).sum().item()
            bit_total   += payload.numel()
            n += 1

    if n == 0:
        return {}

    res = {k: v / n for k, v in running.items()}
    res["psnr_db"]     = psnr_sum  / max(n_imgs, 1)
    res["ssim"]        = ssim_sum  / max(n_imgs, 1)
    res["mse"]         = mse_sum   / max(n_imgs, 1)
    res["bit_accuracy"] = bit_correct / max(bit_total, 1)
    res["ber"]          = 1.0 - res["bit_accuracy"]
    return res


def save_checkpoint(
    model: AdaptiveStegoModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    ckpt_dir: str,
    is_best: bool = False,
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "best_score": best_score,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "phase": "phase5",
    }
    torch.save(state, os.path.join(ckpt_dir, "training_state.pth"))
    if is_best:
        best_path = os.path.join(ckpt_dir, "best_model.pth")
        torch.save(state, best_path)
        torch.save(model.encoder.state_dict(), os.path.join(ckpt_dir, "best_encoder.pth"))
        torch.save(model.decoder.state_dict(), os.path.join(ckpt_dir, "best_decoder.pth"))
        print(f"  [+] Best checkpoint saved -> {best_path}")


def save_sample_images(
    model: AdaptiveStegoModel,
    loader: DataLoader,
    device: torch.device,
    out_dir: str,
    label: str = "phase5",
) -> None:
    """Save cover/stego/diff comparison image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            cover   = batch["image"][:1].to(device)
            payload = torch.randint(0, 2, (1, 1, 256, 256), dtype=torch.float32, device=device)
            out     = model(cover, payload)
            stego   = out["stego_image"]
            diff    = (stego - cover).abs()

        def to_hwc(t):
            return t[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(to_hwc(cover));        axes[0].set_title("Original Cover");    axes[0].axis("off")
        axes[1].imshow(to_hwc(stego));        axes[1].set_title("Stego Image");        axes[1].axis("off")
        axes[2].imshow(to_hwc(diff * 10.0)); axes[2].set_title("Residual x10");       axes[2].axis("off")
        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"{label}_comparison.png"), dpi=150)
        plt.close()
        print(f"  [+] Sample comparison saved -> {out_dir}/{label}_comparison.png")
    except Exception as e:
        print(f"  [!] Could not save sample images: {e}")


def evaluate_test_set(
    model: AdaptiveStegoModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Full test-set evaluation (PSNR, SSIM, MSE, BER)."""
    model.eval()
    psnr_sum = ssim_sum = mse_sum = 0.0
    bit_correct = bit_total = n = 0

    with torch.no_grad():
        for batch in loader:
            cover = batch["image"].to(device)
            b, _, h, w = cover.shape
            payload = torch.randint(0, 2, (b, 1, h, w), dtype=torch.float32, device=device)
            out = model(cover, payload)
            stego     = out["stego_image"]
            extracted = out["extracted_payload"]

            for i in range(b):
                c_np = cover[i].cpu().numpy()
                s_np = stego[i].cpu().numpy()
                psnr_sum += calculate_psnr(c_np, s_np, max_val=1.0)
                ssim_sum += calculate_ssim(c_np, s_np, max_val=1.0)
                mse_sum  += calculate_mse(c_np, s_np)
                n += 1

            pred = (extracted > 0.5).float()
            bit_correct += (pred == payload).sum().item()
            bit_total   += payload.numel()

    avg_psnr = psnr_sum / max(n, 1)
    avg_ssim = ssim_sum / max(n, 1)
    avg_mse  = mse_sum  / max(n, 1)
    avg_ber  = 1.0 - bit_correct / max(bit_total, 1)
    return {
        "avg_psnr": avg_psnr, "avg_ssim": avg_ssim,
        "avg_mse": avg_mse, "avg_ber": avg_ber,
        "avg_bit_acc": 1.0 - avg_ber,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Visual Quality Fine-Tuning")
    parser.add_argument("--source-ckpt",   default="checkpoints/phase2b/phase2b_model.pth")
    parser.add_argument("--dataset-dir",   default="datasets/coco_subset")
    parser.add_argument("--ckpt-dir",      default="checkpoints/phase5")
    parser.add_argument("--output-dir",    default="outputs/phase5/training")
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch-size",    type=int,   default=2)
    parser.add_argument("--lr",            type=float, default=5e-5)
    parser.add_argument("--scaling-factor",type=float, default=0.15)
    # Loss weights
    parser.add_argument("--alpha",         type=float, default=1.0,  help="MSE weight")
    parser.add_argument("--beta",          type=float, default=10.0, help="BCE weight")
    parser.add_argument("--ssim-weight",   type=float, default=2.0,  help="SSIM loss weight")
    parser.add_argument("--edge-weight",   type=float, default=1.0,  help="Edge preservation weight")
    parser.add_argument("--smooth-weight", type=float, default=1.0,  help="Smooth-region penalty weight")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("PHASE 5 — Visual Quality Fine-Tuning with PerceptualSteganographyLoss")
    print(f"{'='*70}")
    print(f"  Source checkpoint : {args.source_ckpt}")
    print(f"  Device            : {device}")
    print(f"  Epochs            : {args.epochs} | LR={args.lr} | Batch={args.batch_size}")
    print(f"  Scaling factor    : {args.scaling_factor}")
    print(f"  Loss weights      : MSE={args.alpha}  BCE={args.beta}  "
          f"SSIM={args.ssim_weight}  Edge={args.edge_weight}  Smooth={args.smooth_weight}")
    print(f"{'='*70}\n")

    train_loader, val_loader, test_loader = build_dataloaders(
        args.dataset_dir, args.batch_size
    )

    model = load_model(args.source_ckpt, device, args.scaling_factor)

    criterion = PerceptualSteganographyLoss(
        alpha=args.alpha,
        beta=args.beta,
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
        smooth_weight=args.smooth_weight,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # CSV history
    history_path = os.path.join(args.output_dir, "training_history.csv")
    csv_file = open(history_path, "w", newline="", encoding="utf-8")
    fieldnames = [
        "epoch", "train_loss", "train_mse", "train_ssim_loss", "train_edge_loss",
        "train_smooth_loss", "train_bce", "train_ber",
        "val_loss", "val_psnr", "val_ssim", "val_mse", "val_ber", "epoch_s"
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Score = PSNR + 10*SSIM — maximise both
    best_score = -float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr = train_epoch(model, train_loader, optimizer, criterion, device)
        vl = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        val_score = vl.get("psnr_db", 0) + 10.0 * vl.get("ssim", 0)
        is_best = val_score > best_score
        if is_best:
            best_score = val_score

        save_checkpoint(model, optimizer, epoch, best_score, args.ckpt_dir, is_best)

        row = {
            "epoch": epoch,
            "train_loss":   round(tr.get("total_loss", 0), 6),
            "train_mse":    round(tr.get("img_mse_loss", 0), 6),
            "train_ssim_loss": round(tr.get("ssim_loss", 0), 6),
            "train_edge_loss": round(tr.get("edge_loss", 0), 6),
            "train_smooth_loss": round(tr.get("smooth_region_loss", 0), 6),
            "train_bce":    round(tr.get("payload_bce_loss", 0), 6),
            "train_ber":    round(tr.get("ber", 0.5), 6),
            "val_loss":     round(vl.get("total_loss", 0), 6),
            "val_psnr":     round(vl.get("psnr_db", 0), 4),
            "val_ssim":     round(vl.get("ssim", 0), 6),
            "val_mse":      round(vl.get("mse", 0), 8),
            "val_ber":      round(vl.get("ber", 0.5), 6),
            "epoch_s":      round(elapsed, 2),
        }
        writer.writerow(row)
        csv_file.flush()

        best_tag = " <- best" if is_best else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Loss={tr['total_loss']:.4f} BER={tr['ber']*100:.1f}% | "
            f"Val PSNR={vl.get('psnr_db',0):.2f}dB "
            f"SSIM={vl.get('ssim',0):.4f} "
            f"BER={vl.get('ber',0.5)*100:.1f}% | "
            f"{elapsed:.1f}s{best_tag}"
        )

    csv_file.close()

    # Load best checkpoint for final evaluation
    best_path = os.path.join(args.ckpt_dir, "best_model.pth")
    if os.path.exists(best_path):
        s = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(s["model_state_dict"])

    print("\n[*] Final test-set evaluation (best checkpoint)...")
    test_m = evaluate_test_set(model, test_loader, device)

    print(f"\n{'='*70}")
    print("PHASE 5 TEST SET RESULTS")
    print(f"  PSNR         : {test_m['avg_psnr']:.4f} dB")
    print(f"  SSIM         : {test_m['avg_ssim']:.6f}")
    print(f"  MSE          : {test_m['avg_mse']:.8f}")
    print(f"  BER          : {test_m['avg_ber']:.6f}  ({test_m['avg_ber']*100:.2f}%)")
    print(f"  Bit Accuracy : {test_m['avg_bit_acc']:.6f}  ({test_m['avg_bit_acc']*100:.2f}%)")
    print(f"{'='*70}")

    metrics_path = os.path.join(args.output_dir, "phase5_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "source_checkpoint": args.source_ckpt,
            "phase5_checkpoint": best_path,
            "scaling_factor": args.scaling_factor,
            "loss_weights": {
                "alpha": args.alpha, "beta": args.beta,
                "ssim_weight": args.ssim_weight,
                "edge_weight": args.edge_weight,
                "smooth_weight": args.smooth_weight,
            },
            "test_metrics": test_m,
            "training": {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size},
        }, f, indent=2)
    print(f"\n[+] Metrics -> {metrics_path}")

    save_sample_images(model, val_loader, device, args.output_dir, label="phase5_best")


if __name__ == "__main__":
    main()
