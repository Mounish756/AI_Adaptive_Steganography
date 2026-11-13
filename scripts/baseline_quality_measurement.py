"""
Baseline Visual Quality Measurement Script (Phase 5 — Pre-improvement).

Measures PSNR, SSIM, MSE on all 5 held-out test images at multiple payload sizes.
Saves original/stego pairs and difference images to outputs/phase5/baseline/.
Does NOT modify any model or checkpoint.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image

from ai.models.full_model import AdaptiveStegoModel
from ai.data.dataset import SteganographyDataset
from ai.metrics.psnr import calculate_psnr
from ai.metrics.ssim import calculate_ssim
from ai.metrics.mse import calculate_mse

OUTDIR = "outputs/phase5/baseline"
os.makedirs(OUTDIR, exist_ok=True)

PAYLOAD_SIZES = [64, 512, 2048, 7083]
PASSWORD = "phase5_baseline_key!"
DEVICE = torch.device("cpu")


def load_model():
    model = AdaptiveStegoModel(
        base_dim=32,
        analyzer_checkpoint="models/adaptive_analyzer/best_model.pth",
        freeze_analyzer=True,
        scaling_factor=0.15,
    ).to(DEVICE)
    ckpt = "checkpoints/phase2b/phase2b_model.pth"
    if not os.path.exists(ckpt):
        ckpt = "checkpoints/phase2a/best_model.pth"
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(
        state["model_state_dict"] if "model_state_dict" in state else state
    )
    model.eval()
    print(f"[+] Model loaded from {ckpt}")
    return model


def save_image(tensor_chw, path):
    """Save (C,H,W) float32 [0,1] tensor as PNG."""
    arr = (tensor_chw.cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_diff_image(cover_chw, stego_chw, path, amplify=5.0):
    """Save amplified absolute difference image."""
    diff = (stego_chw - cover_chw).abs() * amplify
    arr = (diff.cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    model = load_model()
    ds = SteganographyDataset("datasets/coco_subset", split="test", seed=42)
    print(f"[+] Loaded {len(ds)} held-out test images\n")

    all_results = []

    for img_idx in range(len(ds)):
        img_name = os.path.splitext(ds.image_files[img_idx])[0]
        cover = ds[img_idx]["image"].unsqueeze(0).to(DEVICE)  # (1,3,256,256)

        # Save the original cover image once
        cover_path = os.path.join(OUTDIR, f"img{img_idx:02d}_cover.png")
        save_image(cover[0], cover_path)

        row = {"image_index": img_idx, "image_name": img_name, "payloads": []}

        for size in PAYLOAD_SIZES:
            payload = bytes([i % 256 for i in range(size)])
            try:
                stego, metrics = model.embed_secret(cover, payload, PASSWORD)
                recovered = model.extract_secret(stego, PASSWORD, as_bytes=True)
                exact_match = (recovered == payload)

                # Re-measure with consistent [0,1] inputs
                c_np = cover[0].cpu().numpy()
                s_np = stego[0].cpu().numpy()
                psnr = calculate_psnr(c_np, s_np, max_val=1.0)
                ssim = calculate_ssim(c_np, s_np, max_val=1.0)
                mse  = calculate_mse(c_np, s_np)

                # Save stego + diff
                tag = f"img{img_idx:02d}_sz{size}"
                save_image(stego[0], os.path.join(OUTDIR, f"{tag}_stego.png"))
                save_diff_image(cover[0], stego[0], os.path.join(OUTDIR, f"{tag}_diff.png"))

                entry = {
                    "payload_bytes": size,
                    "psnr_db": round(float(psnr), 4),
                    "ssim": round(float(ssim), 6),
                    "mse": round(float(mse), 8),
                    "exact_recovery": exact_match,
                }
                row["payloads"].append(entry)
                status = "✓" if exact_match else "✗"
                print(
                    f"  [img{img_idx}] {size:5d}B  PSNR={psnr:.2f}dB  "
                    f"SSIM={ssim:.4f}  MSE={mse:.6f}  recovery={status}"
                )

            except Exception as e:
                print(f"  [img{img_idx}] {size:5d}B  ERROR: {e}")
                row["payloads"].append({"payload_bytes": size, "error": str(e)})

        all_results.append(row)
        print()

    # Aggregate per payload size
    print("=" * 70)
    print("BASELINE AGGREGATE SUMMARY")
    print("=" * 70)
    for size in PAYLOAD_SIZES:
        psnrs, ssims, mses = [], [], []
        for row in all_results:
            for p in row["payloads"]:
                if p.get("payload_bytes") == size and "psnr_db" in p:
                    psnrs.append(p["psnr_db"])
                    ssims.append(p["ssim"])
                    mses.append(p["mse"])
        if psnrs:
            print(
                f"  {size:5d}B  PSNR={np.mean(psnrs):.2f}±{np.std(psnrs):.2f}dB  "
                f"SSIM={np.mean(ssims):.4f}±{np.std(ssims):.4f}  "
                f"MSE={np.mean(mses):.6f}"
            )

    # Overall mean
    all_psnrs = [p["psnr_db"] for r in all_results for p in r["payloads"] if "psnr_db" in p]
    all_ssims = [p["ssim"]    for r in all_results for p in r["payloads"] if "ssim"    in p]
    all_mses  = [p["mse"]     for r in all_results for p in r["payloads"] if "mse"     in p]
    print(f"\n  OVERALL  PSNR={np.mean(all_psnrs):.2f}dB  SSIM={np.mean(all_ssims):.4f}  MSE={np.mean(all_mses):.6f}")

    # Save JSON
    results_path = os.path.join(OUTDIR, "baseline_metrics.json")
    with open(results_path, "w") as f:
        json.dump({
            "model_checkpoint": "checkpoints/phase2b/phase2b_model.pth",
            "scaling_factor": 0.15,
            "results": all_results,
            "summary": {
                "mean_psnr_db": round(float(np.mean(all_psnrs)), 4),
                "mean_ssim": round(float(np.mean(all_ssims)), 6),
                "mean_mse": round(float(np.mean(all_mses)), 8),
            }
        }, f, indent=2)
    print(f"\n[+] Results saved -> {results_path}")
    print(f"[+] Images saved  -> {OUTDIR}/")


if __name__ == "__main__":
    main()
