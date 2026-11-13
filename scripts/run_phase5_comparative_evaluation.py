"""
Comprehensive Phase 5 Evaluation and Side-by-Side Visual Comparison Script.

Evaluates:
  1. Baseline Model (Phase 2B checkpoint, SF=0.15)
  2. Improved Balanced Model (Phase 5 balanced checkpoint, SF=0.13)

Across:
  - 5 Held-out test images
  - 4 Payload sizes: 64B, 512B, 2048B, 7083B (Max Capacity)
  - Metrics: PSNR (dB), SSIM, MSE, Raw Channel BER, Post-FEC BER, Exact Recovery (%), Runtime (s)
  - Side-by-side visual comparisons: Cover | Baseline Stego | Improved Stego | Diff Baseline (x5) | Diff Improved (x5)
"""

import os
import sys
import time
import json
import numpy as np
from PIL import Image

sys.path.insert(0, ".")
import torch

from ai.models.full_model import AdaptiveStegoModel
from ai.data.dataset import SteganographyDataset
from ai.metrics.psnr import calculate_psnr
from ai.metrics.ssim import calculate_ssim
from ai.metrics.mse import calculate_mse

DEVICE = torch.device("cpu")
OUTDIR = "outputs/phase5/comparison"
os.makedirs(OUTDIR, exist_ok=True)

PAYLOAD_SIZES = [64, 512, 2048, 7083]
PASSWORD = "Phase5ValidationKey#2026"

def tensor_to_np(tensor_chw):
    return (tensor_chw.cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)

def save_side_by_side_grid(cover, stego_base, stego_imp, out_path, amplify=5.0):
    """
    Creates a 5-panel comparison:
    [ Cover ] [ Baseline Stego ] [ Improved Stego ] [ Baseline Diff x5 ] [ Improved Diff x5 ]
    """
    diff_base = (stego_base - cover).abs() * amplify
    diff_imp  = (stego_imp  - cover).abs() * amplify

    c_img  = Image.fromarray(tensor_to_np(cover))
    sb_img = Image.fromarray(tensor_to_np(stego_base))
    si_img = Image.fromarray(tensor_to_np(stego_imp))
    db_img = Image.fromarray(tensor_to_np(diff_base))
    di_img = Image.fromarray(tensor_to_np(diff_imp))

    w, h = c_img.size
    grid = Image.new("RGB", (w * 5, h))
    grid.paste(c_img,  (0, 0))
    grid.paste(sb_img, (w, 0))
    grid.paste(si_img, (w * 2, 0))
    grid.paste(db_img, (w * 3, 0))
    grid.paste(di_img, (w * 4, 0))
    grid.save(out_path)

def evaluate_model_pipeline(model, ds, scaling_factor, model_name):
    print(f"\n--- Evaluating {model_name} (SF={scaling_factor}) ---")
    results = []
    
    for img_idx in range(len(ds)):
        cover = ds[img_idx]["image"].unsqueeze(0).to(DEVICE)
        img_name = os.path.basename(ds.image_files[img_idx])
        
        img_res = {"image_index": img_idx, "filename": img_name, "trials": []}
        
        for size in PAYLOAD_SIZES:
            payload = bytes([k % 256 for k in range(size)])
            t0 = time.time()
            
            try:
                stego, m = model.embed_secret(cover, payload, PASSWORD, scaling_factor=scaling_factor)
                t_embed = time.time() - t0
                
                t1 = time.time()
                recovered = model.extract_secret(stego, PASSWORD, as_bytes=True)
                t_extract = time.time() - t1
                
                exact_recovery = (recovered == payload)
                
                # Raw Channel BER via neural decoder directly
                with torch.no_grad():
                    neural_pred = model.decoder(stego)
                    # Extract the carrier map from the stego embedding
                    # We compare against the binary carrier map
                    salt, nonce, ct, tag = model.embed_secret.__globals__['encrypt_message'](payload, PASSWORD)
                    carrier_map = model.embed_secret.__globals__['encode_payload_to_carrier_map'](salt, nonce, ct, tag).to(DEVICE)
                    raw_bits_pred = (neural_pred > 0.5).float()
                    raw_ber = (raw_bits_pred != carrier_map).float().mean().item()
                
                post_fec_ber = 0.0 if exact_recovery else 1.0
                
                trial_data = {
                    "payload_size": size,
                    "psnr_db": round(float(m["psnr_db"]), 4),
                    "ssim": round(float(m["ssim"]), 6),
                    "mse": round(float(m["mse"]), 8),
                    "raw_ber": round(float(raw_ber), 6),
                    "post_fec_ber": post_fec_ber,
                    "exact_recovery": exact_recovery,
                    "embed_time_s": round(t_embed, 4),
                    "extract_time_s": round(t_extract, 4),
                    "total_time_s": round(t_embed + t_extract, 4),
                    "stego_tensor": stego[0].clone()
                }
                img_res["trials"].append(trial_data)
                status = "PASS" if exact_recovery else "FAIL"
                print(f"  [{img_idx}] {img_name[:20]:<20} | {size:5d}B | PSNR: {m['psnr_db']:.2f}dB | SSIM: {m['ssim']:.4f} | RawBER: {raw_ber*100:.2f}% | {status}")
                
            except Exception as e:
                print(f"  [{img_idx}] {img_name[:20]:<20} | {size:5d}B | ERROR: {e}")
                img_res["trials"].append({
                    "payload_size": size,
                    "error": str(e),
                    "exact_recovery": False
                })
        
        results.append(img_res)
    
    return results

def main():
    ds = SteganographyDataset("datasets/coco_subset", split="test", seed=42)
    print(f"Loaded {len(ds)} held-out test images.")
    
    # 1. Load Baseline Model
    base_model = AdaptiveStegoModel(base_dim=32, scaling_factor=0.15).to(DEVICE)
    ckpt_base = torch.load("checkpoints/phase2b/phase2b_model.pth", map_location=DEVICE, weights_only=False)
    base_model.load_state_dict(ckpt_base["model_state_dict"])
    base_model.eval()
    
    # 2. Load Improved Balanced Model
    imp_model = AdaptiveStegoModel(base_dim=32, scaling_factor=0.13).to(DEVICE)
    ckpt_imp = torch.load("checkpoints/phase5_balanced/best_model.pth", map_location=DEVICE, weights_only=False)
    imp_model.load_state_dict(ckpt_imp["model_state_dict"])
    imp_model.eval()
    
    base_results = evaluate_model_pipeline(base_model, ds, scaling_factor=0.15, model_name="Baseline (Phase 2B)")
    imp_results  = evaluate_model_pipeline(imp_model, ds, scaling_factor=0.13, model_name="Improved Balanced Model")
    
    # Generate side-by-side images
    print("\nSaving side-by-side visual comparisons...")
    for img_idx in range(len(ds)):
        cover = ds[img_idx]["image"].to(DEVICE)
        img_name = os.path.splitext(os.path.basename(ds.image_files[img_idx]))[0]
        
        # Use 2048B payload for the representative visual comparison
        stego_base = base_results[img_idx]["trials"][2]["stego_tensor"]
        stego_imp  = imp_results[img_idx]["trials"][2]["stego_tensor"]
        
        out_path = os.path.join(OUTDIR, f"{img_name}_comparison_grid.png")
        save_side_by_side_grid(cover, stego_base, stego_imp, out_path, amplify=5.0)
        print(f"  Saved: {out_path}")
    
    # Compute summary aggregates
    def summarize(res_list):
        psnrs = [t["psnr_db"] for r in res_list for t in r["trials"] if "psnr_db" in t]
        ssims = [t["ssim"] for r in res_list for t in r["trials"] if "ssim" in t]
        mses  = [t["mse"] for r in res_list for t in r["trials"] if "mse" in t]
        bers  = [t["raw_ber"] for r in res_list for t in r["trials"] if "raw_ber" in t]
        recs  = [1 if t.get("exact_recovery") else 0 for r in res_list for t in r["trials"]]
        times = [t["total_time_s"] for r in res_list for t in r["trials"] if "total_time_s" in t]
        
        return {
            "mean_psnr_db": float(np.mean(psnrs)),
            "std_psnr_db": float(np.std(psnrs)),
            "mean_ssim": float(np.mean(ssims)),
            "std_ssim": float(np.std(ssims)),
            "mean_mse": float(np.mean(mses)),
            "mean_raw_ber": float(np.mean(bers)),
            "recovery_rate_pct": float(np.mean(recs) * 100.0),
            "total_trials": len(recs),
            "passed_trials": int(sum(recs)),
            "mean_runtime_s": float(np.mean(times))
        }

    summary_base = summarize(base_results)
    summary_imp  = summarize(imp_results)
    
    print("\n" + "=" * 75)
    print("FINAL COMPARATIVE EVALUATION RESULTS")
    print("=" * 75)
    print(f"{'Metric':<25} | {'Baseline (Phase 2B)':<22} | {'Improved (Phase 5 Balanced)':<25}")
    print("-" * 75)
    print(f"{'Mean PSNR (dB)':<25} | {summary_base['mean_psnr_db']:.2f} ± {summary_base['std_psnr_db']:.2f} dB{'':<7} | {summary_imp['mean_psnr_db']:.2f} ± {summary_imp['std_psnr_db']:.2f} dB (+{summary_imp['mean_psnr_db']-summary_base['mean_psnr_db']:.2f} dB)")
    print(f"{'Mean SSIM':<25} | {summary_base['mean_ssim']:.4f} ± {summary_base['std_ssim']:.4f}{'':<7} | {summary_imp['mean_ssim']:.4f} ± {summary_imp['std_ssim']:.4f} (+{summary_imp['mean_ssim']-summary_base['mean_ssim']:.4f})")
    print(f"{'Mean MSE':<25} | {summary_base['mean_mse']:.6f}{'':<13} | {summary_imp['mean_mse']:.6f} (-{(1-summary_imp['mean_mse']/summary_base['mean_mse'])*100:.1f}%)")
    print(f"{'Mean Raw Channel BER':<25} | {summary_base['mean_raw_ber']*100:.2f}%{'':<16} | {summary_imp['mean_raw_ber']*100:.2f}%")
    print(f"{'Exact Payload Recovery':<25} | {summary_base['passed_trials']}/{summary_base['total_trials']} ({summary_base['recovery_rate_pct']:.1f}%){'':<8} | {summary_imp['passed_trials']}/{summary_imp['total_trials']} ({summary_imp['recovery_rate_pct']:.1f}%)")
    print(f"{'Mean Runtime per Trial':<25} | {summary_base['mean_runtime_s']:.2f} s{'':<15} | {summary_imp['mean_runtime_s']:.2f} s")
    print("=" * 75)
    
    # Strip stego_tensor before saving JSON
    for res in [base_results, imp_results]:
        for img in res:
            for t in img["trials"]:
                if "stego_tensor" in t:
                    del t["stego_tensor"]
                    
    with open(os.path.join(OUTDIR, "evaluation_summary.json"), "w") as f:
        json.dump({
            "summary_baseline": summary_base,
            "summary_improved": summary_imp,
            "baseline_results": base_results,
            "improved_results": imp_results
        }, f, indent=2)
        
    print(f"\n[+] Detailed summary saved -> {OUTDIR}/evaluation_summary.json")

if __name__ == "__main__":
    main()
