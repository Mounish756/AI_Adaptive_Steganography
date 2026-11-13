"""
Phase 4 Reliability Benchmark:
Runs an extensive evaluation matrix of 35 trials (5 held-out images x 7 payload configurations).
Measures:
- Exact byte recovery rate
- AES-GCM authentication verification rate
- Raw channel BER before FEC
- Post-FEC effective BER
- Perceptual metrics: PSNR (dB), SSIM, MSE
- Latency and memory consumption
"""

import os
import sys
import time
import json
import psutil
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.models.full_model import AdaptiveStegoModel
from ai.data.dataset import SteganographyDataset
from security.encryption import encrypt_message
from security.payload import (
    encode_payload_to_carrier_map,
    decode_carrier_map_to_payload,
    calculate_usable_capacity,
    bytes_to_bits,
    bits_to_bytes
)


def run_phase4_reliability_matrix(output_dir: str = "outputs/phase4"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    model = AdaptiveStegoModel(
        base_dim=32,
        analyzer_checkpoint="models/adaptive_analyzer/best_model.pth",
        freeze_analyzer=True,
        scaling_factor=0.15
    ).to(device)

    ckpt_path = "checkpoints/phase2b/phase2b_model.pth"
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    test_ds = SteganographyDataset("datasets/coco_subset", split="test", seed=42)

    payload_specs = [
        {"name": "Short Text (50B)", "type": "text", "data": "Alpha-Bravo-Charlie: Stealth Transmission Code 99!"},
        {"name": "Long Text (1KB)", "type": "text", "data": ("Research findings in adaptive steganography. " * 25)[:1000]},
        {"name": "Binary Header (256B)", "type": "binary", "data": bytes([i % 256 for i in range(256)])},
        {"name": "Simulated PDF (512B)", "type": "binary", "data": b"%PDF-1.7\n" + bytes([i % 256 for i in range(503)])},
        {"name": "Medium Binary (2.5KB)", "type": "binary", "data": bytes([((i * 7) ^ 0xA5) % 256 for i in range(2500)])},
        {"name": "Large Binary (5.0KB)", "type": "binary", "data": bytes([((i * 13) ^ 0x5C) % 256 for i in range(5000)])},
        {"name": "Max Payload (7083B)", "type": "binary", "data": bytes([((i * 17) ^ 0x3F) % 256 for i in range(7083)])}
    ]

    print(f"\n{'='*74}")
    print("PHASE 4: EXTENSIVE RELIABILITY EXPERIMENT MATRIX")
    print(f"Held-out Cover Images: {len(test_ds)}")
    print(f"Payload Configurations : {len(payload_specs)}")
    print(f"Total Independent Trials : {len(test_ds) * len(payload_specs)}")
    print(f"{'='*74}\n")

    trials = []
    trial_idx = 0
    process = psutil.Process()

    for img_idx in range(len(test_ds)):
        item = test_ds[img_idx]
        cover = item["image"].unsqueeze(0).to(device)
        fname = os.path.basename(item["image_path"])

        for spec in payload_specs:
            trial_idx += 1
            payload = spec["data"]
            pw = f"PassKey_{img_idx}_{trial_idx}#Secure"

            mem_before = process.memory_info().rss / (1024 * 1024)
            t0 = time.perf_counter()

            # 1. Embed
            stego, metrics = model.embed_secret(cover, payload, pw)

            # 2. Measure Raw Channel BER before FEC
            with torch.no_grad():
                salt, nonce, ct, tag = encrypt_message(payload, pw)
                carrier_map = encode_payload_to_carrier_map(salt, nonce, ct, tag).to(device)
                extracted_carrier = model.decoder(stego)
                raw_pred_bits = (extracted_carrier[0, 0].view(-1) > 0.5).int().tolist()
                true_bits = (carrier_map[0, 0].view(-1) > 0.5).int().tolist()
                raw_bit_errors = sum(b1 != b2 for b1, b2 in zip(raw_pred_bits, true_bits))
                raw_ber = raw_bit_errors / len(true_bits)

            # 3. Extract & Authenticate
            as_bytes = (spec["type"] == "binary")
            try:
                recovered = model.extract_secret(stego, pw, as_bytes=as_bytes)
                auth_success = True
                exact_match = (recovered == payload)
                post_fec_ber = 0.0 if exact_match else 1.0
            except Exception as e:
                auth_success = False
                exact_match = False
                post_fec_ber = 1.0
                recovered = str(e)

            elapsed = time.perf_counter() - t0
            mem_after = process.memory_info().rss / (1024 * 1024)

            trial_result = {
                "trial_id": trial_idx,
                "image": fname,
                "payload_name": spec["name"],
                "payload_bytes": len(payload.encode("utf-8") if isinstance(payload, str) else payload),
                "payload_type": spec["type"],
                "psnr_db": round(metrics["psnr_db"], 2),
                "ssim": round(metrics["ssim"], 4),
                "mse": round(metrics["mse"], 6),
                "raw_channel_ber": round(raw_ber, 6),
                "post_fec_ber": post_fec_ber,
                "authenticated": auth_success,
                "exact_match": exact_match,
                "latency_s": round(elapsed, 3),
                "ram_mb": round(mem_after, 1)
            }
            trials.append(trial_result)

            status_str = "[PASS]" if exact_match and auth_success else "[FAIL]"
            print(
                f"Trial {trial_idx:02d}/35 | {fname[:18]:<18} | {spec['name']:<20} | "
                f"PSNR: {trial_result['psnr_db']:<5.2f}dB | Raw BER: {raw_ber*100:<5.2f}% | "
                f"Post BER: {post_fec_ber*100:<4.2f}% | {status_str} in {elapsed:.2f}s"
            )

    # Compute Summary
    total_trials = len(trials)
    exact_matches = sum(1 for t in trials if t["exact_match"])
    auth_successes = sum(1 for t in trials if t["authenticated"])

    summary = {
        "benchmark_metadata": {
            "total_trials": total_trials,
            "images_count": len(test_ds),
            "payload_types_count": len(payload_specs),
            "exact_recovery_success_rate": f"{(exact_matches / total_trials) * 100:.2f}%",
            "aes_gcm_auth_success_rate": f"{(auth_successes / total_trials) * 100:.2f}%",
            "mean_psnr_db": float(np.mean([t["psnr_db"] for t in trials])),
            "mean_ssim": float(np.mean([t["ssim"] for t in trials])),
            "mean_mse": float(np.mean([t["mse"] for t in trials])),
            "mean_raw_channel_ber": float(np.mean([t["raw_channel_ber"] for t in trials])),
            "mean_post_fec_ber": float(np.mean([t["post_fec_ber"] for t in trials])),
            "mean_latency_s": float(np.mean([t["latency_s"] for t in trials])),
            "max_memory_mb": float(np.max([t["ram_mb"] for t in trials]))
        },
        "individual_trials": trials
    }

    report_path = os.path.join(output_dir, "reliability_benchmark.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'-'*74}")
    print("PHASE 4 RELIABILITY EXPERIMENT SUMMARY:")
    print(f"  Total Independent Trials     : {total_trials}")
    print(f"  Exact Recovery Success Rate  : {summary['benchmark_metadata']['exact_recovery_success_rate']} ({exact_matches}/{total_trials})")
    print(f"  AES-GCM Auth Success Rate    : {summary['benchmark_metadata']['aes_gcm_auth_success_rate']} ({auth_successes}/{total_trials})")
    print(f"  Mean Raw Channel BER         : {summary['benchmark_metadata']['mean_raw_channel_ber']*100:.3f}%")
    print(f"  Mean Post-FEC Effective BER  : {summary['benchmark_metadata']['mean_post_fec_ber']*100:.4f}%")
    print(f"  Mean PSNR                    : {summary['benchmark_metadata']['mean_psnr_db']:.2f} dB")
    print(f"  Mean SSIM                    : {summary['benchmark_metadata']['mean_ssim']:.4f}")
    print(f"  Mean MSE                     : {summary['benchmark_metadata']['mean_mse']:.6f}")
    print(f"  Mean End-to-End Latency      : {summary['benchmark_metadata']['mean_latency_s']:.3f} s")
    print(f"{'='*74}\n")
    print(f"[+] Full JSON results saved to: {report_path}")
    return summary


if __name__ == "__main__":
    run_phase4_reliability_matrix()
