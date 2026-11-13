"""
Phase 3 End-to-End Evaluation Script:
Demonstrates AES-256-GCM encryption, Reed-Solomon Forward Error Correction (FEC),
Neural Adaptive Embedding (UNetEncoder), Extraction (UNetDecoder),
Authentication and Exact Plaintext Recovery across all 5 held-out test images.
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.models.full_model import AdaptiveStegoModel
from ai.data.dataset import SteganographyDataset
from security.payload import calculate_usable_capacity


def run_phase3_benchmark(output_dir: str = "outputs/phase3"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    model = AdaptiveStegoModel(
        base_dim=32,
        analyzer_checkpoint="models/adaptive_analyzer/best_model.pth",
        freeze_analyzer=True,
        scaling_factor=0.15
    ).to(device)

    ckpt_path = "checkpoints/phase2b/phase2b_model.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/phase2a/best_model.pth"

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    test_ds = SteganographyDataset("datasets/coco_subset", split="test", seed=42)

    secret_messages = [
        "Confidential Mission Parameters: Latitude 37.7749 N, Longitude 122.4194 W. Authorized Personnel Only.",
        "AES-256-GCM Authenticated Steganography: Neural Carrier Channel Verified with 100% Exact Recovery.",
        "Top Secret Key Exchange Protocol: Ephemeral Session Token 0x8F91A34E72B510C9, Valid for 3600 seconds.",
        "Medical Records Cryptographic Archive: Patient ID #948271, Status Stable, Treatment Plan Approved.",
        "Project Deliverable: Phase 3 End-to-End AI-Based Adaptive Information Concealment Successfully Completed!"
    ]
    password = "MasterPhase3SecretPassword#2026"

    max_usable_cap = calculate_usable_capacity((256, 256), nsym=32)

    print(f"\n{'='*70}")
    print("PHASE 3: END-TO-END CRYPTOGRAPHIC STEGANOGRAPHY BENCHMARK")
    print(f"Model Checkpoint: {ckpt_path}")
    print(f"Maximum Usable Plaintext Capacity: {max_usable_cap} bytes")
    print(f"Held-Out Test Images: {len(test_ds)}")
    print(f"{'='*70}\n")

    results = []

    for idx in range(len(test_ds)):
        item = test_ds[idx]
        cover = item["image"].unsqueeze(0).to(device)
        fname = os.path.basename(item["image_path"])
        msg = secret_messages[idx % len(secret_messages)]

        # 1. Embed
        stego, metrics = model.embed_secret(cover, msg, password, nsym=32)

        # Measure raw channel BER before error correction
        with torch.no_grad():
            suit = model.adaptive_analyzer(cover)
            from security.encryption import encrypt_message
            from security.payload import encode_payload_to_carrier_map, bytes_to_bits, bits_to_bytes
            salt, nonce, ct, tag = encrypt_message(msg, password)
            carrier_map = encode_payload_to_carrier_map(salt, nonce, ct, tag, nsym=32).to(device)
            extracted = model.decoder(stego)
            raw_pred_bits = (extracted[0, 0].view(-1) > 0.5).int().tolist()
            true_bits = (carrier_map[0, 0].view(-1) > 0.5).int().tolist()
            raw_errors = sum(b1 != b2 for b1, b2 in zip(raw_pred_bits, true_bits))
            raw_ber = raw_errors / len(true_bits)

        # 2. Extract & Authenticate
        recovered_text = model.extract_secret(stego, password, as_bytes=False, nsym=32)

        # Exact match assertion
        match = (recovered_text == msg)
        assert match, f"Mismatch in image {fname}"

        # Post-FEC BER is exactly 0.0%
        fec_ber = 0.0

        res_item = {
            "image": fname,
            "plaintext_bytes": len(msg.encode("utf-8")),
            "ciphertext_bytes": metrics["ciphertext_bytes"],
            "psnr_db": round(metrics["psnr_db"], 2),
            "ssim": round(metrics["ssim"], 4),
            "mse": round(metrics["mse"], 6),
            "raw_channel_ber": round(raw_ber, 6),
            "post_fec_ber": fec_ber,
            "authenticated": True,
            "exact_match": match
        }
        results.append(res_item)

        print(
            f"[{idx+1}/{len(test_ds)}] {fname:<25} | "
            f"PSNR: {res_item['psnr_db']:.2f}dB | SSIM: {res_item['ssim']:.4f} | "
            f"Raw BER: {raw_ber*100:.3f}% -> Post-FEC BER: 0.000% | Auth: [VERIFIED]"
        )

        # Save sample visual artifacts for first image
        if idx == 0:
            c_np = (cover[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
            s_np = (stego[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
            res_diff = ((stego[0] - cover[0]).abs().permute(1, 2, 0).numpy().clip(0, 1) * 10.0 * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(c_np).save(os.path.join(output_dir, "sample_cover.png"))
            Image.fromarray(s_np).save(os.path.join(output_dir, "sample_stego.png"))
            Image.fromarray(res_diff).save(os.path.join(output_dir, "sample_residual_x10.png"))

    summary = {
        "dataset": "datasets/coco_subset (held-out test split)",
        "num_test_images": len(results),
        "mean_psnr_db": float(np.mean([r["psnr_db"] for r in results])),
        "mean_ssim": float(np.mean([r["ssim"] for r in results])),
        "mean_mse": float(np.mean([r["mse"] for r in results])),
        "mean_raw_channel_ber": float(np.mean([r["raw_channel_ber"] for r in results])),
        "mean_post_fec_ber": 0.0,
        "max_usable_plaintext_bytes": max_usable_cap,
        "authentication_success_rate": "100.0%",
        "exact_recovery_rate": "100.0%",
        "per_image_results": results
    }

    report_path = os.path.join(output_dir, "phase3_benchmark_metrics.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'-'*70}")
    print("PHASE 3 BENCHMARK SUMMARY:")
    print(f"  Mean PSNR               : {summary['mean_psnr_db']:.2f} dB")
    print(f"  Mean SSIM               : {summary['mean_ssim']:.4f}")
    print(f"  Mean MSE                : {summary['mean_mse']:.6f}")
    print(f"  Raw Channel BER         : {summary['mean_raw_channel_ber']*100:.3f}%")
    print(f"  Post-FEC Effective BER  : {summary['mean_post_fec_ber']:.4f}%")
    print(f"  AES-GCM Auth Success    : {summary['authentication_success_rate']}")
    print(f"  Exact Plaintext Recovery: {summary['exact_recovery_rate']}")
    print(f"  Max Plaintext Capacity  : {max_usable_cap} bytes")
    print(f"{'='*70}\n")
    print(f"[+] Benchmark results saved to: {report_path}")
    return summary


if __name__ == "__main__":
    run_phase3_benchmark()
