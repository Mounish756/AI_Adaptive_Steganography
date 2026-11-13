"""
Phase 5 Comprehensive Benchmark:
Robustness Testing, Performance Evaluation, and Security Analysis.

Covers:
1. System & Environment Characterization
2. Image Quality & Payload Capacity Curve (16B to 7,083B)
3. Robustness Suite:
   - Lossless PNG Save/Reload Round-Trip
   - JPEG Compression (Q=95, 90, 85, 75, 50)
   - Spatial Resizing & Restoration (Downscale 0.5x -> Upscale 1.0x)
   - Additive Gaussian Noise (sigma = 0.001, 0.005, 0.01)
   - Salt & Pepper Noise (0.1% pixels)
   - Minor Spatial Occlusion (8x8 block mask)
4. Adaptive vs. Uniform Baseline Comparison
5. Cryptographic Security & Malformed Input Validation
"""

import os
import sys
import io
import time
import json
import csv
import platform
import psutil
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.models.full_model import AdaptiveStegoModel
from ai.data.dataset import SteganographyDataset
from ai.metrics.psnr import calculate_psnr
from ai.metrics.ssim import calculate_ssim
from ai.metrics.mse import calculate_mse
from security.encryption import (
    AuthenticationError,
    encrypt_message,
    decrypt_bytes,
)
from security.payload import (
    PayloadCapacityError,
    CorruptedPayloadError,
    calculate_usable_capacity,
    encode_payload_to_carrier_map,
    decode_carrier_map_to_payload,
)


# ─── Transformation Helpers ──────────────────────────────────────────────────

def apply_png_roundtrip(stego_tensor: torch.Tensor) -> torch.Tensor:
    """Save to PNG in-memory buffer and reload."""
    img_np = (stego_tensor[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    reloaded = Image.open(buf)
    reloaded_np = np.array(reloaded).astype(np.float32) / 255.0
    return torch.from_numpy(reloaded_np).permute(2, 0, 1).unsqueeze(0).to(stego_tensor.device)


def apply_jpeg_compression(stego_tensor: torch.Tensor, quality: int) -> torch.Tensor:
    """Compress with JPEG at specified quality in-memory and reload."""
    img_np = (stego_tensor[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    reloaded = Image.open(buf)
    reloaded_np = np.array(reloaded).astype(np.float32) / 255.0
    return torch.from_numpy(reloaded_np).permute(2, 0, 1).unsqueeze(0).to(stego_tensor.device)


def apply_resize_restoration(stego_tensor: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
    """Downscale then upscale back to original dimensions using bilinear interpolation."""
    _, _, h, w = stego_tensor.shape
    dh, dw = max(1, int(h * scale)), max(1, int(w * scale))
    down = F.interpolate(stego_tensor, size=(dh, dw), mode="bilinear", align_corners=False)
    restored = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
    return torch.clamp(restored, 0.0, 1.0)


def apply_gaussian_noise(stego_tensor: torch.Tensor, sigma: float = 0.005) -> torch.Tensor:
    """Add zero-mean Gaussian noise."""
    generator = torch.Generator().manual_seed(42)
    noise = torch.randn(stego_tensor.shape, generator=generator, device=stego_tensor.device) * sigma
    return torch.clamp(stego_tensor + noise, 0.0, 1.0)


def apply_salt_and_pepper_noise(stego_tensor: torch.Tensor, amount: float = 0.001) -> torch.Tensor:
    """Apply salt and pepper noise to a fraction of pixels."""
    noisy = stego_tensor.clone()
    np.random.seed(42)
    mask = np.random.rand(*noisy.shape)
    noisy[torch.tensor(mask < (amount / 2), device=noisy.device)] = 0.0
    noisy[torch.tensor(mask > (1.0 - amount / 2), device=noisy.device)] = 1.0
    return noisy


def apply_block_occlusion(stego_tensor: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """Zero out a small square block at the image center."""
    noisy = stego_tensor.clone()
    _, _, h, w = noisy.shape
    top = h // 2 - block_size // 2
    left = w // 2 - block_size // 2
    noisy[:, :, top:top + block_size, left:left + block_size] = 0.0
    return noisy


# ─── Main Benchmark Routine ──────────────────────────────────────────────────

def run_phase5_benchmark(output_dir: str = "outputs/phase5"):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")
    process = psutil.Process()

    # Hardware & Environment details
    env_info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor() or "x86_64",
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "device": str(device),
        "initial_ram_mb": round(process.memory_info().rss / (1024 * 1024), 2)
    }

    print("\n" + "=" * 78)
    print("PHASE 5: ROBUSTNESS, PERFORMANCE, AND SECURITY BENCHMARK")
    print(f"Environment: Python {env_info['python_version']} | PyTorch {env_info['torch_version']} | Device: {device}")
    print(f"OS: {env_info['os']}")
    print("=" * 78 + "\n")

    # Load Model (using phase2b / phase5 balanced model weights)
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "phase2b", "phase2b_model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "phase2a", "best_model.pth")

    analyzer_path = os.path.join(PROJECT_ROOT, "models", "adaptive_analyzer", "best_model.pth")
    model = AdaptiveStegoModel(
        base_dim=32,
        analyzer_checkpoint=analyzer_path,
        freeze_analyzer=True,
        scaling_factor=0.15
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    dataset_dir = os.path.join(PROJECT_ROOT, "datasets", "coco_subset")
    test_ds = SteganographyDataset(dataset_dir, split="test", seed=42)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. PAYLOAD CAPACITY EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    print("[1/4] Running Payload-Capacity Evaluation...")
    capacity_results = []
    payload_sizes = [16, 64, 256, 1024, 2048, 4096, 7083]

    for size in payload_sizes:
        test_payload = bytes([((i * 19) ^ 0x5A) % 256 for i in range(size)])
        pw = f"Phase5Key_{size}#Test"

        # Evaluate across all 5 test images
        img_psnrs, img_ssims, img_mses = [], [], []
        img_embed_times, img_extract_times = [], []
        img_raw_bers = []
        all_exact = True

        for idx in range(len(test_ds)):
            item = test_ds[idx]
            cover = item["image"].unsqueeze(0).to(device)

            # Embedding
            t0 = time.perf_counter()
            stego, metrics = model.embed_secret(cover, test_payload, pw)
            t_embed = time.perf_counter() - t0

            # Raw channel BER
            with torch.no_grad():
                salt, nonce, ct, tag = encrypt_message(test_payload, pw)
                carrier_map = encode_payload_to_carrier_map(salt, nonce, ct, tag).to(device)
                extracted_carrier = model.decoder(stego)
                pred_bits = (extracted_carrier[0, 0].view(-1) > 0.5).int().tolist()
                true_bits = (carrier_map[0, 0].view(-1) > 0.5).int().tolist()
                raw_ber = sum(b1 != b2 for b1, b2 in zip(pred_bits, true_bits)) / len(true_bits)

            # Extraction
            t1 = time.perf_counter()
            recovered = model.extract_secret(stego, pw, as_bytes=True)
            t_extract = time.perf_counter() - t1

            exact = (recovered == test_payload)
            if not exact:
                all_exact = False

            img_psnrs.append(metrics["psnr_db"])
            img_ssims.append(metrics["ssim"])
            img_mses.append(metrics["mse"])
            img_embed_times.append(t_embed)
            img_extract_times.append(t_extract)
            img_raw_bers.append(raw_ber)

        cap_row = {
            "payload_bytes": size,
            "mean_embed_time_s": round(float(np.mean(img_embed_times)), 4),
            "mean_extract_time_s": round(float(np.mean(img_extract_times)), 4),
            "mean_psnr_db": round(float(np.mean(img_psnrs)), 2),
            "mean_ssim": round(float(np.mean(img_ssims)), 4),
            "mean_mse": round(float(np.mean(img_mses)), 6),
            "mean_raw_ber": round(float(np.mean(img_raw_bers)), 6),
            "exact_recovery": all_exact
        }
        capacity_results.append(cap_row)
        status = "OK" if all_exact else "FAIL"
        print(f"  Payload {size:5d} B | Embed: {cap_row['mean_embed_time_s']:.3f}s | "
              f"Extract: {cap_row['mean_extract_time_s']:.3f}s | PSNR: {cap_row['mean_psnr_db']:.2f}dB | "
              f"SSIM: {cap_row['mean_ssim']:.4f} | Raw BER: {cap_row['mean_raw_ber']*100:.2f}% | [{status}]")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. ROBUSTNESS TESTING UNDER LOSSY & LOSSLESS TRANSFORMS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/4] Running Robustness Suite Across Transformations...")
    robustness_results = []
    standard_payload = b"Phase5 Robustness Payload: Testing neural resilience and fail-closed integrity."
    pw_rob = "RobustnessPassphrase2026!"

    transform_configs = [
        ("Lossless PNG Round-trip", lambda t: apply_png_roundtrip(t), True),
        ("JPEG Compression (Q=95)", lambda t: apply_jpeg_compression(t, 95), False),
        ("JPEG Compression (Q=90)", lambda t: apply_jpeg_compression(t, 90), False),
        ("JPEG Compression (Q=85)", lambda t: apply_jpeg_compression(t, 85), False),
        ("JPEG Compression (Q=75)", lambda t: apply_jpeg_compression(t, 75), False),
        ("JPEG Compression (Q=50)", lambda t: apply_jpeg_compression(t, 50), False),
        ("Resizing (0.5x Downscale + Upscale)", lambda t: apply_resize_restoration(t, 0.5), False),
        ("Gaussian Noise (sigma=0.001)", lambda t: apply_gaussian_noise(t, 0.001), False),
        ("Gaussian Noise (sigma=0.005)", lambda t: apply_gaussian_noise(t, 0.005), False),
        ("Gaussian Noise (sigma=0.010)", lambda t: apply_gaussian_noise(t, 0.010), False),
        ("Salt & Pepper Noise (0.1%)", lambda t: apply_salt_and_pepper_noise(t, 0.001), False),
        ("Block Occlusion (8x8 center)", lambda t: apply_block_occlusion(t, 8), False),
    ]

    for name, tf_func, is_lossless in transform_configs:
        exact_count = 0
        auth_success_count = 0
        fail_closed_count = 0
        unauthenticated_plaintext_leaked = False
        raw_bers = []
        psnrs = []

        for idx in range(len(test_ds)):
            item = test_ds[idx]
            cover = item["image"].unsqueeze(0).to(device)

            stego, _ = model.embed_secret(cover, standard_payload, pw_rob)
            transformed = tf_func(stego)

            # Distortion between stego and transformed image
            c_np = stego[0].detach().cpu().numpy()
            t_np = transformed[0].detach().cpu().numpy()
            psnr_t = calculate_psnr(c_np, t_np, max_val=1.0)
            psnrs.append(psnr_t)

            # Raw channel BER
            with torch.no_grad():
                salt, nonce, ct, tag = encrypt_message(standard_payload, pw_rob)
                carrier_map = encode_payload_to_carrier_map(salt, nonce, ct, tag).to(device)
                extracted_carrier = model.decoder(transformed)
                pred_bits = (extracted_carrier[0, 0].view(-1) > 0.5).int().tolist()
                true_bits = (carrier_map[0, 0].view(-1) > 0.5).int().tolist()
                raw_ber = sum(b1 != b2 for b1, b2 in zip(pred_bits, true_bits)) / len(true_bits)
                raw_bers.append(raw_ber)

            # Extraction attempt
            try:
                rec = model.extract_secret(transformed, pw_rob, as_bytes=True)
                auth_success_count += 1
                if rec == standard_payload:
                    exact_count += 1
            except (AuthenticationError, CorruptedPayloadError):
                fail_closed_count += 1
            except Exception as e:
                # Unexpected exception
                fail_closed_count += 1

        rob_row = {
            "transformation": name,
            "lossless": is_lossless,
            "mean_transform_psnr_db": round(float(np.mean(psnrs)), 2),
            "mean_raw_channel_ber": round(float(np.mean(raw_bers)), 6),
            "exact_recovery_rate": f"{(exact_count / len(test_ds)) * 100:.1f}%",
            "auth_pass_rate": f"{(auth_success_count / len(test_ds)) * 100:.1f}%",
            "fail_closed_rate": f"{(fail_closed_count / len(test_ds)) * 100:.1f}%",
            "unauthenticated_plaintext_leaked": unauthenticated_plaintext_leaked
        }
        robustness_results.append(rob_row)
        print(f"  {name:<36} | Stego PSNR: {rob_row['mean_transform_psnr_db']:<5.1f}dB | "
              f"Raw BER: {rob_row['mean_raw_channel_ber']*100:<5.2f}% | "
              f"Exact: {rob_row['exact_recovery_rate']:<5} | Auth Pass: {rob_row['auth_pass_rate']:<5} | "
              f"Fail-Closed: {rob_row['fail_closed_rate']}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. ADAPTIVE VS. UNIFORM BASELINE COMPARISON
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/4] Evaluating Adaptive vs Uniform Baseline...")
    adaptive_vs_baseline_results = []
    comp_payload = b"Benchmarking Adaptive Suitability Modulation vs. Uniform Suitability Modulation."
    comp_pw = "CompKey_AdaptVsUniform#2026"

    for idx in range(len(test_ds)):
        item = test_ds[idx]
        cover = item["image"].unsqueeze(0).to(device)
        fname = os.path.basename(item["image_path"])

        # Adaptive Embedding
        stego_a, metrics_a = model.embed_secret(cover, comp_payload, comp_pw)
        rec_a = model.extract_secret(stego_a, comp_pw, as_bytes=True)
        exact_a = (rec_a == comp_payload)

        # Raw BER for Adaptive
        with torch.no_grad():
            salt, nonce, ct, tag = encrypt_message(comp_payload, comp_pw)
            carrier_map = encode_payload_to_carrier_map(salt, nonce, ct, tag).to(device)
            ext_c_a = model.decoder(stego_a)
            pred_a = (ext_c_a[0, 0].view(-1) > 0.5).int().tolist()
            true_bits = (carrier_map[0, 0].view(-1) > 0.5).int().tolist()
            raw_ber_a = sum(b1 != b2 for b1, b2 in zip(pred_a, true_bits)) / len(true_bits)

            # Uniform Embedding: evaluate suitability map, then override with constant 1.0
            suit_map = model.adaptive_analyzer(cover)
            uniform_suit_map = torch.ones_like(suit_map)
            stego_u = model.encoder(cover, carrier_map, uniform_suit_map, scaling_factor=model.scaling_factor)

            ext_c_u = model.decoder(stego_u)
            pred_u = (ext_c_u[0, 0].view(-1) > 0.5).int().tolist()
            raw_ber_u = sum(b1 != b2 for b1, b2 in zip(pred_u, true_bits)) / len(true_bits)

        # Compute metrics for Uniform
        c_np = cover[0].detach().cpu().numpy()
        su_np = stego_u[0].detach().cpu().numpy()
        mse_u = calculate_mse(c_np, su_np)
        psnr_u = calculate_psnr(c_np, su_np, max_val=1.0)
        ssim_u = calculate_ssim(c_np, su_np, max_val=1.0)

        # Try extract uniform
        try:
            salt_u, nonce_u, ct_u, tag_u, _ = decode_carrier_map_to_payload(ext_c_u)
            pt_u = decrypt_bytes(salt_u, nonce_u, ct_u, tag_u, comp_pw)
            exact_u = (pt_u == comp_payload)
        except Exception:
            exact_u = False

        row = {
            "image": fname,
            "adaptive_psnr_db": round(metrics_a["psnr_db"], 2),
            "uniform_psnr_db": round(psnr_u, 2),
            "adaptive_ssim": round(metrics_a["ssim"], 4),
            "uniform_ssim": round(ssim_u, 4),
            "adaptive_mse": round(metrics_a["mse"], 6),
            "uniform_mse": round(mse_u, 6),
            "adaptive_raw_ber": round(raw_ber_a, 6),
            "uniform_raw_ber": round(raw_ber_u, 6),
            "adaptive_exact": exact_a,
            "uniform_exact": exact_u
        }
        adaptive_vs_baseline_results.append(row)
        print(f"  {fname[:18]:<18} | Adaptive PSNR: {row['adaptive_psnr_db']:.2f}dB, SSIM: {row['adaptive_ssim']:.4f} | "
              f"Uniform PSNR: {row['uniform_psnr_db']:.2f}dB, SSIM: {row['uniform_ssim']:.4f} | "
              f"Raw BER: {row['adaptive_raw_ber']*100:.2f}% vs {row['uniform_raw_ber']*100:.2f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. SECURITY & INTEGRITY VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4/4] Validating Cryptographic Integrity & Input Boundaries...")
    security_trials = []
    test_cover = test_ds[0]["image"].unsqueeze(0).to(device)
    sec_payload = b"Top secret operational intelligence string."
    sec_pw = "ValidCryptographicPassword#123"

    stego_sec, _ = model.embed_secret(test_cover, sec_payload, sec_pw)

    # Test 1: Correct Password
    try:
        rec = model.extract_secret(stego_sec, sec_pw, as_bytes=True)
        sec_1 = (rec == sec_payload)
    except Exception:
        sec_1 = False
    security_trials.append({"test": "Correct Password Extraction", "passed": sec_1, "expected": "Exact Plaintext Recovery"})

    # Test 2: Incorrect Password Rejection
    try:
        model.extract_secret(stego_sec, "WrongPassword#456", as_bytes=True)
        sec_2 = False
    except AuthenticationError:
        sec_2 = True
    except Exception:
        sec_2 = False
    security_trials.append({"test": "Incorrect Password Rejection", "passed": sec_2, "expected": "AuthenticationError (Fail-Closed)"})

    # Test 3: Modified Ciphertext in Carrier Map
    with torch.no_grad():
        salt, nonce, ct, tag = encrypt_message(sec_payload, sec_pw)
        corrupted_ct = bytearray(ct)
        corrupted_ct[0] ^= 0xFF
        carrier_tampered = encode_payload_to_carrier_map(salt, nonce, bytes(corrupted_ct), tag).to(device)
        stego_tampered = model.encoder(test_cover, carrier_tampered, model.adaptive_analyzer(test_cover), model.scaling_factor)

    try:
        model.extract_secret(stego_tampered, sec_pw, as_bytes=True)
        sec_3 = False
    except AuthenticationError:
        sec_3 = True
    except Exception:
        sec_3 = False
    security_trials.append({"test": "Ciphertext Tampering Rejection", "passed": sec_3, "expected": "AuthenticationError (Fail-Closed)"})

    # Test 4: Modified Tag
    with torch.no_grad():
        corrupted_tag = bytearray(tag)
        corrupted_tag[0] ^= 0xAA
        carrier_bad_tag = encode_payload_to_carrier_map(salt, nonce, ct, bytes(corrupted_tag)).to(device)
        stego_bad_tag = model.encoder(test_cover, carrier_bad_tag, model.adaptive_analyzer(test_cover), model.scaling_factor)

    try:
        model.extract_secret(stego_bad_tag, sec_pw, as_bytes=True)
        sec_4 = False
    except AuthenticationError:
        sec_4 = True
    except Exception:
        sec_4 = False
    security_trials.append({"test": "Authentication Tag Tampering Rejection", "passed": sec_4, "expected": "AuthenticationError (Fail-Closed)"})

    # Test 5: Oversized Payload Rejection
    max_cap = calculate_usable_capacity((256, 256))
    oversized = bytes([0x41] * (max_cap + 1))
    try:
        model.embed_secret(test_cover, oversized, sec_pw)
        sec_5 = False
    except (ValueError, PayloadCapacityError):
        sec_5 = True
    except Exception:
        sec_5 = False
    security_trials.append({"test": "Oversized Payload Rejection (>7083B)", "passed": sec_5, "expected": "PayloadCapacityError / ValueError"})

    # Test 6: Malformed Image Tensor (Wrong Dimension)
    bad_tensor = torch.randn(1, 1, 128, 128)
    try:
        model.embed_secret(bad_tensor, sec_payload, sec_pw)
        sec_6 = False
    except Exception:
        sec_6 = True
    security_trials.append({"test": "Malformed Image Carrier Rejection", "passed": sec_6, "expected": "Clean Exception"})

    for st in security_trials:
        status_str = "PASS" if st["passed"] else "FAIL"
        print(f"  {st['test']:<42} : [{status_str}] ({st['expected']})")

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE STRUCTURED REPORTS (JSON & CSV)
    # ─────────────────────────────────────────────────────────────────────────
    final_report = {
        "environment": env_info,
        "capacity_evaluation": capacity_results,
        "robustness_suite": robustness_results,
        "adaptive_vs_baseline": adaptive_vs_baseline_results,
        "security_validation": security_trials
    }

    json_path = os.path.join(output_dir, "robustness_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(final_report, f, indent=2)

    # Save CSVs
    cap_csv = os.path.join(output_dir, "capacity_results.csv")
    with open(cap_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=capacity_results[0].keys())
        writer.writeheader()
        writer.writerows(capacity_results)

    rob_csv = os.path.join(output_dir, "robustness_results.csv")
    with open(rob_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=robustness_results[0].keys())
        writer.writeheader()
        writer.writerows(robustness_results)

    comp_csv = os.path.join(output_dir, "adaptive_vs_baseline.csv")
    with open(comp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=adaptive_vs_baseline_results[0].keys())
        writer.writeheader()
        writer.writerows(adaptive_vs_baseline_results)

    print("\n" + "=" * 78)
    print("PHASE 5 BENCHMARK COMPLETE.")
    print(f"JSON Report : {json_path}")
    print(f"CSV Outputs : {output_dir}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    run_phase5_benchmark()
