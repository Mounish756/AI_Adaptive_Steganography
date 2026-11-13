"""
Main CLI Application: AI-Based Adaptive Image Embedding Framework for Secure Information Concealment.

Commands:
  1. embed      : Encrypts a message or file with AES-256-GCM and embeds it into a cover image using the adaptive UNetEncoder.
  2. extract    : Extracts the hidden payload from a stego image using UNetDecoder, verifies AES-GCM tag, and decrypts the secret.
  3. analyze    : Runs the AI Adaptive Region Analyzer on an image to generate its spatial suitability map.
  4. benchmark  : Runs the full empirical benchmark on real held-out test images.
  5. test       : Executes the entire unit and integration test suite (55 tests).

Usage Examples:
  python main.py embed --cover datasets/coco_subset/nat_fruits.jpg --message "Top Secret Message" --password "MyPass123" --output outputs/stego_fruits.png
  python main.py extract --stego outputs/stego_fruits.png --password "MyPass123"
  python main.py analyze --image datasets/coco_subset/nat_fruits.jpg --output outputs/fruits_suitability.png
  python main.py benchmark
  python main.py test
"""

import os
import sys
import argparse
import subprocess
from typing import Optional
from PIL import Image
import numpy as np
import torch

from ai.models.full_model import AdaptiveStegoModel
from ai.data.preprocessing import preprocess_pipeline, to_tensor, to_numpy, normalize_image, denormalize_image
from ai.metrics.psnr import calculate_psnr
from ai.metrics.ssim import calculate_ssim
from ai.metrics.mse import calculate_mse
from security.encryption import AuthenticationError
from security.payload import PayloadCapacityError, CorruptedPayloadError, calculate_usable_capacity


def load_model(device: torch.device, checkpoint_path: Optional[str] = None, scaling_factor: float = 0.13) -> AdaptiveStegoModel:
    """Loads the trained AdaptiveStegoModel with pretrained weights."""
    model = AdaptiveStegoModel(
        base_dim=32,
        analyzer_checkpoint="models/adaptive_analyzer/best_model.pth",
        freeze_analyzer=True,
        scaling_factor=scaling_factor
    ).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt_path = checkpoint_path
    elif os.path.exists("checkpoints/phase5_balanced/best_model.pth"):
        ckpt_path = "checkpoints/phase5_balanced/best_model.pth"
    elif os.path.exists("checkpoints/phase2b/phase2b_model.pth"):
        ckpt_path = "checkpoints/phase2b/phase2b_model.pth"
    else:
        ckpt_path = "checkpoints/phase2a/best_model.pth"

    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        print(f"[+] Loaded model checkpoint from: {ckpt_path}")
    else:
        print(f"[!] Warning: Checkpoint not found at {ckpt_path}. Using initial weights.")

    model.eval()
    return model


def cmd_embed(args):
    """Encrypt and embed secret message or file into a cover image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing model on {device}...")
    model = load_model(device, checkpoint_path=getattr(args, "checkpoint", None), scaling_factor=args.scaling_factor)

    # 1. Load cover image
    if not os.path.exists(args.cover):
        print(f"[ERROR] Cover image not found: {args.cover}")
        sys.exit(1)

    orig_pil = Image.open(args.cover).convert("RGB")
    orig_w, orig_h = orig_pil.size
    orig_np = np.array(orig_pil)

    use_patch = (orig_w >= 256 and orig_h >= 256 and not getattr(args, "force_resize", False))

    if use_patch:
        # Native Resolution Patch Embedding: Preserves 100% full original resolution, sharpness, and details!
        y0 = (orig_h - 256) // 2
        x0 = (orig_w - 256) // 2
        patch_np = orig_np[y0:y0+256, x0:x0+256]
        patch_norm = normalize_image(patch_np)
        cover_tensor = to_tensor(patch_norm).unsqueeze(0).to(device)
    else:
        y0, x0 = 0, 0
        _, cover_tensor = preprocess_pipeline(args.cover, target_size=(256, 256))
        cover_tensor = cover_tensor.unsqueeze(0).to(device)

    # 2. Prepare payload
    if args.file:
        if not os.path.exists(args.file):
            print(f"[ERROR] Secret file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "rb") as f:
            payload = f.read()
        print(f"[*] Loaded binary payload from '{args.file}' ({len(payload)} bytes)")
    elif args.message:
        payload = args.message
        print(f"[*] Message payload: \"{payload}\" ({len(payload.encode('utf-8'))} bytes)")
    else:
        print("[ERROR] Please provide either --message or --file.")
        sys.exit(1)

    if not args.password:
        print("[ERROR] Password is required for AES-256-GCM encryption (--password).")
        sys.exit(1)

    # 3. Embed secret
    print("[*] Encrypting (AES-256-GCM) + Encoding FEC (Reed-Solomon) + Embedding (UNetEncoder)...")
    try:
        stego_tensor, metrics = model.embed_secret(
            cover_tensor, payload, args.password, scaling_factor=args.scaling_factor
        )
    except PayloadCapacityError as e:
        print(f"[ERROR] Capacity error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        sys.exit(1)

    # 4. Save stego image
    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if use_patch:
        stego_patch_np = denormalize_image(to_numpy(stego_tensor[0]))
        full_stego_np = orig_np.copy()
        full_stego_np[y0:y0+256, x0:x0+256] = stego_patch_np
        Image.fromarray(full_stego_np).save(output_path)

        psnr_full = calculate_psnr(orig_np.astype(float)/255.0, full_stego_np.astype(float)/255.0, max_val=1.0)
        ssim_full = calculate_ssim(orig_np.astype(float)/255.0, full_stego_np.astype(float)/255.0, max_val=1.0)
        mse_full  = calculate_mse(orig_np.astype(float)/255.0, full_stego_np.astype(float)/255.0)

        print(f"\n[+] Stego image successfully saved to: {output_path}")
        print(f"    Mode                 : Native Resolution (NO DOWNSAMPLING BLUR!)")
        print(f"    Image Dimensions     : {orig_w}x{orig_h} (100% original sharpness preserved)")
        print(f"    Full Image PSNR      : {psnr_full:.2f} dB")
        print(f"    Full Image SSIM      : {ssim_full:.4f}")
        print(f"    Full Image MSE       : {mse_full:.6f}")
        print(f"    Embedded Patch PSNR  : {metrics['psnr_db']:.2f} dB (SSIM: {metrics['ssim']:.4f})")
        print(f"    Plaintext Size       : {metrics['plaintext_bytes']} bytes")
        print(f"    Ciphertext Size      : {metrics['ciphertext_bytes']} bytes (AES-256-GCM)")
        print(f"    Total Carrier        : {metrics['carrier_bytes']} bytes (256x256)")
    else:
        stego_np = (stego_tensor[0].cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        Image.fromarray(stego_np).save(output_path)
        print(f"\n[+] Stego image successfully saved to: {output_path}")
        print(f"    PSNR: {metrics['psnr_db']:.2f} dB | SSIM: {metrics['ssim']:.4f} | MSE: {metrics['mse']:.6f}")
        print(f"    Plaintext Size : {metrics['plaintext_bytes']} bytes")
        print(f"    Ciphertext Size: {metrics['ciphertext_bytes']} bytes (AES-256-GCM)")
        print(f"    Total Carrier  : {metrics['carrier_bytes']} bytes (256x256)")


def cmd_extract(args):
    """Extract and decrypt secret payload from a stego image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing model on {device}...")
    model = load_model(device, checkpoint_path=getattr(args, "checkpoint", None))

    # 1. Load stego image
    if not os.path.exists(args.stego):
        print(f"[ERROR] Stego image not found: {args.stego}")
        sys.exit(1)

    stego_pil = Image.open(args.stego).convert("RGB")
    sw, sh = stego_pil.size
    stego_np = np.array(stego_pil)

    if sw >= 256 and sh >= 256 and not getattr(args, "force_resize", False):
        y0 = (sh - 256) // 2
        x0 = (sw - 256) // 2
        patch_np = stego_np[y0:y0+256, x0:x0+256]
        stego_tensor = to_tensor(normalize_image(patch_np)).unsqueeze(0).to(device)
    else:
        _, stego_tensor = preprocess_pipeline(args.stego, target_size=(256, 256))
        stego_tensor = stego_tensor.unsqueeze(0).to(device)

    if not args.password:
        print("[ERROR] Password is required for decryption (--password).")
        sys.exit(1)

    # 2. Extract and authenticate
    print("[*] Extracting Carrier Map (UNetDecoder) + Correcting Errors (Reed-Solomon) + Authenticating (AES-256-GCM)...")
    try:
        recovered = model.extract_secret(stego_tensor, args.password, as_bytes=args.as_bytes)
    except AuthenticationError as e:
        print(f"\n[AUTHENTICATION FAILED] {e}")
        print("Possible causes: Incorrect password or active ciphertext tampering.")
        sys.exit(1)
    except CorruptedPayloadError as e:
        print(f"\n[CORRUPTED PAYLOAD] {e}")
        print("The stego image has been severely compressed or tampered beyond recovery.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        sys.exit(1)

    # 3. Output results
    print("\n[+] SUCCESS! AES-256-GCM GHASH Authentication Tag Verified!")
    if args.save_to:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_to)), exist_ok=True)
        with open(args.save_to, "wb" if isinstance(recovered, bytes) else "w", encoding=None if isinstance(recovered, bytes) else "utf-8") as f:
            f.write(recovered)
        print(f"[+] Recovered payload saved to: {args.save_to}")
    else:
        if isinstance(recovered, str):
            print(f"[+] Recovered Plaintext Message:\n    \"{recovered}\"")
        else:
            print(f"[+] Recovered Binary Payload: {len(recovered)} bytes (hex preview: {recovered[:32].hex()}...)")


def cmd_analyze(args):
    """Run Adaptive Region Analyzer to generate and visualize suitability map."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing AdaptiveRegionAnalyzer on {device}...")
    model = load_model(device, checkpoint_path=getattr(args, "checkpoint", None))

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    _, img_tensor = preprocess_pipeline(args.image, target_size=(256, 256))
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        suit_map = model.adaptive_analyzer(img_tensor)[0, 0].cpu().numpy()

    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    suit_img = (suit_map * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(suit_img, mode="L").save(output_path)

    print(f"[+] Adaptive Suitability Map saved to: {output_path}")
    print(f"    Min Suitability : {suit_map.min():.4f}")
    print(f"    Max Suitability : {suit_map.max():.4f}")
    print(f"    Mean Suitability: {suit_map.mean():.4f}")
    print(f"    Std Deviation   : {suit_map.std():.4f}")


def cmd_benchmark(args):
    """Run full Phase 3 benchmark on real test images."""
    print("[*] Running Phase 3 End-to-End Cryptographic Benchmark...")
    subprocess.run([sys.executable, "scripts/run_phase3_evaluation.py"], check=True)


def cmd_test(args):
    """Run complete pytest test suite."""
    print("[*] Running full test suite (pytest tests/ -v)...")
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    if args.test_file:
        cmd = [sys.executable, "-m", "pytest", args.test_file, "-v"]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="AI-Based Adaptive Steganography Framework (AES-256-GCM + UNet + Reed-Solomon)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: embed
    p_embed = subparsers.add_parser("embed", help="Encrypt and embed a secret into a cover image")
    p_embed.add_argument("--cover", required=True, help="Path to cover image (e.g. datasets/coco_subset/nat_fruits.jpg)")
    p_embed.add_argument("--message", default=None, help="Plaintext message string to hide")
    p_embed.add_argument("--file", default=None, help="Path to arbitrary binary file to hide")
    p_embed.add_argument("--password", required=True, help="Secret password for AES-256-GCM encryption")
    p_embed.add_argument("--output", default="outputs/stego_image.png", help="Path to save stego image")
    p_embed.add_argument("--scaling-factor", type=float, default=0.13, help="Residual scaling factor (default: 0.13)")
    p_embed.add_argument("--checkpoint", default=None, help="Optional model checkpoint path")

    # Command: extract
    p_extract = subparsers.add_parser("extract", help="Extract and decrypt a secret from a stego image")
    p_extract.add_argument("--stego", required=True, help="Path to stego image")
    p_extract.add_argument("--password", required=True, help="Secret password for AES-256-GCM decryption")
    p_extract.add_argument("--as-bytes", action="store_true", help="Extract as raw binary bytes instead of string")
    p_extract.add_argument("--save-to", default=None, help="Optional path to save recovered file/text")
    p_extract.add_argument("--checkpoint", default=None, help="Optional model checkpoint path")

    # Command: analyze
    p_analyze = subparsers.add_parser("analyze", help="Generate AI Adaptive Region Suitability Map")
    p_analyze.add_argument("--image", required=True, help="Path to input image")
    p_analyze.add_argument("--output", default="outputs/suitability_map.png", help="Path to save suitability map")
    p_analyze.add_argument("--checkpoint", default=None, help="Optional model checkpoint path")

    # Command: benchmark
    subparsers.add_parser("benchmark", help="Run full Phase 3 benchmark across held-out real test images")

    # Command: test
    p_test = subparsers.add_parser("test", help="Run test suite")
    p_test.add_argument("--test-file", default=None, help="Optional specific test file (e.g. tests/test_phase3_integration.py)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "embed": cmd_embed,
        "extract": cmd_extract,
        "analyze": cmd_analyze,
        "benchmark": cmd_benchmark,
        "test": cmd_test
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
