"""
Traditional LSB Baseline Experiment Execution Script.

Runs complete pipeline:
Secret Text -> AES-256-GCM Encryption -> Serialization -> LSB Embedding -> Stego Image
-> LSB Extraction -> Deserialization -> AES-256-GCM Decryption -> Decoded Text.

Measures and logs: MSE, PSNR (dB), SSIM, Bits-Per-Pixel (bpp), embedding & extraction latency.
"""

import os
import sys

# Ensure project root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import numpy as np
from PIL import Image

from security.encryption import encrypt_message, decrypt_message
from security.payload import serialize_payload, deserialize_payload
from baselines.lsb import lsb_embed, lsb_extract
from ai.data.preprocessing import load_image, convert_to_rgb, resize_image


def run_lsb_experiment(
    image_path: str,
    secret_text: str = "CONFIDENTIAL: Final Year Academic Research Project Data Payload 2026",
    password: str = "SecureResearchPassphrase2026!"
):
    """
    Runs end-to-end baseline LSB experiment on a cover image.

    Args:
        image_path: Path to cover image file.
        secret_text: Plaintext secret message.
        password: Encryption password.
    """
    print(f"\n==================================================")
    print(f"TRADITIONAL LSB BASELINE EXPERIMENT BENCHMARK")
    print(f"Target Cover Image: {image_path}")
    print(f"Secret Plaintext Length: {len(secret_text)} characters")
    print(f"==================================================")

    # Step 1: Preprocess Cover Image
    pil_img = load_image(image_path)
    rgb_img = convert_to_rgb(pil_img)
    resized_img = resize_image(rgb_img, (256, 256))
    cover_np_uint8 = np.array(resized_img)

    # Step 2: AES-256-GCM Encryption
    salt, nonce, ciphertext, tag = encrypt_message(secret_text, password)
    serialized_payload = serialize_payload(salt, nonce, ciphertext, tag)

    print(f"[1] AES-256-GCM Encrypted & Serialized Payload Size: {len(serialized_payload)} bytes ({len(serialized_payload) * 8} bits)")

    # Step 3: LSB Embedding
    stego_np_uint8, embed_metrics = lsb_embed(cover_np_uint8, serialized_payload)

    print(f"[2] LSB Embedding Complete:")
    print(f"    - Embedding Latency: {embed_metrics['embedding_time_ms']:.2f} ms")
    print(f"    - Bits-Per-Pixel (bpp): {embed_metrics['bpp']:.4f}")
    print(f"    - Mean Squared Error (MSE): {embed_metrics['mse']:.8f}")
    print(f"    - Peak Signal-to-Noise Ratio (PSNR): {embed_metrics['psnr_db']:.2f} dB")
    print(f"    - Structural Similarity Index (SSIM): {embed_metrics['ssim']:.6f}")

    # Step 4: LSB Extraction
    extracted_bytes, extract_metrics = lsb_extract(stego_np_uint8)

    print(f"[3] LSB Extraction Complete:")
    print(f"    - Extraction Latency: {extract_metrics['extraction_time_ms']:.2f} ms")
    print(f"    - Extracted Payload Size: {len(extracted_bytes)} bytes")

    # Step 5: Payload Deserialization and AES-256-GCM Decryption
    ex_salt, ex_nonce, ex_ciphertext, ex_tag, version = deserialize_payload(extracted_bytes)
    decrypted_text = decrypt_message(ex_salt, ex_nonce, ex_ciphertext, ex_tag, password)

    print(f"[4] AES-256-GCM Decryption Result:")
    print(f"    - Recovered Secret Message: '{decrypted_text}'")

    assert decrypted_text == secret_text, "ERROR: Recovered text does not match original secret text!"
    print(f"[SUCCESS] Payload perfectly recovered without errors (Bit Error Rate = 0.00%).")


if __name__ == "__main__":
    sample_images = glob.glob("datasets/*/*.png") + glob.glob("datasets/*/*.jpg")
    if not sample_images:
        from scripts.generate_synthetic_data import generate_dataset_samples
        generate_dataset_samples()
        sample_images = glob.glob("datasets/*/*.png")

    if sample_images:
        run_lsb_experiment(sample_images[0])
