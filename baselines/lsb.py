"""
Traditional Least Significant Bit (LSB) Steganography Baseline.

IMPORTANT ACADEMIC NOTICE:
This module implements traditional spatial domain 1-bit LSB steganography.
It serves strictly as a baseline benchmark for comparative evaluation against the
proposed AI-based adaptive steganography framework.
DO NOT label or refer to this baseline implementation as AI.
"""

from typing import Tuple, Dict, Any, List
import time
import numpy as np

from security.payload import bytes_to_bits, bits_to_bytes
from ai.metrics.mse import calculate_mse
from ai.metrics.psnr import calculate_psnr
from ai.metrics.ssim import calculate_ssim
from ai.metrics.payload import calculate_bpp, calculate_ber, Timer


def lsb_embed(
    cover_image_uint8: np.ndarray,
    payload_bytes: bytes
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Embeds serialized binary payload bytes into the LSB of cover image pixels (Traditional LSB Baseline).

    Args:
        cover_image_uint8: uint8 RGB NumPy array of shape (H, W, 3).
        payload_bytes: Binary serialized payload (includes header + ciphertext).

    Returns:
        Tuple of (stego_image_uint8, benchmark_metrics_dict)
    """
    if cover_image_uint8.dtype != np.uint8:
        raise ValueError("Cover image must be uint8 array [0, 255].")

    payload_bits = bytes_to_bits(payload_bytes)
    # Append 32-bit payload length prefix to know exact bit length during extraction
    len_bits = bytes_to_bits(len(payload_bits).to_bytes(4, byteorder="big"))
    full_bitstream = len_bits + payload_bits

    total_capacity_bits = cover_image_uint8.size
    if len(full_bitstream) > total_capacity_bits:
        raise ValueError(
            f"Payload length ({len(full_bitstream)} bits) exceeds image capacity ({total_capacity_bits} bits)."
        )

    stego_image = cover_image_uint8.copy()
    flat_pixels = stego_image.reshape(-1)

    with Timer() as timer:
        for i, bit in enumerate(full_bitstream):
            # Clear LSB (mask 0xFE = 254) and insert payload bit
            flat_pixels[i] = (flat_pixels[i] & np.uint8(0xFE)) | np.uint8(bit)

    stego_image = flat_pixels.reshape(cover_image_uint8.shape)

    # Calculate baseline evaluation metrics
    norm_cover = cover_image_uint8.astype(np.float32) / 255.0
    norm_stego = stego_image.astype(np.float32) / 255.0

    mse_val = calculate_mse(norm_cover, norm_stego)
    psnr_val = calculate_psnr(norm_cover, norm_stego, max_val=1.0)
    ssim_val = calculate_ssim(norm_cover, norm_stego, max_val=1.0)
    bpp_val = calculate_bpp(len(full_bitstream), cover_image_uint8.shape)

    metrics = {
        "method": "Traditional LSB Baseline",
        "embedding_time_ms": timer.elapsed_ms,
        "payload_bytes": len(payload_bytes),
        "total_embedded_bits": len(full_bitstream),
        "bpp": bpp_val,
        "mse": mse_val,
        "psnr_db": psnr_val,
        "ssim": ssim_val
    }

    return stego_image, metrics


def lsb_extract(stego_image_uint8: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
    """
    Extracts LSB embedded binary payload bytes from a stego image (Traditional LSB Baseline).

    Args:
        stego_image_uint8: uint8 RGB NumPy array of shape (H, W, 3).

    Returns:
        Tuple of (extracted_payload_bytes, extraction_metrics_dict)
    """
    if stego_image_uint8.dtype != np.uint8:
        raise ValueError("Stego image must be uint8 array [0, 255].")

    flat_pixels = stego_image_uint8.reshape(-1)

    with Timer() as timer:
        # Step 1: Extract 32-bit length prefix (4 bytes = 32 bits)
        len_bits = [(flat_pixels[i] & 1) for i in range(32)]
        payload_len_bits = int.from_bytes(bits_to_bytes(len_bits), byteorder="big")

        if payload_len_bits <= 0 or payload_len_bits > (len(flat_pixels) - 32):
            raise ValueError(f"Invalid extracted payload bit length: {payload_len_bits}")

        # Step 2: Extract exact payload bits
        extracted_bits = [(flat_pixels[32 + i] & 1) for i in range(payload_len_bits)]
        extracted_bytes = bits_to_bytes(extracted_bits)

    metrics = {
        "method": "Traditional LSB Baseline",
        "extraction_time_ms": timer.elapsed_ms,
        "extracted_payload_bytes": len(extracted_bytes),
        "extracted_bits_count": payload_len_bits
    }

    return extracted_bytes, metrics
