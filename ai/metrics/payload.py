"""
Payload & Performance Capacity Metric Module.

What it measures:
    - Payload size (bytes and bits)
    - Bits-Per-Pixel (bpp): Embedding density ratio = total payload bits / total image pixels
    - Bit Error Rate (BER): Bit recovery error percentage
    - Execution Time: Embedding and extraction latency in milliseconds

Direction:
    - bpp: Higher means higher embedding capacity
    - BER: Lower is better (0.0 means 100% accurate bit recovery)
    - Execution Time: Lower latency is better

Steganographic Relevance:
    Evaluates the tradeoff between payload capacity, embedding speed, and secret recovery fidelity.
"""

from typing import List, Dict, Any
import time
import numpy as np


def calculate_bpp(payload_bit_count: int, image_shape: tuple) -> float:
    """
    Calculates Bits-Per-Pixel (bpp) embedding density.

    Args:
        payload_bit_count: Total embedded bits.
        image_shape: Tuple of (H, W) or (H, W, C).

    Returns:
        Bits-per-pixel ratio (float).
    """
    h, w = image_shape[0], image_shape[1]
    total_pixels = h * w
    if total_pixels == 0:
        return 0.0
    return float(payload_bit_count / total_pixels)


def calculate_ber(original_bits: List[int], recovered_bits: List[int]) -> float:
    """
    Calculates Bit Error Rate (BER) between original and recovered secret bits.

    Args:
        original_bits: Ground truth list of 0/1 bits.
        recovered_bits: Extracted list of 0/1 bits.

    Returns:
        BER ratio in range [0.0, 1.0].
    """
    if len(original_bits) == 0:
        return 0.0

    min_len = min(len(original_bits), len(recovered_bits))
    errors = sum(b1 != b2 for b1, b2 in zip(original_bits[:min_len], recovered_bits[:min_len]))
    # Add length mismatch errors if any
    errors += abs(len(original_bits) - len(recovered_bits))

    return float(errors / max(len(original_bits), 1))


class Timer:
    """Utility timer for latency benchmarking."""

    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000.0
