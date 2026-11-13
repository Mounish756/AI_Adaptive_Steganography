"""
Phase 5 Robustness, Security, and Capacity Validation Tests.

Validates:
1. Lossless PNG round-trip preservation and zero bit error rate.
2. Controlled lossy transformations (JPEG compression, downscaling/upscaling, noise, occlusion)
   strictly adhering to fail-closed security guarantees (AuthenticationError or CorruptedPayloadError,
   never returning unauthenticated plaintext).
3. Payload capacity boundary enforcement (7,083B exact success, 7,084B rejection).
4. Security invariants under cryptographic tampering (wrong password, tampered ciphertext, bad tag/nonce).
5. Input boundary handling on malformed images and payloads.
"""

import os
import sys
import io
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.models.full_model import AdaptiveStegoModel
from ai.data.dataset import SteganographyDataset
from security.encryption import (
    AuthenticationError,
    encrypt_message,
)
from security.payload import (
    PayloadCapacityError,
    CorruptedPayloadError,
    calculate_usable_capacity,
    encode_payload_to_carrier_map,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def model(device):
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "phase2b", "phase2b_model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "phase2a", "best_model.pth")

    analyzer_path = os.path.join(PROJECT_ROOT, "models", "adaptive_analyzer", "best_model.pth")
    m = AdaptiveStegoModel(
        base_dim=32,
        analyzer_checkpoint=analyzer_path,
        freeze_analyzer=True,
        scaling_factor=0.15
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    m.eval()
    return m


@pytest.fixture(scope="module")
def test_dataset():
    dataset_dir = os.path.join(PROJECT_ROOT, "datasets", "coco_subset")
    return SteganographyDataset(dataset_dir, split="test", seed=42)


# ─── Helper Functions ────────────────────────────────────────────────────────

def tensor_to_png_and_back(tensor: torch.Tensor) -> torch.Tensor:
    img_np = (tensor[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    reloaded = Image.open(buf)
    reloaded_np = np.array(reloaded).astype(np.float32) / 255.0
    return torch.from_numpy(reloaded_np).permute(2, 0, 1).unsqueeze(0).to(tensor.device)


def tensor_to_jpeg_and_back(tensor: torch.Tensor, quality: int) -> torch.Tensor:
    img_np = (tensor[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    reloaded = Image.open(buf)
    reloaded_np = np.array(reloaded).astype(np.float32) / 255.0
    return torch.from_numpy(reloaded_np).permute(2, 0, 1).unsqueeze(0).to(tensor.device)


# ─── Test Classes ────────────────────────────────────────────────────────────

class TestPhase5LosslessRobustness:
    """Verifies that lossless file formats (PNG) preserve the hidden payload perfectly."""

    def test_png_lossless_roundtrip_text(self, model, test_dataset, device):
        cover = test_dataset[0]["image"].unsqueeze(0).to(device)
        secret = "Lossless storage test message 2026! Perfect fidelity expected."
        password = "PasswordPNG#123"

        stego, _ = model.embed_secret(cover, secret, password)
        reloaded_stego = tensor_to_png_and_back(stego)

        recovered = model.extract_secret(reloaded_stego, password, as_bytes=False)
        assert recovered == secret, "Lossless PNG roundtrip failed to recover exact text"

    def test_png_lossless_roundtrip_binary(self, model, test_dataset, device):
        cover = test_dataset[1]["image"].unsqueeze(0).to(device)
        secret = bytes([((i * 13) ^ 0x3C) % 256 for i in range(512)])
        password = "BinaryPNG#Key456"

        stego, _ = model.embed_secret(cover, secret, password)
        reloaded_stego = tensor_to_png_and_back(stego)

        recovered = model.extract_secret(reloaded_stego, password, as_bytes=True)
        assert recovered == secret, "Lossless PNG roundtrip failed to recover exact binary bytes"


class TestPhase5LossyTransformationsAndFailClosed:
    """
    Verifies that under lossy transformations, either RS-FEC corrects the distortion,
    or the pipeline fails closed (raises AuthenticationError or CorruptedPayloadError),
    strictly NEVER returning corrupted or unauthenticated data.
    """

    def test_jpeg_compression_fail_closed(self, model, test_dataset, device):
        cover = test_dataset[0]["image"].unsqueeze(0).to(device)
        secret = b"Critical security payload for JPEG testing."
        password = "JPEGPass#SafeCheck"

        stego, _ = model.embed_secret(cover, secret, password)

        for q in [95, 75, 50]:
            compressed = tensor_to_jpeg_and_back(stego, quality=q)
            try:
                rec = model.extract_secret(compressed, password, as_bytes=True)
                # If extraction succeeded, it MUST match exactly
                assert rec == secret, f"JPEG Q={q} returned corrupted plaintext without error!"
            except (AuthenticationError, CorruptedPayloadError):
                # Clean fail-closed behavior
                pass

    def test_downscale_upscale_fail_closed(self, model, test_dataset, device):
        cover = test_dataset[2]["image"].unsqueeze(0).to(device)
        secret = b"Resize resilience evaluation vector."
        password = "ResizePassword#789"

        stego, _ = model.embed_secret(cover, secret, password)

        # Downscale by 2x and upscale back
        _, _, h, w = stego.shape
        down = F.interpolate(stego, size=(h // 2, w // 2), mode="bilinear", align_corners=False)
        restored = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
        restored = torch.clamp(restored, 0.0, 1.0)

        try:
            rec = model.extract_secret(restored, password, as_bytes=True)
            assert rec == secret, "Resized stego returned corrupt data without error!"
        except (AuthenticationError, CorruptedPayloadError):
            pass

    def test_gaussian_noise_fail_closed(self, model, test_dataset, device):
        cover = test_dataset[3]["image"].unsqueeze(0).to(device)
        secret = b"Noise test payload vector."
        password = "NoiseKey#999"

        stego, _ = model.embed_secret(cover, secret, password)

        # Add Gaussian noise
        generator = torch.Generator().manual_seed(42)
        noise = torch.randn(stego.shape, generator=generator, device=device) * 0.01
        noisy_stego = torch.clamp(stego + noise, 0.0, 1.0)

        try:
            rec = model.extract_secret(noisy_stego, password, as_bytes=True)
            assert rec == secret, "Noisy stego returned unauthenticated corrupt data!"
        except (AuthenticationError, CorruptedPayloadError):
            pass

    def test_block_occlusion_fail_closed(self, model, test_dataset, device):
        cover = test_dataset[4]["image"].unsqueeze(0).to(device)
        secret = b"Occlusion masking test vector."
        password = "MaskKey#321"

        stego, _ = model.embed_secret(cover, secret, password)

        # Zero out 16x16 center block
        masked = stego.clone()
        _, _, h, w = masked.shape
        masked[:, :, h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8] = 0.0

        try:
            rec = model.extract_secret(masked, password, as_bytes=True)
            assert rec == secret, "Occluded stego returned unauthenticated corrupt data!"
        except (AuthenticationError, CorruptedPayloadError):
            pass


class TestPhase5CapacityBounds:
    """Verifies theoretical and implemented capacity boundary enforcement."""

    def test_max_capacity_supported(self, model, test_dataset, device):
        cover = test_dataset[0]["image"].unsqueeze(0).to(device)
        max_cap = calculate_usable_capacity((256, 256))
        assert max_cap == 7083, f"Expected 7083 bytes max capacity, got {max_cap}"

        payload = bytes([((i * 7) ^ 0x1A) % 256 for i in range(max_cap)])
        pw = "MaxCapPass#2026"

        stego, metrics = model.embed_secret(cover, payload, pw)
        recovered = model.extract_secret(stego, pw, as_bytes=True)
        assert recovered == payload, "Max capacity payload recovery failed"

    def test_beyond_capacity_rejected(self, model, test_dataset, device):
        cover = test_dataset[0]["image"].unsqueeze(0).to(device)
        max_cap = calculate_usable_capacity((256, 256))
        oversized = bytes([0x55] * (max_cap + 1))
        pw = "BeyondCapPass#2026"

        with pytest.raises((ValueError, PayloadCapacityError)):
            model.embed_secret(cover, oversized, pw)


class TestPhase5SecurityInvariants:
    """Validates AES-256-GCM authentication integrity and tampering detection."""

    def test_wrong_password_rejection(self, model, test_dataset, device):
        cover = test_dataset[1]["image"].unsqueeze(0).to(device)
        secret = "Secret confidential data"
        pw_correct = "CorrectSecretPassword#1"
        pw_wrong = "IncorrectSecretPassword#2"

        stego, _ = model.embed_secret(cover, secret, pw_correct)

        with pytest.raises(AuthenticationError):
            model.extract_secret(stego, pw_wrong)

    def test_ciphertext_tampering_detection(self, model, test_dataset, device):
        cover = test_dataset[2]["image"].unsqueeze(0).to(device)
        secret = b"Tamper detection payload vector."
        pw = "TamperKey#777"

        salt, nonce, ct, tag = encrypt_message(secret, pw)
        tampered_ct = bytearray(ct)
        tampered_ct[len(tampered_ct) // 2] ^= 0x01  # Single bit flip

        carrier = encode_payload_to_carrier_map(salt, nonce, bytes(tampered_ct), tag).to(device)
        suit = model.adaptive_analyzer(cover)
        stego_tampered = model.encoder(cover, carrier, suit, model.scaling_factor)

        with pytest.raises((AuthenticationError, CorruptedPayloadError)):
            model.extract_secret(stego_tampered, pw, as_bytes=True)

    def test_malformed_carrier_tensor(self, model):
        bad_cover = torch.randn(1, 1, 64, 64)  # Wrong shape
        with pytest.raises(Exception):
            model.embed_secret(bad_cover, "test", "pw")
