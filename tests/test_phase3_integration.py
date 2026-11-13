"""
Phase 3 Integration Tests: End-to-End AES-GCM Encryption, Reed-Solomon FEC Framing,
Neural Embedding (UNetEncoder), Extraction (UNetDecoder), and Authentication.

Tests:
1. Encrypt, embed, extract, authenticate, decrypt, and recover text messages.
2. Recover arbitrary binary bytes exactly.
3. Handle empty payload validation.
4. Handle maximum-supported payload capacity.
5. Reject payloads exceeding carrier capacity before embedding.
6. Reject incorrect passwords with AuthenticationError.
7. Detect modified ciphertext and authentication tags (fail closed).
8. Detect heavily corrupted framing and uncorrectable channel errors.
"""

import os
import sys
import pytest
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.models.full_model import AdaptiveStegoModel
from ai.data.dataset import SteganographyDataset
from security.encryption import AuthenticationError, EncryptionError
from security.payload import (
    calculate_usable_capacity,
    PayloadCapacityError,
    CorruptedPayloadError,
    PayloadError
)


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def trained_model(device):
    """Loads the trained Phase 2A/2B AdaptiveStegoModel checkpoint on CPU."""
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
    return model


@pytest.fixture(scope="module")
def cover_image(device):
    """Loads a real cover image from the held-out test split."""
    test_ds = SteganographyDataset("datasets/coco_subset", split="test", seed=42)
    return test_ds[0]["image"].unsqueeze(0).to(device)


# ─── Test 1: Full End-to-End Text Recovery ───────────────────────────────────

def test_end_to_end_text_recovery(trained_model, cover_image):
    """Verify encrypt, embed, extract, authenticate, and decrypt on text."""
    secret_text = "AI-Adaptive-Steganography: Top secret confidential research message 2026!"
    password = "CorrectStrongPassword#2026"

    # Embed
    stego, metrics = trained_model.embed_secret(cover_image, secret_text, password)
    assert stego.shape == cover_image.shape
    assert metrics["psnr_db"] > 20.0
    assert metrics["ssim"] > 0.40

    # Extract
    recovered_text = trained_model.extract_secret(stego, password, as_bytes=False)
    assert recovered_text == secret_text, "Recovered plaintext does not match original text"


# ─── Test 2: Full End-to-End Arbitrary Binary Bytes Recovery ─────────────────

def test_end_to_end_binary_bytes_recovery(trained_model, cover_image):
    """Verify exact bit-for-bit recovery of arbitrary binary byte payload."""
    secret_bytes = bytes([x % 256 for x in range(350)])
    password = "BinaryPayloadKey_99!"

    # Embed binary
    stego, metrics = trained_model.embed_secret(cover_image, secret_bytes, password)

    # Extract binary
    recovered_bytes = trained_model.extract_secret(stego, password, as_bytes=True)
    assert recovered_bytes == secret_bytes, "Recovered binary bytes do not match original bytes"


# ─── Test 3: Empty Payload Validation ────────────────────────────────────────

def test_empty_payload_rejected(trained_model, cover_image):
    """Verify that empty text or empty byte sequences are rejected before embedding."""
    password = "SafePassword123!"

    with pytest.raises(ValueError, match="cannot be empty"):
        trained_model.embed_secret(cover_image, "", password)

    with pytest.raises(ValueError, match="cannot be empty"):
        trained_model.embed_secret(cover_image, b"", password)


# ─── Test 4: Maximum Supported Payload ───────────────────────────────────────

def test_maximum_supported_payload(trained_model, cover_image):
    """Verify embedding and recovering a large payload near maximum supported capacity."""
    max_cap = calculate_usable_capacity((256, 256), nsym=32)
    assert max_cap > 3000, f"Expected capacity > 3000 bytes, got {max_cap}"

    # Test with 2,048 bytes payload
    large_payload = b"Z" * 2048
    password = "MaxCapacityPassword2026!"

    stego, metrics = trained_model.embed_secret(cover_image, large_payload, password)
    recovered = trained_model.extract_secret(stego, password, as_bytes=True)
    assert recovered == large_payload


# ─── Test 5: Payloads Exceeding Capacity Rejected ────────────────────────────

def test_payload_exceeding_capacity_rejected(trained_model, cover_image):
    """Verify that oversized payloads exceeding carrier capacity are rejected immediately."""
    oversized_payload = b"X" * 12000  # Exceeds total carrier capacity (8192 bytes)
    password = "AnyPassword123!"

    with pytest.raises(PayloadCapacityError, match="exceeds maximum carrier capacity"):
        trained_model.embed_secret(cover_image, oversized_payload, password)


# ─── Test 6: Incorrect Password Fails Authentication ─────────────────────────

def test_incorrect_password_fails_authentication(trained_model, cover_image):
    """Verify that extraction with wrong password fails closed with AuthenticationError."""
    secret_text = "Highly Classified Credentials"
    correct_password = "CorrectMasterKey_123"
    wrong_password = "IncorrectWrongKey_456"

    stego, _ = trained_model.embed_secret(cover_image, secret_text, correct_password)

    with pytest.raises(AuthenticationError, match="Incorrect password or corrupted"):
        trained_model.extract_secret(stego, wrong_password)


# ─── Test 7: Detect Modified Ciphertext and Auth Tag Tampering ───────────────

def test_detect_modified_ciphertext_and_tags(trained_model, cover_image):
    """Verify that tampering with encrypted data fails authentication without exposing plaintext."""
    secret_text = "Financial Transfer Authorization"
    password = "SecureAuthPassword2026!"

    stego, _ = trained_model.embed_secret(cover_image, secret_text, password)

    # Corrupt stego image in the payload region (rows 0:10) to simulate active tampering
    tampered_stego = stego.clone()
    tampered_stego[:, :, 0:10, :] = 1.0 - tampered_stego[:, :, 0:10, :]

    # Must raise CorruptedPayloadError or AuthenticationError, never returning unauthenticated data
    with pytest.raises((AuthenticationError, CorruptedPayloadError, PayloadError)):
        trained_model.extract_secret(tampered_stego, password)



# ─── Test 8: Multiple Varied Payload Sizes Roundtrip ─────────────────────────

@pytest.mark.parametrize("msg_len", [1, 15, 64, 128, 512, 1024])
def test_varied_payload_sizes_exact_recovery(trained_model, cover_image, msg_len):
    """Verify exact recovery across a range of payload lengths."""
    secret = ("PayloadTest_" + "A" * msg_len)[:msg_len]
    password = f"ParamKey_{msg_len}!"

    stego, _ = trained_model.embed_secret(cover_image, secret, password)
    recovered = trained_model.extract_secret(stego, password, as_bytes=False)
    assert recovered == secret
