"""
Unit Tests for Traditional LSB Baseline Steganography.
"""

import numpy as np
from baselines.lsb import lsb_embed, lsb_extract
from security.encryption import encrypt_message, decrypt_message
from security.payload import serialize_payload, deserialize_payload


def test_lsb_embedding_extraction_roundtrip():
    """Tests exact payload embedding and extraction using LSB baseline."""
    # Create cover image (256x256 RGB)
    cover = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    message = "Secret LSB Benchmark Payload 2026!"
    password = "BaselinePassword!"

    salt, nonce, ciphertext, tag = encrypt_message(message, password)
    serialized_payload = serialize_payload(salt, nonce, ciphertext, tag)

    # Embed
    stego, embed_metrics = lsb_embed(cover, serialized_payload)
    assert stego.shape == cover.shape
    assert stego.dtype == np.uint8
    assert embed_metrics["psnr_db"] > 40.0  # LSB 1-bit perturbation PSNR is typically > 50 dB

    # Extract
    extracted_bytes, extract_metrics = lsb_extract(stego)
    assert extracted_bytes == serialized_payload

    # Decrypt
    ex_salt, ex_nonce, ex_ct, ex_tag, version = deserialize_payload(extracted_bytes)
    decrypted = decrypt_message(ex_salt, ex_nonce, ex_ct, ex_tag, password)

    assert decrypted == message
