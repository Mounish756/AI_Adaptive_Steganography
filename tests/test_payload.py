"""
Unit Tests for Payload Serialization and Binary Bit Conversion Subsystem.
"""

import pytest
import torch
import numpy as np

from security.payload import (
    serialize_payload,
    deserialize_payload,
    bytes_to_bits,
    bits_to_bytes,
    bits_to_tensor,
    tensor_to_bits,
    PayloadError,
    MAGIC_HEADER
)


def test_payload_serialization_roundtrip():
    """Tests end-to-end serialization and deserialization of payload fields."""
    salt = b"1234567890123456"
    nonce = b"123456789012"
    tag = b"1234567890123456"
    ciphertext = b"EncryptedSecretDataBytesPayload"

    serialized = serialize_payload(salt, nonce, ciphertext, tag)
    ex_salt, ex_nonce, ex_ct, ex_tag, version = deserialize_payload(serialized)

    assert ex_salt == salt
    assert ex_nonce == nonce
    assert ex_ct == ciphertext
    assert ex_tag == tag
    assert version == 1


def test_invalid_magic_header_fails():
    """Tests that deserialization fails when magic header is invalid."""
    invalid_header = b"BADH" + b"\x01" + (b"\x00" * 48)
    with pytest.raises(PayloadError):
        deserialize_payload(invalid_header)


def test_bytes_to_bits_conversion_roundtrip():
    """Tests bit packing and unpacking precision."""
    original_bytes = b"Hello Steganography 2026!"
    bits = bytes_to_bits(original_bytes)
    recovered_bytes = bits_to_bytes(bits)

    assert len(bits) == len(original_bytes) * 8
    assert recovered_bytes == original_bytes


def test_tensor_conversion():
    """Tests PyTorch tensor bit conversions."""
    bits = [1, 0, 1, 1, 0, 0, 1, 0]
    tensor = bits_to_tensor(bits)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (8,)

    recovered_bits = tensor_to_bits(tensor)
    assert recovered_bits == bits
