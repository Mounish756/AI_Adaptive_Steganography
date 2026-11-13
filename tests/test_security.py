"""
Unit Tests for AES-256-GCM Security and Encryption Subsystem.
"""

import pytest
from security.encryption import (
    encrypt_message,
    decrypt_message,
    AuthenticationError,
    derive_key
)
from security.key_management import validate_password_strength, generate_random_salt


def test_encryption_decryption_success():
    """Tests successful encryption and decryption with correct password."""
    message = "Confidential Payload Test Message 123!"
    password = "StrongPassword2026!"

    salt, nonce, ciphertext, tag = encrypt_message(message, password)

    decrypted = decrypt_message(salt, nonce, ciphertext, tag, password)
    assert decrypted == message


def test_incorrect_password_fails():
    """Tests that decryption fails with AuthenticationError on wrong password."""
    message = "Secret Message"
    password = "CorrectPassword123"
    wrong_password = "WrongPassword456"

    salt, nonce, ciphertext, tag = encrypt_message(message, password)

    with pytest.raises(AuthenticationError):
        decrypt_message(salt, nonce, ciphertext, tag, wrong_password)


def test_corrupted_ciphertext_fails():
    """Tests that tampering with ciphertext triggers tag verification failure."""
    message = "Tamper Test Message"
    password = "Passphrase123!"

    salt, nonce, ciphertext, tag = encrypt_message(message, password)

    # Corrupt a single byte in ciphertext
    corrupted_ct = bytearray(ciphertext)
    corrupted_ct[0] ^= 0xFF

    with pytest.raises(AuthenticationError):
        decrypt_message(salt, nonce, bytes(corrupted_ct), tag, password)


def test_empty_message_validation():
    """Tests that empty messages raise ValueError."""
    password = "Passphrase123!"

    with pytest.raises(ValueError):
        encrypt_message("", password)


def test_different_messages_produce_different_ciphertexts():
    """Tests distinct inputs produce non-identical ciphertexts even with same password."""
    password = "SharedPassword"
    msg1 = "Message A"
    msg2 = "Message B"

    salt1, nonce1, ct1, tag1 = encrypt_message(msg1, password)
    salt2, nonce2, ct2, tag2 = encrypt_message(msg2, password)

    assert ct1 != ct2
    assert nonce1 != nonce2


def test_password_strength_validation():
    """Tests password strength utility checks."""
    assert validate_password_strength("Short") is False
    assert validate_password_strength("ValidLongPassword") is True
