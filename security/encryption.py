"""
Security Encryption Module using AES-256-GCM.
Provides secure authenticated encryption and decryption for steganographic payload data.
Supports both UTF-8 text messages and arbitrary binary byte payloads.
"""

import os
from typing import Tuple, Union
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidTag


class EncryptionError(Exception):
    """Base exception for encryption/decryption errors."""
    pass


class AuthenticationError(EncryptionError):
    """Exception raised when authentication fails (wrong password or corrupted ciphertext/tag)."""
    pass


def derive_key(password: str, salt: bytes, iterations: int = 100000) -> bytes:
    """
    Derives a 256-bit (32-byte) AES key from a user password and salt using PBKDF2-HMAC-SHA256.

    Args:
        password: User password string.
        salt: Cryptographic salt bytes (minimum 16 bytes).
        iterations: Number of PBKDF2 iterations (default 100,000 as per NIST SP 800-132).

    Returns:
        32-byte derived key.
    """
    if not password:
        raise ValueError("Password cannot be empty.")
    if len(salt) < 16:
        raise ValueError("Salt must be at least 16 bytes.")

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations
    )
    return kdf.derive(password.encode("utf-8"))


def encrypt_message(
    message: Union[str, bytes],
    password: str,
    iterations: int = 100000
) -> Tuple[bytes, bytes, bytes, bytes]:
    """
    Encrypts a plaintext message or binary payload using AES-256-GCM.

    Args:
        message: Plaintext message (string or bytes).
        password: User password for key derivation.
        iterations: PBKDF2 iteration count.

    Returns:
        Tuple of (salt, nonce, ciphertext, tag)
    """
    if isinstance(message, str):
        plaintext = message.encode("utf-8")
    else:
        plaintext = bytes(message)

    if len(plaintext) == 0:
        raise ValueError("Message payload cannot be empty.")

    salt = os.urandom(16)
    key = derive_key(password, salt, iterations=iterations)

    # AES-GCM recommended nonce length is 12 bytes; random generation guarantees no nonce reuse
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)

    # AESGCM.encrypt returns ciphertext + 16-byte auth tag
    ct_with_tag = aesgcm.encrypt(nonce, plaintext, associated_data=None)

    ciphertext = ct_with_tag[:-16]
    tag = ct_with_tag[-16:]

    return salt, nonce, ciphertext, tag


def decrypt_bytes(
    salt: bytes,
    nonce: bytes,
    ciphertext: bytes,
    tag: bytes,
    password: str,
    iterations: int = 100000
) -> bytes:
    """
    Decrypts an AES-256-GCM encrypted payload and verifies authenticity.
    Returns decrypted binary bytes directly without string conversion.

    Args:
        salt: Cryptographic salt (16 bytes).
        nonce: Cryptographic nonce (12 bytes).
        ciphertext: Encrypted payload bytes.
        tag: Authentication tag (16 bytes).
        password: User password.
        iterations: PBKDF2 iteration count.

    Returns:
        Decrypted bytes.

    Raises:
        AuthenticationError: If tag verification fails or password is incorrect.
    """
    try:
        key = derive_key(password, salt, iterations=iterations)
        aesgcm = AESGCM(key)

        # Reconstruct combined ciphertext + tag
        ct_with_tag = ciphertext + tag
        return aesgcm.decrypt(nonce, ct_with_tag, associated_data=None)
    except InvalidTag:
        raise AuthenticationError("Decryption failed: Incorrect password or corrupted payload tag.")
    except Exception as e:
        if isinstance(e, AuthenticationError):
            raise
        raise EncryptionError(f"Decryption failed due to unexpected error: {str(e)}")


def decrypt_message(
    salt: bytes,
    nonce: bytes,
    ciphertext: bytes,
    tag: bytes,
    password: str,
    iterations: int = 100000
) -> str:
    """
    Decrypts an AES-256-GCM encrypted payload, verifies authenticity, and decodes to UTF-8.

    Args:
        salt: Cryptographic salt (16 bytes).
        nonce: Cryptographic nonce (12 bytes).
        ciphertext: Encrypted payload bytes.
        tag: Authentication tag (16 bytes).
        password: User password.
        iterations: PBKDF2 iteration count.

    Returns:
        Decrypted UTF-8 string message.

    Raises:
        AuthenticationError: If tag verification fails or password is incorrect.
    """
    raw_bytes = decrypt_bytes(salt, nonce, ciphertext, tag, password, iterations=iterations)
    return raw_bytes.decode("utf-8")
