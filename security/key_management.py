"""
Key Management Subsystem for Secure Password and Key Handling.
Handles runtime environment key loading, salt generation, and safe key lifecycle management.
"""

import os
from typing import Optional


def get_environment_key(env_var_name: str = "STEGO_ENCRYPTION_KEY") -> Optional[str]:
    """
    Retrieves the encryption key or password from the environment variable.

    Args:
        env_var_name: Environment variable name.

    Returns:
        Key string if set, otherwise None.
    """
    return os.environ.get(env_var_name, None)


def generate_random_salt(length: int = 16) -> bytes:
    """
    Generates cryptographically secure random salt bytes.

    Args:
        length: Salt length in bytes (default 16).

    Returns:
        Random byte sequence.
    """
    if length < 16:
        raise ValueError("Salt length must be at least 16 bytes for cryptographic security.")
    return os.urandom(length)


def validate_password_strength(password: str) -> bool:
    """
    Basic sanity check for password strength.

    Args:
        password: Password string to evaluate.

    Returns:
        True if password meets minimum length guidelines.
    """
    if not password or len(password) < 6:
        return False
    return True
