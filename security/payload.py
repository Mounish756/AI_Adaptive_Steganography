"""
Payload Serialization & Binary Stream Conversion Module.
Handles structured header packaging, byte-to-bit stream conversion,
Reed-Solomon Forward Error Correction (FEC), pseudo-random spatial interleaving,
and spatial carrier map encoding/decoding.
"""

import struct
import hashlib
from typing import Tuple, List, Optional
import numpy as np
import torch
import reedsolo

MAGIC_HEADER = b"ASTG"
VERSION = 1
HEADER_FIXED_SIZE = 4 + 1 + 16 + 12 + 16 + 4  # 53 bytes total header

CARRIER_MAGIC = b"AST3"
CARRIER_HDR_DATA_LEN = 7                      # [4B magic | 1B ver | 2B payload_len]
CARRIER_HDR_PARITY = 14                       # 14 RS parity symbols for header
CARRIER_HDR_TOTAL_LEN = CARRIER_HDR_DATA_LEN + CARRIER_HDR_PARITY  # 21 bytes

INTERLEAVER_SEED = 42                         # Standard deterministic interleaver seed


class PayloadError(Exception):
    """Base exception raised for payload packing/unpacking issues."""
    pass


class PayloadCapacityError(PayloadError):
    """Exception raised when payload exceeds available carrier capacity."""
    pass


class CorruptedPayloadError(PayloadError):
    """Exception raised when payload or header corruption exceeds error correction capacity."""
    pass


def get_interleaver_indices(total_bits: int, seed: int = INTERLEAVER_SEED) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a deterministic pseudo-random interleaving permutation and its inverse.
    Interleaving disperses spatial burst errors (e.g. convolution boundary artifacts)
    uniformly across all Reed-Solomon blocks.

    Args:
        total_bits: Number of bits in the carrier map (e.g. 65,536).
        seed: Random seed for permutation reproducibility.

    Returns:
        Tuple of (forward_permutation_indices, inverse_permutation_indices).
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total_bits)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(total_bits)
    return perm, inv_perm


def serialize_payload(
    salt: bytes,
    nonce: bytes,
    ciphertext: bytes,
    tag: bytes,
    version: int = VERSION
) -> bytes:
    """
    Serializes encrypted components and metadata into a binary payload stream.

    Binary Layout:
    [ Magic (4B) | Version (1B) | Salt (16B) | Nonce (12B) | Tag (16B) | Ciphertext Length (4B) | Ciphertext (NB) ]

    Args:
        salt: 16-byte salt
        nonce: 12-byte nonce
        ciphertext: N-byte ciphertext
        tag: 16-byte authentication tag
        version: Payload format version

    Returns:
        Combined serialized byte sequence.
    """
    if len(salt) != 16:
        raise PayloadError(f"Invalid salt length: {len(salt)} (expected 16)")
    if len(nonce) != 12:
        raise PayloadError(f"Invalid nonce length: {len(nonce)} (expected 12)")
    if len(tag) != 16:
        raise PayloadError(f"Invalid tag length: {len(tag)} (expected 16)")

    ciphertext_len = len(ciphertext)
    header = struct.pack(
        ">4sB16s12s16sI",
        MAGIC_HEADER,
        version,
        salt,
        nonce,
        tag,
        ciphertext_len
    )
    return header + ciphertext


def deserialize_payload(serialized: bytes) -> Tuple[bytes, bytes, bytes, bytes, int]:
    """
    Deserializes a binary payload stream back into header metadata and ciphertext.

    Args:
        serialized: Complete serialized payload bytes.

    Returns:
        Tuple of (salt, nonce, ciphertext, tag, version)
    """
    if len(serialized) < HEADER_FIXED_SIZE:
        raise PayloadError("Payload is too short to contain a valid header.")

    header_bytes = serialized[:HEADER_FIXED_SIZE]
    magic, version, salt, nonce, tag, ciphertext_len = struct.unpack(">4sB16s12s16sI", header_bytes)

    if magic != MAGIC_HEADER:
        raise PayloadError(f"Invalid payload magic header: {magic}. Expected {MAGIC_HEADER}.")

    ciphertext = serialized[HEADER_FIXED_SIZE:HEADER_FIXED_SIZE + ciphertext_len]
    if len(ciphertext) != ciphertext_len:
        raise PayloadError(f"Truncated ciphertext: got {len(ciphertext)} bytes, expected {ciphertext_len}.")

    return salt, nonce, ciphertext, tag, version


def bytes_to_bits(data: bytes) -> List[int]:
    """
    Converts a byte sequence into a list of binary bits (0 or 1), MSB first.

    Args:
        data: Byte sequence.

    Returns:
        List of 0 and 1 integer bits.
    """
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    """
    Converts a list of binary bits back into a byte sequence.
    Pads with zeros if bit length is not a multiple of 8.

    Args:
        bits: List of integer bits (0 or 1).

    Returns:
        Converted bytes object.
    """
    remainder = len(bits) % 8
    if remainder != 0:
        bits = bits + [0] * (8 - remainder)

    byte_list = []
    for i in range(0, len(bits), 8):
        byte_val = 0
        for bit in bits[i:i + 8]:
            byte_val = (byte_val << 1) | int(bit)
        byte_list.append(byte_val)
    return bytes(byte_list)


def bits_to_tensor(bits: List[int]) -> torch.Tensor:
    """
    Converts a list of bits into a PyTorch FloatTensor.

    Args:
        bits: List of 0/1 bits.

    Returns:
        1D PyTorch Tensor of shape (len(bits),) with float32 values 0.0 or 1.0.
    """
    return torch.tensor(bits, dtype=torch.float32)


def tensor_to_bits(tensor: torch.Tensor, threshold: float = 0.5) -> List[int]:
    """
    Converts a PyTorch FloatTensor into a list of integer bits (0 or 1).

    Args:
        tensor: PyTorch tensor containing float values.
        threshold: Decision threshold for binarization (default 0.5).

    Returns:
        List of integer bits (0 or 1).
    """
    binary = (tensor > threshold).int().tolist()
    if len(binary) > 0 and isinstance(binary[0], list):
        return [bit for sublist in binary for bit in sublist]
    return binary


def calculate_usable_capacity(
    carrier_shape: Tuple[int, int] = (256, 256),
    nsym: int = 32
) -> int:
    """
    Calculates the maximum usable plaintext/ciphertext payload capacity in bytes
    after accounting for Reed-Solomon parity overhead, framing header, and carrier bounds.

    Carrier: H x W bits = (H * W // 8) total carrier bytes.
    Header: CARRIER_HDR_TOTAL_LEN (21 bytes).
    Remaining: Available for RS-encoded framed payload.
    In RSCodec(nsym=32): 255-byte block holds 223 data bytes (ratio ~ 223 / 255 = 0.8745).
    Header framing: 53 bytes fixed header (MAGIC, version, salt, nonce, tag, length).

    Args:
        carrier_shape: (H, W) carrier spatial dimensions.
        nsym: Number of Reed-Solomon parity symbols per 255-byte block.

    Returns:
        Maximum supported plaintext/ciphertext payload size in bytes.
    """
    total_carrier_bytes = (carrier_shape[0] * carrier_shape[1]) // 8
    avail_for_ecc = total_carrier_bytes - CARRIER_HDR_TOTAL_LEN
    block_size = 255
    data_per_block = block_size - nsym
    num_blocks = avail_for_ecc // block_size
    max_data_bytes = num_blocks * data_per_block
    usable_plaintext_capacity = max_data_bytes - HEADER_FIXED_SIZE
    return max(usable_plaintext_capacity, 0)


def encode_payload_to_carrier_map(
    salt: bytes,
    nonce: bytes,
    ciphertext: bytes,
    tag: bytes,
    carrier_shape: Tuple[int, int] = (256, 256),
    nsym: int = 32,
    interleave: bool = True,
    interleaver_seed: int = INTERLEAVER_SEED
) -> torch.Tensor:
    """
    Encapsulates an AES-GCM encrypted package into a (1, 1, H, W) spatial carrier map
    protected by two-tier Reed-Solomon Forward Error Correction (FEC) and pseudo-random spatial interleaving.

    Structure:
    1. Framed payload: serialize_payload(salt, nonce, ciphertext, tag) -> 53B + len(ciphertext)
    2. Payload ECC: RSCodec(nsym=32) encodes framed payload.
    3. Carrier Header: [CARRIER_MAGIC (4B) | Version (1B) | data_ecc_len (2B)]
       protected by RSCodec(14) -> 21 bytes total header.
    4. Padding: Deterministic pseudo-random whitening padding fills up to 8,192 bytes.
    5. Spatial Interleaving: Scatters bits across carrier plane to break boundary burst errors.
    6. Bitstream conversion: Reshaped to (1, 1, H, W) float32 tensor in {0.0, 1.0}.

    Args:
        salt: 16-byte cryptographic salt.
        nonce: 12-byte cryptographic nonce.
        ciphertext: Encrypted payload bytes.
        tag: 16-byte authentication tag.
        carrier_shape: Target carrier dimensions (default 256x256).
        nsym: Number of RS parity symbols for payload data.
        interleave: If True, applies pseudo-random bit interleaving across the carrier.
        interleaver_seed: Seed for the interleaver permutation.

    Returns:
        PyTorch FloatTensor of shape (1, 1, H, W) with binary values 0.0 or 1.0.

    Raises:
        PayloadCapacityError: If payload exceeds carrier capacity.
    """
    total_carrier_bytes = (carrier_shape[0] * carrier_shape[1]) // 8
    max_capacity = calculate_usable_capacity(carrier_shape, nsym=nsym)

    if len(ciphertext) > max_capacity:
        raise PayloadCapacityError(
            f"Payload length ({len(ciphertext)} bytes) exceeds maximum carrier capacity "
            f"({max_capacity} bytes for {carrier_shape} carrier with nsym={nsym})."
        )

    # 1. Package framed payload
    framed = serialize_payload(salt, nonce, ciphertext, tag, version=VERSION)

    # 2. RS Encode payload
    rsc_data = reedsolo.RSCodec(nsym=nsym)
    data_ecc = bytes(rsc_data.encode(framed))

    # 3. Create carrier header with dedicated RS protection
    rsc_hdr = reedsolo.RSCodec(CARRIER_HDR_PARITY)
    hdr_plain = struct.pack(">4sBH", CARRIER_MAGIC, VERSION, len(data_ecc))
    hdr_ecc = bytes(rsc_hdr.encode(hdr_plain))
    assert len(hdr_ecc) == CARRIER_HDR_TOTAL_LEN

    combined = hdr_ecc + data_ecc
    if len(combined) > total_carrier_bytes:
        raise PayloadCapacityError(f"Encoded payload ({len(combined)} B) exceeds carrier ({total_carrier_bytes} B)")

    # 4. Deterministic whitening padding up to total carrier capacity
    pad = b""
    pad_seed = 0
    while len(combined) + len(pad) < total_carrier_bytes:
        pad += hashlib.sha256(f"stego_whitening_{pad_seed}".encode("utf-8")).digest()
        pad_seed += 1
    carrier_bytes = combined + pad[:total_carrier_bytes - len(combined)]

    # 5. Convert to bitstream and optionally interleave
    bits = bytes_to_bits(carrier_bytes)
    if interleave:
        perm, _ = get_interleaver_indices(len(bits), seed=interleaver_seed)
        bits = np.array(bits, dtype=np.float32)[perm].tolist()

    return torch.tensor(bits, dtype=torch.float32).view(1, 1, carrier_shape[0], carrier_shape[1])


def decode_carrier_map_to_payload(
    carrier_tensor: torch.Tensor,
    threshold: float = 0.5,
    nsym: int = 32,
    interleave: bool = True,
    interleaver_seed: int = INTERLEAVER_SEED
) -> Tuple[bytes, bytes, bytes, bytes, int]:
    """
    Extracts, de-interleaves, and corrects an AES-GCM encrypted package from a neural decoder output tensor.

    Applies:
    1. Binarization of predicted probability map via threshold.
    2. De-interleaving permutation to restore sequential bitstream.
    3. Bitstream to byte reconstruction.
    4. Reed-Solomon decoding of Carrier Header -> recovers valid data_ecc_len.
    5. Reed-Solomon decoding of Payload -> corrects channel bit/byte errors.
    6. Deserialization of framed payload -> returns (salt, nonce, ciphertext, tag, version).

    Args:
        carrier_tensor: Neural output tensor (1, 1, H, W) or (H, W).
        threshold: Binarization decision threshold (default 0.5).
        nsym: RS parity symbols used during encoding.
        interleave: If True, de-interleaves bits before decoding.
        interleaver_seed: Seed for the interleaver permutation.

    Returns:
        Tuple of (salt, nonce, ciphertext, tag, version).

    Raises:
        CorruptedPayloadError: If header or payload errors exceed RS correction capability.
        PayloadError: If header magic is invalid or structure is malformed.
    """
    bits = (carrier_tensor.view(-1) > threshold).int().tolist()

    if interleave:
        _, inv_perm = get_interleaver_indices(len(bits), seed=interleaver_seed)
        bits = np.array(bits, dtype=int)[inv_perm].tolist()

    carrier_bytes = bits_to_bytes(bits)

    if len(carrier_bytes) < CARRIER_HDR_TOTAL_LEN:
        raise CorruptedPayloadError("Carrier stream too short to contain header.")

    # 1. Decode header using dedicated RS codec
    rsc_hdr = reedsolo.RSCodec(CARRIER_HDR_PARITY)
    try:
        hdr_decoded = bytes(rsc_hdr.decode(carrier_bytes[:CARRIER_HDR_TOTAL_LEN])[0])
        magic, ver, data_ecc_len = struct.unpack(">4sBH", hdr_decoded)
    except Exception as e:
        raise CorruptedPayloadError(f"Carrier header corrupted beyond correction capacity: {e}")

    if magic != CARRIER_MAGIC:
        raise CorruptedPayloadError(f"Invalid carrier magic '{magic}'. Expected '{CARRIER_MAGIC}'.")

    if CARRIER_HDR_TOTAL_LEN + data_ecc_len > len(carrier_bytes):
        raise CorruptedPayloadError(
            f"Carrier payload truncated: expected {data_ecc_len} ECC bytes, "
            f"available {len(carrier_bytes) - CARRIER_HDR_TOTAL_LEN} bytes."
        )

    # 2. Decode payload using data RS codec
    data_ecc = carrier_bytes[CARRIER_HDR_TOTAL_LEN:CARRIER_HDR_TOTAL_LEN + data_ecc_len]
    rsc_data = reedsolo.RSCodec(nsym=nsym)
    try:
        framed = bytes(rsc_data.decode(data_ecc)[0])
    except Exception as e:
        raise CorruptedPayloadError(f"Carrier payload corrupted beyond error correction capacity: {e}")

    # 3. Deserialize framed payload
    return deserialize_payload(framed)
