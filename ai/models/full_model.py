"""
Full AI Adaptive Steganography Architecture Container.
Integrates Neural Adaptive Analyzer, U-Net Encoder, and U-Net Decoder into an end-to-end model,
with complete AES-256-GCM authenticated encryption and Reed-Solomon error correction interfaces.
"""

import os
from typing import Tuple, Dict, Any, Union, Optional
import torch
import torch.nn as nn
import numpy as np

from ai.models.adaptive_analyzer import NeuralAdaptiveAnalyzer
from ai.models.encoder import UNetEncoder
from ai.models.decoder import UNetDecoder
from ai.metrics.mse import calculate_mse
from ai.metrics.psnr import calculate_psnr
from ai.metrics.ssim import calculate_ssim

from security.encryption import encrypt_message, decrypt_message, decrypt_bytes, AuthenticationError
from security.payload import (
    encode_payload_to_carrier_map,
    decode_carrier_map_to_payload,
    calculate_usable_capacity,
    PayloadCapacityError,
    CorruptedPayloadError,
    PayloadError
)


class AdaptiveStegoModel(nn.Module):
    """
    End-to-End AI Steganography Model.
    Combines:
    1. Neural Adaptive Analyzer -> spatial suitability map
    2. U-Net Encoder -> stego image
    3. U-Net Decoder -> extracted secret payload
    """

    def __init__(
        self,
        base_dim: int = 32,
        analyzer_checkpoint: Optional[str] = "models/adaptive_analyzer/best_model.pth",
        freeze_analyzer: bool = True,
        scaling_factor: float = 0.15
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.adaptive_analyzer = NeuralAdaptiveAnalyzer(in_channels=3, base_dim=base_dim)

        if analyzer_checkpoint and os.path.exists(analyzer_checkpoint):
            try:
                ckpt = torch.load(analyzer_checkpoint, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.adaptive_analyzer.load_state_dict(ckpt["model_state_dict"])
                elif isinstance(ckpt, dict):
                    self.adaptive_analyzer.load_state_dict(ckpt)
                print(f"[+] Loaded pretrained NeuralAdaptiveAnalyzer from: {analyzer_checkpoint}")
            except Exception as e:
                print(f"[!] Warning: Could not load analyzer checkpoint: {e}")

        if freeze_analyzer:
            for p in self.adaptive_analyzer.parameters():
                p.requires_grad = False
            self.adaptive_analyzer.eval()

        self.encoder = UNetEncoder(in_channels=5, out_channels=3, base_dim=base_dim)
        self.decoder = UNetDecoder(in_channels=3, base_dim=base_dim)

    def forward(
        self,
        cover_image: torch.Tensor,
        payload_map: torch.Tensor,
        external_suitability_map: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through full pipeline.

        Args:
            cover_image: Cover image tensor (B, 3, H, W) in [0, 1]
            payload_map: Payload map tensor (B, 1, H, W) in {0, 1}
            external_suitability_map: Optional handcrafted map override

        Returns:
            Dictionary containing:
            - 'suitability_map': Adaptive suitability map (B, 1, H, W)
            - 'stego_image': Stego image (B, 3, H, W)
            - 'extracted_payload': Extracted secret payload map (B, 1, H, W)
        """
        if external_suitability_map is not None:
            suitability_map = external_suitability_map
        else:
            suitability_map = self.adaptive_analyzer(cover_image)

        stego_image = self.encoder(cover_image, payload_map, suitability_map, scaling_factor=self.scaling_factor)
        extracted_payload = self.decoder(stego_image)

        return {
            "suitability_map": suitability_map,
            "stego_image": stego_image,
            "extracted_payload": extracted_payload
        }

    def embed_secret(
        self,
        cover_image: torch.Tensor,
        secret_payload: Union[str, bytes],
        password: str,
        iterations: int = 100000,
        nsym: int = 32,
        scaling_factor: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Complete Phase 3 Secure Embedding Pipeline:
        1. Encrypts plaintext/binary payload with AES-256-GCM.
        2. Encapsulates encrypted components with Reed-Solomon Forward Error Correction (FEC).
        3. Formats framed bitstream into a spatial carrier map (1, 1, H, W).
        4. Embeds carrier map into cover image guided by Adaptive Region Analyzer.
        5. Computes embedding fidelity metrics (MSE, PSNR, SSIM).

        Args:
            cover_image: Cover image tensor (1, 3, H, W) or (3, H, W) in range [0, 1].
            secret_payload: Secret plaintext string or binary bytes.
            password: User secret password for AES key derivation.
            iterations: PBKDF2 HMAC-SHA256 iterations.
            nsym: Reed-Solomon parity symbols per data block.
            scaling_factor: Optional override for residual perturbation scaling.

        Returns:
            Tuple of (stego_image_tensor, metrics_dict).
        """
        device = next(self.parameters()).device
        sf = scaling_factor if scaling_factor is not None else self.scaling_factor

        # Standardize input dimensions
        if cover_image.ndim == 3:
            cover = cover_image.unsqueeze(0).to(device)
        else:
            cover = cover_image.to(device)

        b, c, h, w = cover.shape
        carrier_shape = (h, w)

        # 1. Encrypt payload with AES-256-GCM
        salt, nonce, ciphertext, tag = encrypt_message(
            secret_payload, password, iterations=iterations
        )

        # 2. Package and encode to carrier map with Reed-Solomon FEC
        carrier_map = encode_payload_to_carrier_map(
            salt, nonce, ciphertext, tag, carrier_shape=carrier_shape, nsym=nsym
        ).to(device)

        # 3. Neural forward pass
        self.eval()
        with torch.no_grad():
            suitability_map = self.adaptive_analyzer(cover)
            stego_image = self.encoder(cover, carrier_map, suitability_map, scaling_factor=sf)

        # 4. Compute metrics
        c_np = cover[0].cpu().numpy()
        s_np = stego_image[0].cpu().numpy()
        mse_val = calculate_mse(c_np, s_np)
        psnr_val = calculate_psnr(c_np, s_np, max_val=1.0)
        ssim_val = calculate_ssim(c_np, s_np, max_val=1.0)

        metrics = {
            "plaintext_bytes": len(secret_payload.encode("utf-8") if isinstance(secret_payload, str) else secret_payload),
            "ciphertext_bytes": len(ciphertext),
            "carrier_bytes": (h * w) // 8,
            "mse": float(mse_val),
            "psnr_db": float(psnr_val),
            "ssim": float(ssim_val),
            "scaling_factor": float(sf)
        }

        return stego_image, metrics

    def extract_secret(
        self,
        stego_image: torch.Tensor,
        password: str,
        as_bytes: bool = False,
        iterations: int = 100000,
        nsym: int = 32,
        threshold: float = 0.5
    ) -> Union[str, bytes]:
        """
        Complete Phase 3 Secure Extraction Pipeline:
        1. Extracts spatial carrier map probabilities from stego image via UNetDecoder.
        2. Decodes carrier map using Reed-Solomon Forward Error Correction (FEC).
        3. Recovers salt, nonce, ciphertext, and authentication tag.
        4. Verifies AES-GCM authentication tag and decrypts payload.

        Args:
            stego_image: Stego image tensor (1, 3, H, W) or (3, H, W).
            password: User secret password for AES key derivation.
            as_bytes: If True, returns raw bytes; if False, decodes UTF-8 string.
            iterations: PBKDF2 HMAC-SHA256 iterations.
            nsym: Reed-Solomon parity symbols.
            threshold: Binarization decision threshold.

        Returns:
            Recovered plaintext message string or binary bytes.

        Raises:
            AuthenticationError: If password is incorrect or authentication tag verification fails.
            CorruptedPayloadError: If channel corruption exceeds Reed-Solomon error correction capability.
        """
        device = next(self.parameters()).device

        if stego_image.ndim == 3:
            stego = stego_image.unsqueeze(0).to(device)
        else:
            stego = stego_image.to(device)

        # 1. Neural decoder forward pass
        self.eval()
        with torch.no_grad():
            extracted_carrier = self.decoder(stego)

        # 2. Decode carrier map with Reed-Solomon FEC
        salt, nonce, ciphertext, tag, version = decode_carrier_map_to_payload(
            extracted_carrier, threshold=threshold, nsym=nsym
        )

        # 3. Authenticate and decrypt with AES-256-GCM
        if as_bytes:
            return decrypt_bytes(salt, nonce, ciphertext, tag, password, iterations=iterations)
        else:
            return decrypt_message(salt, nonce, ciphertext, tag, password, iterations=iterations)


def embed_secret(
    model: AdaptiveStegoModel,
    cover_image: torch.Tensor,
    secret_payload: Union[str, bytes],
    password: str,
    iterations: int = 100000,
    nsym: int = 32,
    scaling_factor: Optional[float] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Module-level helper to embed a secret message into a cover image."""
    return model.embed_secret(
        cover_image, secret_payload, password,
        iterations=iterations, nsym=nsym, scaling_factor=scaling_factor
    )


def extract_secret(
    model: AdaptiveStegoModel,
    stego_image: torch.Tensor,
    password: str,
    as_bytes: bool = False,
    iterations: int = 100000,
    nsym: int = 32,
    threshold: float = 0.5
) -> Union[str, bytes]:
    """Module-level helper to extract and decrypt a secret payload from a stego image."""
    return model.extract_secret(
        stego_image, password, as_bytes=as_bytes,
        iterations=iterations, nsym=nsym, threshold=threshold
    )
