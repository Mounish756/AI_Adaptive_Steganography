"""
Neural Steganographic Decoder Module.
Extracts embedded secret payload bits from stego images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ai.models.attention import CBAMBlock


class UNetDecoder(nn.Module):
    """
    CNN Decoder network for payload extraction from stego images.
    Input: Stego Image (B, 3, H, W)
    Output: Extracted payload map (B, 1, H, W) logits/probabilities.
    """

    def __init__(self, in_channels: int = 3, base_dim: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 3, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.LeakyReLU(0.2, inplace=True),
            CBAMBlock(base_dim),

            nn.Conv2d(base_dim, base_dim * 2, 3, padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            CBAMBlock(base_dim * 2),

            nn.Conv2d(base_dim * 2, base_dim * 4, 3, padding=1),
            nn.BatchNorm2d(base_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_dim * 4, base_dim * 2, 3, padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_dim * 2, base_dim, 3, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, stego_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            stego_image: Stego image tensor (B, 3, H, W) in [0, 1].

        Returns:
            Extracted secret payload bit probabilities (B, 1, H, W) in range [0, 1].
        """
        return self.features(stego_image)
