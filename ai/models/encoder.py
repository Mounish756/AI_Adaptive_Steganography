"""
U-Net Style Neural Encoder Module.
Encodes secret payload bits into cover image guided by spatial suitability map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ai.models.attention import CBAMBlock


class ConvBlock(nn.Module):
    """Basic Double Convolution Block with BatchNorm and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetEncoder(nn.Module):
    """
    Adaptive U-Net Steganographic Encoder.

    Input concatenation:
    - Cover image (3 channels)
    - Payload map (1 channel spatially replicated)
    - Adaptive suitability map (1 channel)
    Total input channels = 5
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 3, base_dim: int = 32):
        super().__init__()
        # Encoder contracting path
        self.enc1 = ConvBlock(in_channels, base_dim)
        self.att1 = CBAMBlock(base_dim)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = ConvBlock(base_dim, base_dim * 2)
        self.att2 = CBAMBlock(base_dim * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = ConvBlock(base_dim * 2, base_dim * 4)
        self.att3 = CBAMBlock(base_dim * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_dim * 4, base_dim * 8)

        # Decoder expanding path
        self.up3 = nn.ConvTranspose2d(base_dim * 8, base_dim * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_dim * 8, base_dim * 4)

        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_dim * 4, base_dim * 2)

        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_dim * 2, base_dim)

        # Residual perturbation generator
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_dim, out_channels, kernel_size=1),
            nn.Tanh()  # Output residual perturbation in [-1, 1], scaled down by residual weight
        )

    def forward(
        self,
        cover_image: torch.Tensor,
        payload_map: torch.Tensor,
        suitability_map: torch.Tensor,
        scaling_factor: float = 0.1
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            cover_image: (B, 3, H, W) in [0, 1]
            payload_map: (B, 1, H, W) binary spatial payload
            suitability_map: (B, 1, H, W) adaptive suitability map in [0, 1]
            scaling_factor: Max residual perturbation amplitude multiplier

        Returns:
            Stego image tensor (B, 3, H, W) clipped to [0, 1].
        """
        # Concatenate inputs along channel dimension -> (B, 5, H, W)
        x_in = torch.cat([cover_image, payload_map, suitability_map], dim=1)

        e1 = self.att1(self.enc1(x_in))
        e2 = self.att2(self.enc2(self.pool1(e1)))
        e3 = self.att3(self.enc3(self.pool2(e2)))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Generate adaptive residual perturbation guided by suitability map
        residual = self.out_conv(d1) * suitability_map * scaling_factor
        stego_image = torch.clamp(cover_image + residual, 0.0, 1.0)

        return stego_image
