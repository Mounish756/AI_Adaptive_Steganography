"""
AI-Based Adaptive Region Analyzer Neural Network Module.

Implements AdaptiveRegionAnalyzer: a PyTorch CNN model featuring multi-level feature extraction,
CBAM (Channel & Spatial) attention modules, residual skip connections, and an upsampling decoder.

Input:  Cover Image Tensor (B, 3, 256, 256) in range [0.0, 1.0]
Output: Continuous Adaptive Suitability Map (B, 1, 256, 256) scaled to range [0.0, 1.0] via Sigmoid.
"""

from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.models.attention import CBAMBlock
from ai.data.adaptive_features import extract_supporting_features, generate_reference_suitability_map


class ConvBlock(nn.Module):
    """Convolution Block with BatchNorm, GELU, and optional Residual Connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.GELU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv(x) + res
        return self.act(out)


class AdaptiveRegionAnalyzer(nn.Module):
    """
    Trainable AI-Based Adaptive Region Analyzer.

    Architecture Overview:
    1. Initial Conv Block (3 -> base_dim)
    2. Encoder Stage 1 (base_dim -> base_dim * 2, stride 2) + CBAM Attention
    3. Encoder Stage 2 (base_dim * 2 -> base_dim * 4, stride 2) + CBAM Attention
    4. Bottleneck Convolution Block
    5. Decoder Stage 1 (ConvTranspose2d: base_dim * 4 -> base_dim * 2, stride 2)
    6. Decoder Stage 2 (ConvTranspose2d: base_dim * 2 -> base_dim, stride 2)
    7. Final 1x1 Conv (base_dim -> 1)
    8. Sigmoid activation -> Output spatial suitability map in [0.0, 1.0]
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_dim: int = 32):
        """
        Args:
            in_channels: Input image channels (default 3 for RGB).
            out_channels: Output map channels (default 1).
            base_dim: Base feature channels count.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_dim = base_dim

        # Initial Feature Stem
        self.stem = ConvBlock(in_channels, base_dim)

        # Encoder Stage 1
        self.enc1 = ConvBlock(base_dim, base_dim * 2, stride=2)  # Downsample 256 -> 128
        self.att1 = CBAMBlock(base_dim * 2)

        # Encoder Stage 2
        self.enc2 = ConvBlock(base_dim * 2, base_dim * 4, stride=2)  # Downsample 128 -> 64
        self.att2 = CBAMBlock(base_dim * 4)

        # Bottleneck
        self.bottleneck = ConvBlock(base_dim * 4, base_dim * 4)

        # Decoder Stage 1
        self.dec1_up = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=2, stride=2)  # Upsample 64 -> 128
        self.dec1_block = ConvBlock(base_dim * 4, base_dim * 2)  # Concatenated skip connection (base_dim*2 + base_dim*2)

        # Decoder Stage 2
        self.dec2_up = nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=2, stride=2)  # Upsample 128 -> 256
        self.dec2_block = ConvBlock(base_dim * 2, base_dim)  # Concatenated skip connection (base_dim + base_dim)

        # Output Map Generator
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_dim, base_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim // 2, out_channels, kernel_size=1),
            nn.Sigmoid()  # Guarantees output values are strictly within range [0.0, 1.0]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor of shape (B, 3, H, W) in range [0.0, 1.0].

        Returns:
            Adaptive suitability map tensor of shape (B, 1, H, W) in range [0.0, 1.0].
        """
        # Feature Stem -> (B, 32, 256, 256)
        x_stem = self.stem(x)

        # Encoder 1 -> (B, 64, 128, 128)
        e1 = self.att1(self.enc1(x_stem))

        # Encoder 2 -> (B, 128, 64, 64)
        e2 = self.att2(self.enc2(e1))

        # Bottleneck -> (B, 128, 64, 64)
        b = self.bottleneck(e2)

        # Decoder 1 with Skip Connection -> (B, 64, 128, 128)
        d1 = self.dec1_up(b)
        d1 = self.dec1_block(torch.cat([d1, e1], dim=1))

        # Decoder 2 with Skip Connection -> (B, 32, 256, 256)
        d2 = self.dec2_up(d1)
        d2 = self.dec2_block(torch.cat([d2, x_stem], dim=1))

        # Output Suitability Map -> (B, 1, 256, 256)
        map_out = self.final_conv(d2)

        return map_out

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Backwards compatibility alias for Phase 1 Handcrafted Adaptive Analyzer
class HandcraftedAdaptiveAnalyzer:
    def __init__(self, edge_weight: float = 0.4, texture_weight: float = 0.3, variance_weight: float = 0.3):
        self.edge_w = edge_weight
        self.texture_w = texture_weight
        self.variance_w = variance_weight

    def generate_suitability_map(self, image: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        edge_map, var_map, texture_map = extract_supporting_features(image)
        fused = generate_reference_suitability_map(
            image,
            edge_weight=self.edge_w,
            texture_weight=self.texture_w,
            variance_weight=self.variance_w
        )
        return edge_map, var_map, texture_map, fused


# Alias for legacy compatibility
NeuralAdaptiveAnalyzer = AdaptiveRegionAnalyzer
