"""
Attention Mechanism Module for AI Steganography Framework.

Implementations:
1. ChannelAttention: Recalibrates inter-channel feature responses via AdaptiveAvgPool and AdaptiveMaxPool + Shared MLP.
2. SpatialAttention: Aggregates channel descriptor along spatial dimensions via Conv2d (7x7 kernel) to generate spatial weight map.
3. CBAMBlock (Convolutional Block Attention Module): Combines Channel Attention and Spatial Attention sequentially.

Purpose:
- Allows the neural network to dynamically focus on spatial regions and feature channels that have high visual masking capacity.
- Refines feature maps prior to upsampling decoder map generation.

Tensor Dimensions:
- Input shape:  (B, C, H, W)
- Output shape: (B, C, H, W)  [Shape preserved, values weighted by attention mask in range [0, 1]]

Scientific Note:
Attention mechanisms guide model focus towards feature-rich regions; they do NOT automatically guarantee statistical security or un-detectability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM).

    Input shape:  (B, C, H, W)
    Output shape: (B, C, 1, 1) attention weights in range [0.0, 1.0]

    What it does:
    Compresses spatial dimensions using both global average pooling and max pooling,
    then computes inter-channel dependencies via a shared Multi-Layer Perceptron (MLP).
    """

    def __init__(self, in_planes: int, reduction_ratio: int = 16):
        super().__init__()
        reduced_planes = max(in_planes // reduction_ratio, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_planes, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM).

    Input shape:  (B, C, H, W)
    Output shape: (B, 1, H, W) spatial attention map in range [0.0, 1.0]

    What it does:
    Aggregates channel information by computing mean and max along the channel axis (dim=1),
    concatenates the two feature maps (2, H, W), and applies a 7x7 spatial convolution.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(pooled)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Input shape:  (B, C, H, W)
    Output shape: (B, C, H, W)

    Combines Channel Attention and Spatial Attention sequentially:
    F' = F * CAM(F)
    F'' = F' * SAM(F')
    """

    def __init__(self, in_planes: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)
        return out
