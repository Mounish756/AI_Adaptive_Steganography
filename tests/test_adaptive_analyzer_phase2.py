"""
Phase 2 Unit Tests for AI-Based Adaptive Region Analyzer, Attention, Features, and Loss Suite.
"""

import pytest
import torch
import numpy as np

from ai.models.attention import ChannelAttention, SpatialAttention, CBAMBlock
from ai.models.adaptive_analyzer import AdaptiveRegionAnalyzer
from ai.data.adaptive_features import extract_supporting_features, generate_reference_suitability_map
from ai.training.losses import AdaptiveAnalyzerLoss, total_variation_loss


def test_attention_modules():
    """Tests Channel Attention, Spatial Attention, and CBAM Block shape preservation."""
    b, c, h, w = 2, 64, 32, 32
    dummy_input = torch.randn(b, c, h, w)

    cam = ChannelAttention(in_planes=c)
    sam = SpatialAttention()
    cbam = CBAMBlock(in_planes=c)

    cam_out = cam(dummy_input)
    sam_out = sam(dummy_input)
    cbam_out = cbam(dummy_input)

    assert cam_out.shape == (b, c, 1, 1)
    assert sam_out.shape == (b, 1, h, w)
    assert cbam_out.shape == (b, c, h, w)
    assert not torch.isnan(cbam_out).any()


def test_adaptive_region_analyzer_architecture():
    """Tests AdaptiveRegionAnalyzer model initialization, forward pass, shape, and bounds."""
    model = AdaptiveRegionAnalyzer(in_channels=3, out_channels=1, base_dim=16)
    param_count = model.count_parameters()

    assert param_count > 0

    dummy_image = torch.rand(2, 3, 256, 256)
    output_map = model(dummy_image)

    # Output dimensions check: [B, 1, 256, 256]
    assert output_map.shape == (2, 1, 256, 256)

    # Range check: 0.0 <= map <= 1.0 (Sigmoid output)
    assert output_map.min() >= 0.0
    assert output_map.max() <= 1.0

    # NaN check
    assert not torch.isnan(output_map).any()


def test_supporting_features_and_reference_map():
    """Tests extraction of Sobel edge, local variance, Laplacian texture, and reference map."""
    synthetic_image = np.zeros((256, 256, 3), dtype=np.float32)
    synthetic_image[50:150, 50:150, :] = 1.0  # White square in center

    edge_map, var_map, texture_map = extract_supporting_features(synthetic_image)
    ref_map = generate_reference_suitability_map(synthetic_image)

    assert edge_map.shape == (256, 256, 1)
    assert var_map.shape == (256, 256, 1)
    assert texture_map.shape == (256, 256, 1)
    assert ref_map.shape == (256, 256, 1)

    assert ref_map.min() >= 0.0
    assert ref_map.max() <= 1.0
    assert not np.isnan(ref_map).any()


def test_adaptive_analyzer_loss():
    """Tests Map Reconstruction MSE loss and Total Variation smoothness loss."""
    pred_map = torch.rand(2, 1, 256, 256, requires_grad=True)
    ref_map = torch.rand(2, 1, 256, 256)

    criterion = AdaptiveAnalyzerLoss(map_weight=1.0, smoothness_weight=0.1)
    losses = criterion(pred_map, ref_map)

    assert "total_loss" in losses
    assert "map_reconstruction_loss" in losses
    assert "smoothness_loss" in losses

    total_loss = losses["total_loss"]
    assert total_loss.item() > 0.0

    # Gradient backward pass test
    total_loss.backward()
    assert pred_map.grad is not None
    assert not torch.isnan(pred_map.grad).any()
