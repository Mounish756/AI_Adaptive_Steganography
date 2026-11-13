"""
Unit Tests for Adaptive Image Analysis Module.
"""

import numpy as np
import torch
from ai.models.adaptive_analyzer import HandcraftedAdaptiveAnalyzer, NeuralAdaptiveAnalyzer


def test_handcrafted_adaptive_map_shapes_and_ranges():
    """Tests handcrafted feature maps and fused suitability map shape and bounds."""
    # Create 256x256 RGB image with synthetic edge and texture
    img = np.zeros((256, 256, 3), dtype=np.float32)
    img[100:150, 100:150] = 1.0  # White square in middle creating sharp edges

    analyzer = HandcraftedAdaptiveAnalyzer()
    edge_map, var_map, texture_map, suitability_map = analyzer.generate_suitability_map(img)

    assert edge_map.shape == (256, 256, 1)
    assert var_map.shape == (256, 256, 1)
    assert texture_map.shape == (256, 256, 1)
    assert suitability_map.shape == (256, 256, 1)

    assert suitability_map.min() >= 0.0
    assert suitability_map.max() <= 1.0

    # Ensure edge map highlights the square boundary
    assert suitability_map[100, 100, 0] > 0.0


def test_neural_adaptive_analyzer_forward():
    """Tests neural adaptive analyzer PyTorch forward pass."""
    dummy_input = torch.randn(2, 3, 256, 256)
    model = NeuralAdaptiveAnalyzer()
    output = model(dummy_input)

    assert output.shape == (2, 1, 256, 256)
    assert output.min() >= 0.0
    assert output.max() <= 1.0
