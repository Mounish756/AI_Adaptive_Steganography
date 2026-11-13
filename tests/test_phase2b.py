"""
Phase 2B Unit Tests: Adaptive Region Analyzer Integration and Residual Modulation.

Tests:
1. AdaptiveRegionAnalyzer loads and remains frozen (requires_grad is False).
2. Preprocessing, input range [0, 1], and output suitability map range [0, 1].
3. Residual perturbation modulation: verify residual is strictly modulated by suitability_map.
4. Separate checkpoint loading and verification for Phase 2B.
"""

import os
import sys
import pytest
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.models.full_model import AdaptiveStegoModel
from ai.models.adaptive_analyzer import AdaptiveRegionAnalyzer
from ai.models.encoder import UNetEncoder
from ai.models.decoder import UNetDecoder


@pytest.fixture
def device():
    return torch.device("cpu")


def test_analyzer_loads_and_remains_frozen(device):
    """Verify analyzer loads weights and has requires_grad=False."""
    model = AdaptiveStegoModel(
        base_dim=32,
        analyzer_checkpoint="models/adaptive_analyzer/best_model.pth",
        freeze_analyzer=True
    ).to(device)

    # Check analyzer parameters
    for name, param in model.adaptive_analyzer.named_parameters():
        assert not param.requires_grad, f"Parameter {name} should be frozen (requires_grad=False)"

    # Check encoder and decoder are trainable
    encoder_trainable = any(p.requires_grad for p in model.encoder.parameters())
    decoder_trainable = any(p.requires_grad for p in model.decoder.parameters())
    assert encoder_trainable, "Encoder parameters must be trainable"
    assert decoder_trainable, "Decoder parameters must be trainable"


def test_analyzer_input_output_ranges(device):
    """Verify input [0, 1] produces output suitability map in [0, 1] with shape (B, 1, 256, 256)."""
    analyzer = AdaptiveRegionAnalyzer(in_channels=3, out_channels=1, base_dim=32).to(device)
    analyzer.eval()

    cover = torch.rand(2, 3, 256, 256, device=device)
    with torch.no_grad():
        suit_map = analyzer(cover)

    assert suit_map.shape == (2, 1, 256, 256), f"Unexpected shape {suit_map.shape}"
    assert suit_map.min().item() >= 0.0 - 1e-5, "Suitability map values below 0.0"
    assert suit_map.max().item() <= 1.0 + 1e-5, "Suitability map values above 1.0"


def test_encoder_residual_modulated_by_suitability(device):
    """Verify that zero suitability completely suppresses the embedding residual."""
    encoder = UNetEncoder(in_channels=5, out_channels=3, base_dim=16).to(device)
    encoder.eval()

    cover = torch.rand(1, 3, 256, 256, device=device)
    payload = torch.randint(0, 2, (1, 1, 256, 256), dtype=torch.float32, device=device)

    # 1. Zero suitability map
    zero_suit = torch.zeros(1, 1, 256, 256, device=device)
    with torch.no_grad():
        stego_zero = encoder(cover, payload, zero_suit, scaling_factor=0.15)

    # If suitability is 0 everywhere, residual must be 0 everywhere -> stego == cover
    diff_zero = (stego_zero - cover).abs().max().item()
    assert diff_zero < 1e-6, f"Residual was not suppressed by zero suitability (max diff: {diff_zero})"

    # 2. Non-zero suitability map
    ones_suit = torch.ones(1, 1, 256, 256, device=device)
    with torch.no_grad():
        stego_ones = encoder(cover, payload, ones_suit, scaling_factor=0.15)
    diff_ones = (stego_ones - cover).abs().mean().item()
    assert diff_ones > 1e-4, "Non-zero suitability should produce non-zero residual"


def test_phase2b_checkpoints_exist_and_load(device):
    """Verify Phase 2B checkpoints exist under checkpoints/phase2b and reload cleanly."""
    model_path = "checkpoints/phase2b/phase2b_model.pth"
    assert os.path.exists(model_path), f"Phase 2B checkpoint missing: {model_path}"

    model = AdaptiveStegoModel(base_dim=32, scaling_factor=0.15).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    load_res = model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    assert len(load_res.missing_keys) == 0, f"Missing keys when reloading Phase 2B model: {load_res.missing_keys}"
