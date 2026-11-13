"""
Phase 2A Unit Tests: UNet Encoder, UNet Decoder, Full Pipeline.

Tests:
1. Encoder output shape and image range
2. Decoder output shape and payload range
3. Full pipeline forward pass (encoder → decoder)
4. Loss calculation and gradient propagation
5. Minimal training smoke test (1 epoch, synthetic batch)
6. Checkpoint save and reload
7. Missing dataset error handling
8. Invalid checkpoint handling
9. CPU execution
"""

import os
import sys
import json
import tempfile
import shutil

import pytest
import torch

# Ensure project root on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.models.encoder import UNetEncoder
from ai.models.decoder import UNetDecoder
from ai.models.full_model import AdaptiveStegoModel
from ai.training.losses import SteganographyLoss
from ai.training.train import train_epoch, save_checkpoint, load_checkpoint, set_seed
from ai.training.validate import validate_epoch


# ─── Fixtures ─────────────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")
B, H, W = 2, 256, 256


@pytest.fixture
def cover():
    return torch.rand(B, 3, H, W)


@pytest.fixture
def payload():
    return torch.randint(0, 2, (B, 1, H, W), dtype=torch.float32)


@pytest.fixture
def suitability():
    return torch.rand(B, 1, H, W)


@pytest.fixture
def encoder():
    return UNetEncoder(in_channels=5, out_channels=3, base_dim=16)


@pytest.fixture
def decoder():
    return UNetDecoder(in_channels=3, base_dim=16)


@pytest.fixture
def full_model():
    return AdaptiveStegoModel(base_dim=16).to(DEVICE)


@pytest.fixture
def criterion():
    return SteganographyLoss(alpha=1.0, beta=2.0)


# ─── Test 1: Encoder output shape and image range ─────────────────────────────

def test_encoder_output_shape_and_range(encoder, cover, payload, suitability):
    """Encoder must output (B, 3, H, W) stego image in [0, 1]."""
    encoder.eval()
    with torch.no_grad():
        stego = encoder(cover, payload, suitability)
    assert stego.shape == (B, 3, H, W), f"Expected (B,3,H,W), got {stego.shape}"
    assert stego.min().item() >= 0.0 - 1e-5, "Stego pixel below 0"
    assert stego.max().item() <= 1.0 + 1e-5, "Stego pixel above 1"


# ─── Test 2: Decoder output shape and payload range ───────────────────────────

def test_decoder_output_shape_and_range(decoder, cover):
    """Decoder must output (B, 1, H, W) payload probabilities in [0, 1]."""
    decoder.eval()
    with torch.no_grad():
        extracted = decoder(cover)
    assert extracted.shape == (B, 1, H, W), f"Expected (B,1,H,W), got {extracted.shape}"
    assert extracted.min().item() >= 0.0 - 1e-5, "Extracted value below 0"
    assert extracted.max().item() <= 1.0 + 1e-5, "Extracted value above 1"


# ─── Test 3: Full pipeline forward pass ───────────────────────────────────────

def test_full_pipeline_forward(full_model, cover, payload):
    """Full model must return correct shapes for all three outputs."""
    cover = cover.to(DEVICE)
    payload = payload.to(DEVICE)
    full_model.eval()
    with torch.no_grad():
        out = full_model(cover, payload)
    assert "suitability_map"   in out, "Missing suitability_map"
    assert "stego_image"       in out, "Missing stego_image"
    assert "extracted_payload" in out, "Missing extracted_payload"

    assert out["suitability_map"].shape   == (B, 1, H, W)
    assert out["stego_image"].shape       == (B, 3, H, W)
    assert out["extracted_payload"].shape == (B, 1, H, W)

    assert out["stego_image"].min() >= 0.0 - 1e-5
    assert out["stego_image"].max() <= 1.0 + 1e-5
    assert out["extracted_payload"].min() >= 0.0 - 1e-5
    assert out["extracted_payload"].max() <= 1.0 + 1e-5


# ─── Test 4: Loss calculation and gradient propagation ────────────────────────

def test_loss_and_gradient(full_model, cover, payload, criterion):
    """Loss must be a finite scalar and gradients must flow to all parameters."""
    cover = cover.to(DEVICE)
    payload = payload.to(DEVICE)

    out = full_model(cover, payload)
    stego     = out["stego_image"]
    extracted = out["extracted_payload"]

    losses = criterion(cover, stego, payload, extracted)
    total_loss = losses["total_loss"]

    assert total_loss.ndim == 0,           "Loss should be a scalar"
    assert torch.isfinite(total_loss),     "Loss is NaN or Inf"
    assert total_loss.item() > 0.0,        "Loss should be positive"

    total_loss.backward()

    # Verify at least some parameters received gradients
    grads_exist = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in full_model.parameters()
    )
    assert grads_exist, "No gradients flowed to model parameters"


# ─── Test 5: Minimal smoke training (1 epoch, tiny fake loader) ───────────────

def test_smoke_training(full_model, criterion):
    """One training epoch must complete without error and reduce or track loss."""
    from torch.utils.data import DataLoader, TensorDataset

    # Tiny fake dataset: 4 images
    fake_images = torch.rand(4, 3, H, W)
    fake_ds = TensorDataset(fake_images)

    class DictWrapper(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, i):
            return {"image": self.ds[i][0]}

    loader = DataLoader(DictWrapper(fake_ds), batch_size=2, shuffle=False)
    optimizer = torch.optim.Adam(full_model.parameters(), lr=1e-4)

    metrics = train_epoch(full_model, loader, optimizer, criterion, DEVICE)

    assert "total_loss"       in metrics
    assert "img_mse_loss"     in metrics
    assert "payload_bce_loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["total_loss"])), "Training produced NaN loss"
    assert metrics["total_loss"] > 0.0


# ─── Test 6: Checkpoint save and reload ───────────────────────────────────────

def test_checkpoint_save_and_reload(full_model, criterion):
    """Saved checkpoint must reload with identical parameter values."""
    tmpdir = tempfile.mkdtemp()
    try:
        optimizer = torch.optim.Adam(full_model.parameters(), lr=1e-4)
        cfg = {"phase2a": {"epochs": 1, "batch_size": 2,
                           "learning_rate": 1e-4, "loss": {"alpha": 1.0, "beta": 2.0}}}

        save_checkpoint(full_model, optimizer, epoch=1, best_val_loss=0.5,
                        cfg=cfg, ckpt_dir=tmpdir, is_best=True)

        assert os.path.exists(os.path.join(tmpdir, "best_model.pth"))
        assert os.path.exists(os.path.join(tmpdir, "best_encoder.pth"))
        assert os.path.exists(os.path.join(tmpdir, "best_decoder.pth"))
        assert os.path.exists(os.path.join(tmpdir, "training_state.pth"))

        # Create a fresh model and reload
        new_model = AdaptiveStegoModel(base_dim=16).to(DEVICE)
        new_opt   = torch.optim.Adam(new_model.parameters(), lr=1e-4)
        start_epoch, bvl = load_checkpoint(
            os.path.join(tmpdir, "best_model.pth"), new_model, new_opt, DEVICE
        )

        assert start_epoch == 2          # epoch + 1
        assert abs(bvl - 0.5) < 1e-6

        # Parameters should match
        for (n1, p1), (n2, p2) in zip(
            full_model.named_parameters(), new_model.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Parameter mismatch: {n1}"
    finally:
        shutil.rmtree(tmpdir)


# ─── Test 7: Missing dataset error handling ───────────────────────────────────

def test_missing_dataset_raises():
    """build_dataloaders must raise FileNotFoundError for a non-existent dataset."""
    import yaml
    from ai.training.train import build_dataloaders

    cfg = {
        "phase2a": {
            "batch_size": 2, "num_workers": 0, "seed": 42
        },
        "paths": {
            "dataset_root": "/nonexistent/dataset/path_xyz_99999",
            "train":      "/nonexistent/dataset/path_xyz_99999/train",
            "validation": "/nonexistent/dataset/path_xyz_99999/validation",
            "test":       "/nonexistent/dataset/path_xyz_99999/test",
        }
    }
    with pytest.raises((FileNotFoundError, RuntimeError)):
        build_dataloaders(cfg, smoke_test=False)


# ─── Test 8: Invalid checkpoint error handling ────────────────────────────────

def test_invalid_checkpoint_raises():
    """load_checkpoint must raise an error for a corrupt/missing checkpoint."""
    model = AdaptiveStegoModel(base_dim=16)
    optimizer = torch.optim.Adam(model.parameters())

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        load_checkpoint("/nonexistent/checkpoint.pth", model, optimizer, DEVICE)

    # File with wrong keys
    tmpdir = tempfile.mkdtemp()
    try:
        bad_path = os.path.join(tmpdir, "bad.pth")
        torch.save({"wrong_key": 123}, bad_path)
        with pytest.raises(ValueError):
            load_checkpoint(bad_path, model, optimizer, DEVICE)
    finally:
        shutil.rmtree(tmpdir)


# ─── Test 9: CPU execution ────────────────────────────────────────────────────

def test_cpu_execution(cover, payload):
    """Full model must run correctly on CPU."""
    model = AdaptiveStegoModel(base_dim=16).to(torch.device("cpu"))
    cover   = cover.cpu()
    payload = payload.cpu()
    model.eval()
    with torch.no_grad():
        out = model(cover, payload)
    assert out["stego_image"].device.type == "cpu"
    assert out["stego_image"].shape == (B, 3, H, W)
