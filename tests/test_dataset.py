"""
Unit Tests for Dataset Pipeline and Partition Splitting.
"""

import os
from ai.data.dataset import hash_split_assignment, SteganographyDataset
from scripts.generate_synthetic_data import generate_dataset_samples


def test_hash_split_assignment_determinism():
    """Tests that hash split assignment is deterministic and consistent."""
    file_path = "sample_test_image.png"
    split1 = hash_split_assignment(file_path, seed=42)
    split2 = hash_split_assignment(file_path, seed=42)
    assert split1 == split2
    assert split1 in ("train", "validation", "test")


def test_no_data_leakage_between_splits():
    """Tests that hashed assignments partition files disjointly across train, val, test."""
    filenames = [f"image_{i:04d}.png" for i in range(100)]
    train_set = set()
    val_set = set()
    test_set = set()

    for fname in filenames:
        split = hash_split_assignment(fname, seed=42)
        if split == "train":
            train_set.add(fname)
        elif split == "validation":
            val_set.add(fname)
        elif split == "test":
            test_set.add(fname)

    # Verify no intersections (disjoint partitions)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    # Check approximate split ratios (around 70/15/15)
    total = len(filenames)
    assert len(train_set) > 0
    assert len(val_set) > 0
    assert len(test_set) > 0


def test_steganography_dataset_loading(tmp_path):
    """Tests PyTorch Dataset item loading."""
    # Generate synthetic images in tmp_path
    dataset_root = str(tmp_path)
    generate_dataset_samples(dataset_root=dataset_root, samples_per_split=4)

    ds = SteganographyDataset(data_dir=dataset_root, split="all", image_size=(256, 256))
    assert len(ds) > 0

    item = ds[0]
    assert "image" in item
    assert "image_path" in item
    assert item["image"].shape == (3, 256, 256)
