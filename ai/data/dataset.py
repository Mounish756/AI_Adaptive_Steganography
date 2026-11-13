"""
PyTorch Steganography Dataset and Data Loading Module.
Implements leak-free dataset split management (70% train / 15% validation / 15% test),
dynamic image discovery, preprocessing integration, and PyTorch DataLoader utilities.
"""

import os
import hashlib
from typing import List, Tuple, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
import torch

from ai.data.preprocessing import preprocess_pipeline, SUPPORTED_EXTENSIONS


def get_image_files(directory: str) -> List[str]:
    """
    Recursively scans directory for supported image files.

    Args:
        directory: Root path to search.

    Returns:
        Sorted list of absolute image file paths.
    """
    files = []
    if not os.path.exists(directory):
        return files

    for root, _, filenames in os.walk(directory):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(os.path.abspath(os.path.join(root, fname)))
    return sorted(files)


def hash_split_assignment(file_path: str, seed: int = 42) -> str:
    """
    Uses deterministic MD5 hashing to assign a file to 'train', 'validation', or 'test'.
    This guarantees consistent, leak-free partition splits across runs regardless of order.

    Split allocation:
    - 0.00 to 0.70 -> train (70%)
    - 0.70 to 0.85 -> validation (15%)
    - 0.85 to 1.00 -> test (15%)

    Args:
        file_path: File path or basename.
        seed: Random seed for hash salting.

    Returns:
        One of 'train', 'validation', or 'test'.
    """
    key = f"{seed}_{os.path.basename(file_path)}"
    hash_val = hashlib.md5(key.encode("utf-8")).hexdigest()
    # Convert first 8 hex characters to normalized float [0, 1]
    norm_val = int(hash_val[:8], 16) / 0xFFFFFFFF

    if norm_val < 0.70:
        return "train"
    elif norm_val < 0.85:
        return "validation"
    else:
        return "test"


class SteganographyDataset(Dataset):
    """
    PyTorch Dataset for image steganography training, validation, and testing.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        seed: int = 42
    ):
        """
        Args:
            data_dir: Path to directory containing images or explicit split directory.
            split: Target dataset split ('train', 'validation', 'test', or 'all').
            image_size: Target (width, height) tuple.
            seed: Random seed for partition hashing.
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split.lower()
        self.image_size = image_size
        self.seed = seed

        all_files = get_image_files(data_dir)

        if self.split == "all":
            self.image_files = all_files
        else:
            # Filter by deterministic split function or sub-folder matching
            self.image_files = [
                f for f in all_files
                if hash_split_assignment(f, seed=self.seed) == self.split
            ]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.image_files[idx]
        norm_arr, tensor_img = preprocess_pipeline(image_path, target_size=self.image_size)

        return {
            "image": tensor_img,  # PyTorch FloatTensor (3, H, W)
            "image_path": image_path,
            "image_name": os.path.basename(image_path)
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    image_size: Tuple[int, int] = (256, 256),
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test PyTorch DataLoaders.

    Args:
        data_dir: Root dataset directory.
        batch_size: Batch size.
        image_size: Image target dimensions.
        num_workers: DataLoader worker count.
        seed: Partition random seed.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_ds = SteganographyDataset(data_dir, split="train", image_size=image_size, seed=seed)
    val_ds = SteganographyDataset(data_dir, split="validation", image_size=image_size, seed=seed)
    test_ds = SteganographyDataset(data_dir, split="test", image_size=image_size, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
