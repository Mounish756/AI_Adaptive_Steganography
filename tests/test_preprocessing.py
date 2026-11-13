"""
Unit Tests for Image Preprocessing Subsystem.
"""

import numpy as np
import torch
from PIL import Image
import pytest

from ai.data.preprocessing import (
    convert_to_rgb,
    resize_image,
    normalize_image,
    denormalize_image,
    to_tensor,
    to_numpy,
    preprocess_pipeline
)


def test_convert_to_rgb():
    """Tests grayscale and RGBA to RGB conversion."""
    gray_img = Image.new("L", (100, 100), color=128)
    rgb_img = convert_to_rgb(gray_img)
    assert rgb_img.mode == "RGB"

    rgba_img = Image.new("RGBA", (100, 100), color=(200, 100, 50, 255))
    rgb_img2 = convert_to_rgb(rgba_img)
    assert rgb_img2.mode == "RGB"


def test_resize_image():
    """Tests image resizing to 256x256."""
    img = Image.new("RGB", (512, 300), color=(100, 150, 200))
    resized = resize_image(img, (256, 256))
    assert resized.size == (256, 256)


def test_normalization_range():
    """Tests normalized pixel values fall in range [0.0, 1.0] float32."""
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    norm = normalize_image(arr)

    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_denormalization():
    """Tests denormalization back to uint8 [0, 255]."""
    norm_arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    uint8_arr = denormalize_image(norm_arr)

    assert uint8_arr.dtype == np.uint8
    assert list(uint8_arr) == [0, 128, 255]


def test_to_tensor_conversion():
    """Tests conversion from HWC numpy to CHW torch Tensor."""
    hwc = np.ones((256, 256, 3), dtype=np.float32)
    tensor = to_tensor(hwc)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 256, 256)

    recovered_hwc = to_numpy(tensor)
    assert recovered_hwc.shape == (256, 256, 3)


def test_preprocess_pipeline():
    """Tests end-to-end preprocessing pipeline with PIL Image."""
    pil_img = Image.new("RGB", (400, 400), color=(50, 100, 150))
    norm_arr, tensor_img = preprocess_pipeline(pil_img, target_size=(256, 256))

    assert norm_arr.shape == (256, 256, 3)
    assert tensor_img.shape == (3, 256, 256)
    assert tensor_img.dtype == torch.float32
