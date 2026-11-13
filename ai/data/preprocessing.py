"""
Image Preprocessing Pipeline Module.
Provides modular functions for loading, validating, resizing, RGB conversion,
normalization [0, 1], and tensor transformation without mutating original disk files.
"""

import os
from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class PreprocessingError(Exception):
    """Exception raised for image preprocessing errors."""
    pass


def validate_image_file(image_path: str) -> bool:
    """
    Validates whether the file exists and has a supported extension.

    Args:
        image_path: Path to the image file.

    Returns:
        True if valid.

    Raises:
        PreprocessingError: If path does not exist or extension is unsupported.
    """
    if not os.path.exists(image_path):
        raise PreprocessingError(f"Image file path does not exist: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise PreprocessingError(
            f"Unsupported image extension: {ext}. Supported formats: {SUPPORTED_EXTENSIONS}"
        )

    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        raise PreprocessingError(f"Corrupted image file {image_path}: {str(e)}")


def load_image(image_path: str) -> Image.Image:
    """
    Loads an image from disk as a PIL Image object.

    Args:
        image_path: Absolute or relative file path.

    Returns:
        PIL Image instance.
    """
    validate_image_file(image_path)
    return Image.open(image_path)


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Ensures image is in 3-channel RGB format.

    Args:
        image: Input PIL Image.

    Returns:
        RGB PIL Image.
    """
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def resize_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (256, 256)
) -> Image.Image:
    """
    Resizes image to target dimensions (width, height) using high-quality Bicubic interpolation.

    Args:
        image: Input PIL Image.
        target_size: Desired (width, height) tuple.

    Returns:
        Resized PIL Image.
    """
    return image.resize(target_size, resample=Image.Resampling.BICUBIC)


def normalize_image(image_array: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel values of a numpy array to float32 in range [0.0, 1.0].

    Args:
        image_array: Input NumPy array (uint8 or float).

    Returns:
        float32 NumPy array with values in [0.0, 1.0].
    """
    if image_array.dtype == np.uint8:
        return (image_array / 255.0).astype(np.float32)
    elif np.issubdtype(image_array.dtype, np.floating):
        # Clip to ensure valid range
        return np.clip(image_array, 0.0, 1.0).astype(np.float32)
    else:
        raise PreprocessingError(f"Unsupported array dtype for normalization: {image_array.dtype}")


def denormalize_image(image_array: np.ndarray) -> np.ndarray:
    """
    Converts a normalized [0, 1] float array back to uint8 [0, 255].

    Args:
        image_array: Input float array in range [0.0, 1.0].

    Returns:
        uint8 NumPy array in range [0, 255].
    """
    clipped = np.clip(image_array, 0.0, 1.0)
    return (clipped * 255.0).round().astype(np.uint8)


def to_tensor(image_array: np.ndarray) -> torch.Tensor:
    """
    Converts a HWC NumPy array to a PyTorch FloatTensor in CHW shape.

    Args:
        image_array: NumPy array of shape (H, W, C) or (H, W).

    Returns:
        PyTorch FloatTensor of shape (C, H, W) or (1, H, W).
    """
    if image_array.ndim == 2:
        image_array = np.expand_dims(image_array, axis=-1)

    # Transpose HWC to CHW
    chw_array = np.transpose(image_array, (2, 0, 1))
    return torch.from_numpy(chw_array).float()


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a CHW PyTorch FloatTensor back to a HWC NumPy array.

    Args:
        tensor: PyTorch tensor of shape (C, H, W) or (1, H, W) or (H, W).

    Returns:
        NumPy array of shape (H, W, C) or (H, W).
    """
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)
    return arr


def preprocess_pipeline(
    image_input: Union[str, Image.Image],
    target_size: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    End-to-end preprocessing pipeline: load -> RGB -> resize -> normalize -> tensor.

    Args:
        image_input: Path string or PIL Image object.
        target_size: Target (width, height) tuple (default 256x256).

    Returns:
        Tuple of (normalized_numpy_HWC, float_tensor_CHW)
    """
    if isinstance(image_input, str):
        img = load_image(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise PreprocessingError(f"Invalid image input type: {type(image_input)}")

    rgb_img = convert_to_rgb(img)
    resized_img = resize_image(rgb_img, target_size)
    np_arr = np.array(resized_img)
    norm_arr = normalize_image(np_arr)
    tensor_img = to_tensor(norm_arr)

    return norm_arr, tensor_img
