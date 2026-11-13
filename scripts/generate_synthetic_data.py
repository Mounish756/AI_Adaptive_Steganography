"""
Synthetic Image Dataset Generator Script.
Generates sample cover images (geometric shapes, gradients, textures) for immediate testing
and places them deterministically into datasets/train, datasets/validation, and datasets/test.
"""

import os
import numpy as np
from PIL import Image, ImageDraw


def create_synthetic_image(image_type: str, size: int = 256) -> Image.Image:
    """
    Creates a synthetic RGB PIL Image of specified type.

    Args:
        image_type: Type of pattern ('gradient', 'circles', 'checkerboard', 'noise').
        size: Image height and width.

    Returns:
        RGB PIL Image instance.
    """
    if image_type == "gradient":
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                arr[i, j, 0] = int((i / size) * 255)
                arr[i, j, 1] = int((j / size) * 255)
                arr[i, j, 2] = int(((i + j) / (2 * size)) * 255)
        return Image.fromarray(arr, mode="RGB")

    elif image_type == "circles":
        img = Image.new("RGB", (size, size), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.ellipse([40, 40, 216, 216], fill=(50, 120, 220), outline=(0, 0, 0), width=4)
        draw.rectangle([80, 80, 176, 176], fill=(220, 80, 50), outline=(0, 0, 0), width=3)
        return img

    elif image_type == "checkerboard":
        grid = 32
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                if ((i // grid) + (j // grid)) % 2 == 0:
                    arr[i, j] = [200, 200, 200]
                else:
                    arr[i, j] = [40, 40, 40]
        return Image.fromarray(arr, mode="RGB")

    elif image_type == "noise":
        np.random.seed(42)
        arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    else:
        # Default gradient
        return create_synthetic_image("gradient", size)


def generate_dataset_samples(dataset_root: str = "datasets", samples_per_split: int = 4):
    """
    Generates synthetic sample images and places them into dataset split folders.

    Args:
        dataset_root: Path to datasets directory.
        samples_per_split: Number of images per split.
    """
    splits = ["train", "validation", "test"]
    patterns = ["gradient", "circles", "checkerboard", "noise"]

    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        os.makedirs(split_dir, exist_ok=True)

        for i in range(samples_per_split):
            pattern_type = patterns[i % len(patterns)]
            img = create_synthetic_image(pattern_type, size=256)
            file_name = f"sample_{pattern_type}_{i+1}.png"
            file_path = os.path.join(split_dir, file_name)
            img.save(file_path)
            print(f"[+] Created synthetic image: {file_path}")


if __name__ == "__main__":
    generate_dataset_samples()
