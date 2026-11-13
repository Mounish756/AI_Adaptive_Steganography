"""
Adaptive Suitability Map Visualization Script.
Generates multi-panel comparison grids displaying:
1. Original Cover Image
2. Edge/Gradient Map (Sobel)
3. Local Variance Map
4. Texture/Complexity Map (Laplacian)
5. Final Adaptive Suitability Map

Saves visualization figures to results/adaptive_maps/ with proper labels and colormaps.
"""

import os
import sys

# Ensure project root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import matplotlib.pyplot as plt
import numpy as np

from ai.data.preprocessing import load_image, preprocess_pipeline
from ai.models.adaptive_analyzer import HandcraftedAdaptiveAnalyzer


def visualize_adaptive_features(
    image_path: str,
    output_dir: str = "results/adaptive_maps",
    edge_weight: float = 0.4,
    texture_weight: float = 0.3,
    variance_weight: float = 0.3
):
    """
    Computes and plots the feature map breakdown and fused adaptive suitability map.

    Args:
        image_path: Input cover image path.
        output_dir: Output save directory.
        edge_weight: Edge map weight.
        texture_weight: Texture map weight.
        variance_weight: Local variance weight.
    """
    os.makedirs(output_dir, exist_ok=True)

    norm_arr, tensor_img = preprocess_pipeline(image_path, target_size=(256, 256))

    analyzer = HandcraftedAdaptiveAnalyzer(
        edge_weight=edge_weight,
        texture_weight=texture_weight,
        variance_weight=variance_weight
    )

    edge_map, var_map, texture_map, suitability_map = analyzer.generate_suitability_map(norm_arr)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Adaptive Embedding Region Feature Extraction Breakdown", fontsize=14, fontweight="bold")

    # Panel 1: Original Image
    axes[0].imshow(norm_arr)
    axes[0].set_title("Original Cover Image\n(256 × 256 RGB)", fontsize=10)
    axes[0].axis("off")

    # Panel 2: Edge Map
    im1 = axes[1].imshow(edge_map[:, :, 0], cmap="magma")
    axes[1].set_title(f"1. Edge/Gradient Map\n(Weight: {edge_weight:.2f})", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Variance Map
    im2 = axes[2].imshow(var_map[:, :, 0], cmap="magma")
    axes[2].set_title(f"2. Local Variance Map\n(Weight: {variance_weight:.2f})", fontsize=10)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Texture Map
    im3 = axes[3].imshow(texture_map[:, :, 0], cmap="magma")
    axes[3].set_title(f"3. Texture Complexity\n(Weight: {texture_weight:.2f})", fontsize=10)
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    # Panel 5: Final Adaptive Suitability Map
    im4 = axes[4].imshow(suitability_map[:, :, 0], cmap="viridis")
    axes[4].set_title("Final Adaptive\nSuitability Map [0, 1]", fontsize=10, fontweight="bold")
    axes[4].axis("off")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = os.path.join(output_dir, f"adaptive_map_{base_name}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[+] Saved adaptive suitability map visualization: {output_filename}")
    return output_filename


if __name__ == "__main__":
    # Scan datasets for images to visualize
    sample_images = glob.glob("datasets/*/*.png") + glob.glob("datasets/*/*.jpg")
    if not sample_images:
        from scripts.generate_synthetic_data import generate_dataset_samples
        generate_dataset_samples()
        sample_images = glob.glob("datasets/*/*.png")

    for img_path in sample_images[:3]:
        visualize_adaptive_features(img_path)
