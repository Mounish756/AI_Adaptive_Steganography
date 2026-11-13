"""
Adaptive Suitability Map Visualization & Statistical Profiling Module.

Generates comprehensive 5-panel visualization grids displaying:
1. Original Cover Image
2. Edge / Gradient Map (Sobel)
3. Texture & Local Variance Map
4. Reference Heuristic Suitability Map
5. AI Model Predicted Adaptive Suitability Map

Computes and prints map statistics:
- Min, Max, Mean, Median, Standard Deviation
- Suitability Percentages: Low (0.00-0.33), Medium (0.33-0.66), High (0.66-1.00)
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

from ai.data.preprocessing import preprocess_pipeline
from ai.data.adaptive_features import extract_supporting_features, generate_reference_suitability_map
from ai.models.adaptive_analyzer import AdaptiveRegionAnalyzer


def calculate_map_statistics(suitability_map: np.ndarray) -> Dict[str, float]:
    """
    Calculates statistical properties of a suitability map.

    Args:
        suitability_map: 2D or 3D NumPy array in range [0.0, 1.0].

    Returns:
        Dictionary of computed statistical metrics.
    """
    flat = suitability_map.flatten()
    total_pixels = len(flat)

    low_count = np.sum(flat < 0.33)
    med_count = np.sum((flat >= 0.33) & (flat < 0.66))
    high_count = np.sum(flat >= 0.66)

    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "std": float(np.std(flat)),
        "pct_low": float((low_count / total_pixels) * 100.0),
        "pct_medium": float((med_count / total_pixels) * 100.0),
        "pct_high": float((high_count / total_pixels) * 100.0)
    }


def visualize_adaptive_analyzer_comparison(
    image_path: str,
    model: torch.nn.Module = None,
    output_dir: str = "results/adaptive_analyzer",
    device: str = "cpu"
) -> Tuple[str, Dict[str, float]]:
    """
    Generates 5-panel visual plot comparing original image, features, reference map, and AI prediction.

    Args:
        image_path: Input cover image file path.
        model: Trained AdaptiveRegionAnalyzer PyTorch module (optional).
        output_dir: Output save directory.
        device: PyTorch device ('cpu' or 'cuda').

    Returns:
        Tuple of (output_file_path, map_statistics_dict)
    """
    os.makedirs(output_dir, exist_ok=True)

    norm_arr, tensor_img = preprocess_pipeline(image_path, target_size=(256, 256))
    edge_map, var_map, texture_map = extract_supporting_features(norm_arr)
    ref_map = generate_reference_suitability_map(norm_arr)

    # Predict map using AI model if supplied, otherwise fallback to reference
    if model is not None:
        model.eval()
        with torch.no_grad():
            inp_tensor = tensor_img.unsqueeze(0).to(device)
            pred_tensor = model(inp_tensor)
            ai_pred_map = pred_tensor[0, 0].cpu().numpy()
            ai_pred_map = np.expand_dims(ai_pred_map, axis=-1)
    else:
        ai_pred_map = ref_map

    stats = calculate_map_statistics(ai_pred_map[:, :, 0])

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    fig.suptitle(f"Adaptive Region Analyzer Decomposition — {os.path.basename(image_path)}", fontsize=13, fontweight="bold")

    # 1. Original Image
    axes[0].imshow(norm_arr)
    axes[0].set_title("1. Original Cover\n(256 × 256 RGB)", fontsize=9)
    axes[0].axis("off")

    # 2. Edge Map
    im1 = axes[1].imshow(edge_map[:, :, 0], cmap="magma")
    axes[1].set_title("2. Sobel Edge Map", fontsize=9)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Local Variance & Texture
    combined_tex = (var_map[:, :, 0] + texture_map[:, :, 0]) / 2.0
    im2 = axes[2].imshow(combined_tex, cmap="magma")
    axes[2].set_title("3. Variance & Texture", fontsize=9)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Reference Heuristic Map
    im3 = axes[3].imshow(ref_map[:, :, 0], cmap="viridis")
    axes[3].set_title("4. Reference Map S_ref", fontsize=9)
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    # 5. AI Model Predicted Adaptive Map
    im4 = axes[4].imshow(ai_pred_map[:, :, 0], cmap="viridis")
    axes[4].set_title(f"5. AI Adaptive Map\n(Mean: {stats['mean']:.3f}, High: {stats['pct_high']:.1f}%)", fontsize=9, fontweight="bold")
    axes[4].axis("off")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"adaptive_analysis_{base_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[+] Saved visualization artifact: {output_path}")
    return output_path, stats
