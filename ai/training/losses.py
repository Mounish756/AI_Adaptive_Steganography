"""
Loss Functions Suite for Adaptive Steganography Training.

Implements:
1. AdaptiveAnalyzerLoss  — suitability map training (MSE + TV).
2. SteganographyLoss     — backwards-compatible MSE + BCE.
3. PerceptualSteganographyLoss — Phase 5 quality improvement:
   Combined pixel MSE + SSIM structural loss + edge preservation +
   smooth-region penalty + payload BCE.

Phase 5 loss motivation:
   Pure MSE optimises mean pixel error without regard for perceptual
   quality. It treats every pixel identically — smooth gradients are
   corrupted as much as high-frequency texture regions. Adding an
   SSIM component forces the network to preserve luminance, contrast,
   and local structure. The edge-preservation term adds a gradient
   domain penalty that directly penalises blurring of edges. The
   smooth-region penalty amplifies the MSE signal for flat image
   regions where even tiny changes are perceptually visible.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def total_variation_loss(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes Total Variation (TV) Loss for spatial smoothness regularization.

    Args:
        img_tensor: Tensor of shape (B, C, H, W).

    Returns:
        Scalar TV loss tensor.
    """
    dh = torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :])
    dw = torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1])
    return torch.mean(dh) + torch.mean(dw)


class AdaptiveAnalyzerLoss(nn.Module):
    """
    Loss function for training AdaptiveRegionAnalyzer.
    Combines Map Reconstruction MSE Loss and Total Variation Smoothness Loss.
    """

    def __init__(self, map_weight: float = 1.0, smoothness_weight: float = 0.1):
        """
        Args:
            map_weight: Weight multiplier for map reconstruction MSE loss.
            smoothness_weight: Weight multiplier for TV smoothness regularization.
        """
        super().__init__()
        self.map_weight = map_weight
        self.smoothness_weight = smoothness_weight
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        predicted_map: torch.Tensor,
        reference_map: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward loss computation.

        Args:
            predicted_map: Model output suitability map tensor (B, 1, H, W) in [0.0, 1.0].
            reference_map: Heuristic reference suitability map tensor (B, 1, H, W) in [0.0, 1.0].

        Returns:
            Dictionary containing 'total_loss', 'map_reconstruction_loss', and 'smoothness_loss'.
        """
        rec_loss = self.mse_loss(predicted_map, reference_map)
        tv_loss = total_variation_loss(predicted_map)

        total_loss = self.map_weight * rec_loss + self.smoothness_weight * tv_loss

        return {
            "total_loss": total_loss,
            "map_reconstruction_loss": rec_loss,
            "smoothness_loss": tv_loss
        }


class SteganographyLoss(nn.Module):
    """Backwards-compatible loss for end-to-end stego model training."""

    def __init__(self, alpha: float = 1.0, beta: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(
        self,
        cover_img: torch.Tensor,
        stego_img: torch.Tensor,
        target_payload: torch.Tensor,
        extracted_payload: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        img_mse = self.mse_loss(stego_img, cover_img)
        payload_bce = self.bce_loss(extracted_payload, target_payload)
        total_loss = self.alpha * img_mse + self.beta * payload_bce
        return {
            "total_loss": total_loss,
            "img_mse_loss": img_mse,
            "payload_bce_loss": payload_bce
        }


# ─── Helpers for PerceptualSteganographyLoss ─────────────────────────────────

def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Generates a 1-D Gaussian kernel for differentiable SSIM computation."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    return gauss


def _create_window(window_size: int = 11, num_channels: int = 3) -> torch.Tensor:
    """Creates a 2-D Gaussian window for per-channel SSIM convolution."""
    k1d = _gaussian_kernel(window_size)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)   # (W, W)
    window = k2d.unsqueeze(0).unsqueeze(0)        # (1, 1, W, W)
    return window.expand(num_channels, 1, window_size, window_size).contiguous()


def _ssim_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Differentiable SSIM loss (1 - SSIM), averaged over all pixels.
    Inputs x, y: (B, C, H, W) in [0, 1].
    """
    B, C, H, W = x.shape
    window = _create_window(window_size, C).to(x.device)

    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=C)
    mu_y = F.conv2d(y, window, padding=pad, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y2
    sigma_xy  = F.conv2d(x * y, window, padding=pad, groups=C) - mu_xy

    ssim_map = (
        (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return 1.0 - ssim_map.mean()


def _sobel_gradients(x: torch.Tensor) -> torch.Tensor:
    """
    Computes spatial gradient magnitude of a (B, C, H, W) tensor using
    fixed Sobel filters. Returns gradient magnitude (B, C, H, W).
    """
    kx = torch.tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
        dtype=x.dtype, device=x.device
    ).unsqueeze(0).expand(x.shape[1], 1, 3, 3) / 8.0  # (C, 1, 3, 3)

    ky = torch.tensor(
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
        dtype=x.dtype, device=x.device
    ).unsqueeze(0).expand(x.shape[1], 1, 3, 3) / 8.0

    gx = F.conv2d(x, kx, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, ky, padding=1, groups=x.shape[1])
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


class PerceptualSteganographyLoss(nn.Module):
    """
    Phase 5 Perceptual Steganography Loss.

    Combines four complementary components:

    1. **Pixel MSE** (alpha):
       Penalises per-pixel intensity deviation. Classic baseline term.

    2. **SSIM loss** (ssim_weight):
       Differentiable 1-SSIM computed with an 11×11 Gaussian window.
       Preserves luminance, contrast, and local structure — targets the
       perceptual quality that pure MSE misses.

    3. **Edge preservation loss** (edge_weight):
       L1 loss between the Sobel gradient magnitudes of the stego and
       cover images. Directly penalises blurring or sharpening of edges
       and fine texture details.

    4. **Smooth-region penalty** (smooth_weight):
       Weighted MSE where the per-pixel weight is inversely proportional
       to the Sobel gradient of the cover image. Flat, smooth regions
       receive a higher penalty because human visual sensitivity to
       distortion is highest there.

    5. **Payload BCE** (beta):
       Cross-entropy on extracted vs target bits. Maintains secret-
       recovery accuracy throughout quality improvement training.

    Total = alpha*MSE + ssim_weight*(1-SSIM) + edge_weight*EdgeL1
          + smooth_weight*SmoothMSE + beta*BCE
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 10.0,
        ssim_weight: float = 2.0,
        edge_weight: float = 1.0,
        smooth_weight: float = 1.0,
    ):
        """
        Args:
            alpha:        Pixel MSE weight.
            beta:         Payload BCE weight (secret recovery).
            ssim_weight:  1-SSIM structural loss weight.
            edge_weight:  Gradient-domain edge preservation weight.
            smooth_weight: Smooth-region amplified MSE weight.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.smooth_weight = smooth_weight

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(
        self,
        cover_img: torch.Tensor,
        stego_img: torch.Tensor,
        target_payload: torch.Tensor,
        extracted_payload: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            cover_img:         (B, 3, H, W) in [0, 1]
            stego_img:         (B, 3, H, W) in [0, 1]
            target_payload:    (B, 1, H, W) binary {0, 1}
            extracted_payload: (B, 1, H, W) probabilities in (0, 1)

        Returns:
            Dict with 'total_loss' and all individual component losses.
        """
        # 1. Pixel MSE
        img_mse = self.mse_loss(stego_img, cover_img)

        # 2. Differentiable SSIM loss (1 - SSIM)
        ssim_l = _ssim_loss(stego_img, cover_img)

        # 3. Edge preservation: gradient magnitude L1
        cover_grad = _sobel_gradients(cover_img)
        stego_grad = _sobel_gradients(stego_img)
        edge_l = F.l1_loss(stego_grad, cover_grad)

        # 4. Smooth-region penalty: MSE weighted inversely by cover activity
        #    weight = 1 / (cover_grad + eps)  -> high weight in flat regions
        smooth_weight_map = 1.0 / (cover_grad.detach() + 0.05)
        smooth_weight_map = smooth_weight_map / smooth_weight_map.mean()  # normalise
        diff_sq = (stego_img - cover_img) ** 2
        smooth_l = (smooth_weight_map * diff_sq).mean()

        # 5. Payload BCE
        payload_bce = self.bce_loss(extracted_payload, target_payload)

        total_loss = (
            self.alpha * img_mse
            + self.ssim_weight * ssim_l
            + self.edge_weight * edge_l
            + self.smooth_weight * smooth_l
            + self.beta * payload_bce
        )

        return {
            "total_loss": total_loss,
            "img_mse_loss": img_mse,
            "ssim_loss": ssim_l,
            "edge_loss": edge_l,
            "smooth_region_loss": smooth_l,
            "payload_bce_loss": payload_bce,
        }
