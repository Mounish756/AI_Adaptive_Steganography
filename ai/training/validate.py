"""
Phase 2A Validation Epoch Module.

Evaluates the model on the validation split after each training epoch.
Reports: Total Loss, Image MSE, Payload BCE, PSNR, Bit Accuracy.

No gradient computation is performed here.
"""

from typing import Dict
import torch
from torch.utils.data import DataLoader

from ai.training.losses import SteganographyLoss
from ai.metrics.psnr import calculate_psnr


def validate_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: SteganographyLoss,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate one validation epoch.

    Tensor shapes expected:
        cover_image       (B, 3, 256, 256)  float32 [0, 1]
        payload_map       (B, 1, 256, 256)  float32 {0, 1}
        stego_image       (B, 3, 256, 256)  float32 [0, 1]
        extracted_payload (B, 1, 256, 256)  float32 [0, 1]

    Args:
        model: AdaptiveStegoModel in eval mode.
        dataloader: Validation DataLoader.
        criterion: SteganographyLoss instance.
        device: Torch device.

    Returns:
        Dict containing average validation metrics for the epoch.
    """
    model.eval()
    running = {
        "total_loss": 0.0,
        "img_mse_loss": 0.0,
        "payload_bce_loss": 0.0,
    }
    psnr_sum = 0.0
    bit_correct = 0
    bit_total = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            cover = batch["image"].to(device)
            b, _, h, w = cover.shape

            payload_map = torch.randint(
                0, 2, (b, 1, h, w), dtype=torch.float32, device=device
            )

            output = model(cover, payload_map)
            stego     = output["stego_image"]
            extracted = output["extracted_payload"]

            losses = criterion(cover, stego, payload_map, extracted)

            for k in running:
                running[k] += losses[k].item()

            # PSNR per image in batch
            for i in range(b):
                psnr_sum += calculate_psnr(
                    cover[i].cpu().numpy(),
                    stego[i].cpu().numpy(),
                    max_val=1.0
                )

            # Bit accuracy
            pred_bits = (extracted > 0.5).float()
            bit_correct += (pred_bits == payload_map).sum().item()
            bit_total   += payload_map.numel()

            n_batches += 1

    if n_batches == 0:
        return {
            "total_loss": 0.0,
            "img_mse_loss": 0.0,
            "payload_bce_loss": 0.0,
            "psnr_db": 0.0,
            "bit_accuracy": 0.0,
        }

    n_images = n_batches * dataloader.batch_size  # approximate

    return {
        "total_loss":       running["total_loss"]       / n_batches,
        "img_mse_loss":     running["img_mse_loss"]     / n_batches,
        "payload_bce_loss": running["payload_bce_loss"] / n_batches,
        "psnr_db":          psnr_sum / max(n_images, 1),
        "bit_accuracy":     bit_correct / max(bit_total, 1),
        "ber":              1.0 - (bit_correct / max(bit_total, 1)),
    }
