"""
Phase 5 Balanced Perceptual Fine-Tuning.

Loss configuration:
  alpha = 1.0 (MSE)
  beta = 40.0 (BCE - strong secret recovery constraint)
  ssim_weight = 0.5 (SSIM structural loss)
  edge_weight = 0.2 (edge preservation)
  smooth_weight = 0.3 (smooth-region penalty)

Key goal:
  Improve PSNR to >25.5 dB and SSIM to >0.65 while strictly keeping BER < 0.25%
  so that 100% of payloads up to 7,083B recover with zero errors.
"""

import os
import sys
import time

sys.path.insert(0, ".")
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from ai.models.full_model import AdaptiveStegoModel
from ai.training.losses import PerceptualSteganographyLoss
from ai.data.dataset import SteganographyDataset
from ai.metrics.psnr import calculate_psnr
from ai.metrics.ssim import calculate_ssim
from ai.metrics.mse import calculate_mse

device = torch.device("cpu")
torch.manual_seed(42)
np.random.seed(42)

# Load baseline Phase 2B model
model = AdaptiveStegoModel(base_dim=32, scaling_factor=0.15).to(device)
ckpt = torch.load("checkpoints/phase2b/phase2b_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

# Keep analyzer frozen
for p in model.adaptive_analyzer.parameters():
    p.requires_grad = False
model.adaptive_analyzer.eval()

# Dataloaders
train_ds = SteganographyDataset("datasets/coco_subset", split="train", seed=42)
val_ds   = SteganographyDataset("datasets/coco_subset", split="validation", seed=42)
test_ds  = SteganographyDataset("datasets/coco_subset", split="test", seed=42)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

criterion = PerceptualSteganographyLoss(
    alpha=1.0,
    beta=40.0,         # Strong payload recovery constraint
    ssim_weight=0.5,   # Gentle structural guide
    edge_weight=0.2,   # Edge preservation
    smooth_weight=0.3  # Smooth region penalty
)

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable, lr=3e-5)

print("Starting Balanced Perceptual Fine-Tuning (8 epochs)...")
best_psnr = 0.0

for epoch in range(1, 9):
    t0 = time.time()
    model.train()
    model.adaptive_analyzer.eval()
    
    tr_loss = 0.0
    for batch in train_loader:
        cover = batch["image"].to(device)
        b, _, h, w = cover.shape
        payload = torch.randint(0, 2, (b, 1, h, w), dtype=torch.float32, device=device)
        
        optimizer.zero_grad()
        out = model(cover, payload)
        losses = criterion(cover, out["stego_image"], payload, out["extracted_payload"])
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        tr_loss += losses["total_loss"].item()
    
    # Validation
    model.eval()
    val_psnrs, val_ssims, val_mses, val_bers = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            cover = batch["image"].to(device)
            b, _, h, w = cover.shape
            payload = torch.randint(0, 2, (b, 1, h, w), dtype=torch.float32, device=device)
            out = model(cover, payload)
            for i in range(b):
                c_np = cover[i].cpu().numpy()
                s_np = out["stego_image"][i].cpu().numpy()
                val_psnrs.append(calculate_psnr(c_np, s_np, max_val=1.0))
                val_ssims.append(calculate_ssim(c_np, s_np, max_val=1.0))
                val_mses.append(calculate_mse(c_np, s_np))
            pred = (out["extracted_payload"] > 0.5).float()
            val_bers.append((pred != payload).float().mean().item())
            
    elapsed = time.time() - t0
    psnr_mean = np.mean(val_psnrs)
    ssim_mean = np.mean(val_ssims)
    ber_mean = np.mean(val_bers)
    
    print(f"Epoch {epoch:02d} | TrainLoss={tr_loss/len(train_loader):.4f} | Val PSNR={psnr_mean:.2f}dB | SSIM={ssim_mean:.4f} | BER={ber_mean*100:.3f}% | Time={elapsed:.1f}s")
    
    # Save checkpoint if BER is low and PSNR is good
    if ber_mean < 0.003 and psnr_mean > best_psnr:
        best_psnr = psnr_mean
        os.makedirs("checkpoints/phase5_balanced", exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "psnr": psnr_mean,
            "ssim": ssim_mean,
            "ber": ber_mean
        }, "checkpoints/phase5_balanced/best_model.pth")
        print(f"  --> Saved new best balanced model (PSNR={psnr_mean:.2f}dB, SSIM={ssim_mean:.4f}, BER={ber_mean*100:.3f}%)")

print("\nDone training balanced model.")
