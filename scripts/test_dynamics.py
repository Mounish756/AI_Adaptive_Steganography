import sys
sys.path.insert(0, ".")
import torch
from ai.models.full_model import AdaptiveStegoModel
from ai.training.losses import SteganographyLoss

torch.manual_seed(42)
m = AdaptiveStegoModel(base_dim=32)
for p in m.adaptive_analyzer.parameters():
    p.requires_grad = False

crit = SteganographyLoss(alpha=1.0, beta=10.0)
opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)

cover = torch.rand(2, 3, 256, 256)
payload = torch.randint(0, 2, (2, 1, 256, 256), dtype=torch.float32)

print("Testing 10 training steps:")
for step in range(10):
    opt.zero_grad()
    out = m(cover, payload)
    losses = crit(cover, out['stego_image'], payload, out['extracted_payload'])
    losses['total_loss'].backward()
    opt.step()
    
    pred_bits = (out['extracted_payload'] > 0.5).float()
    acc = (pred_bits == payload).float().mean().item()
    ber = 1.0 - acc
    mse = losses['img_mse_loss'].item()
    bce = losses['payload_bce_loss'].item()
    print(f"Step {step:02d}: Loss={losses['total_loss'].item():.4f} (MSE={mse:.6f}, BCE={bce:.4f}) | BER={ber*100:.2f}% | Acc={acc*100:.2f}%")
