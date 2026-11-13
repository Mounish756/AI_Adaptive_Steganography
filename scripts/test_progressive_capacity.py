import sys, os, time, torch
sys.path.insert(0, ".")
from ai.models.full_model import AdaptiveStegoModel
from ai.data.dataset import SteganographyDataset
from security.payload import calculate_usable_capacity

device = torch.device("cpu")
model = AdaptiveStegoModel(base_dim=32, scaling_factor=0.15).to(device)
ckpt = torch.load("checkpoints/phase2b/phase2b_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

test_ds = SteganographyDataset("datasets/coco_subset", split="test", seed=42)
cover = test_ds[0]["image"].unsqueeze(0).to(device)

test_sizes = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 7083]
password = "CapacityTestPass#2026"

print(f"Testing Progressive Capacity on cover image '{os.path.basename(test_ds[0]['image_path'])}':")
print(f"{'Size (B)':<10} | {'PSNR (dB)':<10} | {'SSIM':<10} | {'Time (s)':<10} | {'Recovery':<12}")
print("-" * 60)

for size in test_sizes:
    payload = b"K" * size
    t0 = time.time()
    try:
        stego, metrics = model.embed_secret(cover, payload, password)
        recovered = model.extract_secret(stego, password, as_bytes=True)
        elapsed = time.time() - t0
        match = (recovered == payload)
        status = "SUCCESS" if match else "MISMATCH"
        print(f"{size:<10} | {metrics['psnr_db']:<10.2f} | {metrics['ssim']:<10.4f} | {elapsed:<10.2f} | {status:<12}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"{size:<10} | {'FAILED':<10} | {'-':<10} | {elapsed:<10.2f} | Error: {e}")
