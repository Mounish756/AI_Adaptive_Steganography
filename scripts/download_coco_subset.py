"""
Download a manageable subset of real natural images from COCO 2017.
Used for Phase 2A.1 real-image validation and secret recovery verification.

Downloads 24 public natural RGB images from COCO 2017 val set.
Total download size: ~3 MB.
"""

import os
import sys
import urllib.request
from PIL import Image

DEST_DIR = "datasets/coco_subset"

# Curated list of valid COCO 2017 validation image IDs
COCO_IDS = [
    139, 285, 632, 724, 776, 785, 802, 872, 885, 1000,
    1268, 1296, 1353, 1422, 1442, 1490, 1503, 1532, 1584, 1676,
    1732, 1737, 1760, 1787
]


def download_coco_subset(dest_dir: str = DEST_DIR):
    os.makedirs(dest_dir, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    downloaded = 0

    print(f"[*] Downloading {len(COCO_IDS)} real COCO 2017 natural images into '{dest_dir}'...")

    for cid in COCO_IDS:
        fname = f"coco_val_{cid:012d}.jpg"
        fpath = os.path.join(dest_dir, fname)

        if os.path.exists(fpath):
            try:
                with Image.open(fpath) as img:
                    img.verify()
                continue
            except Exception:
                pass  # Re-download if corrupt

        url = f"http://images.cocodataset.org/val2017/{cid:012d}.jpg"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            with open(fpath, "wb") as f:
                f.write(data)

            # Verify it's a valid RGB image
            with Image.open(fpath) as img:
                img.verify()

            downloaded += 1
            print(f"  [+] Downloaded: {fname} ({len(data) // 1024} KB)")
        except Exception as e:
            print(f"  [!] Failed to download {fname}: {e}")

    total_images = len([f for f in os.listdir(dest_dir) if f.lower().endswith(('.jpg', '.png'))])
    print(f"[+] COCO subset ready: {total_images} real natural images in '{dest_dir}'.")
    return total_images


if __name__ == "__main__":
    download_coco_subset()
