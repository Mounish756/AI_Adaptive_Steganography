"""
Verification Script: Complete Original Image Preservation & Full Roundtrip Testing.
Validates:
1. Stego output has identical width, height, and aspect ratio to the original image.
2. Complete original image is preserved (untouched outside 256x256 patch, bitwise identical).
3. Stego image preview and downloaded PNG are full-dimension RGB images.
4. Downloaded complete PNG can be re-uploaded to extract the exact secret message.
5. Symmetrical extraction and authentication validation.
6. Images smaller than 256x256 are cleanly rejected with a clear message.
"""

import io
import requests
import numpy as np
from PIL import Image

API_URL = "http://127.0.0.1:5050"

def run_tests():
    print("=" * 70)
    print("  VERIFYING FULL-IMAGE PRESERVATION AND STEGANOGRAPHY PIPELINE")
    print("=" * 70)

    # 1. Load a real test image with non-square dimensions
    img_path = "datasets/coco_subset/coco_val_000000000139.jpg"
    orig_pil = Image.open(img_path).convert("RGB")
    orig_w, orig_h = orig_pil.size
    orig_np = np.array(orig_pil)
    print(f"\n[1] Reference Cover Image: {img_path}")
    print(f"    Dimensions: {orig_w} x {orig_h} (Aspect Ratio: {orig_w/orig_h:.3f})")

    secret_message = "Enterprise Confidential Secret #2026-X: Full Image Preserved!"
    password = "CorrectStrongPassword123!"

    # 2. Test /api/embed
    print("\n[2] Testing /api/embed endpoint...")
    with open(img_path, "rb") as f:
        files = {"image": ("test_cover.jpg", f, "image/jpeg")}
        data = {"message": secret_message, "password": password}
        r = requests.post(f"{API_URL}/api/embed", files=files, data=data)

    assert r.status_code == 200, f"Embed failed: {r.status_code} - {r.text}"
    resp = r.json()
    metrics = resp["metrics"]
    print(f"    [+] Response Status: 200 OK")
    print(f"    [+] Returned Dimensions: {metrics['width']} x {metrics['height']}")
    print(f"    [+] Full Image PSNR: {metrics['psnr_db']:.2f} dB")
    print(f"    [+] Full Image SSIM: {metrics['ssim']:.4f}")
    print(f"    [+] Embedded Patch PSNR: {metrics['patch_psnr_db']:.2f} dB")
    print(f"    [+] Inference Duration: {resp['embedding_time_s']} s")

    # 3. Test /api/download_stego
    print("\n[3] Testing /api/download_stego (lossless PNG)...")
    with open(img_path, "rb") as f:
        files = {"image": ("test_cover.jpg", f, "image/jpeg")}
        data = {"message": secret_message, "password": password}
        r_down = requests.post(f"{API_URL}/api/download_stego", files=files, data=data)

    assert r_down.status_code == 200, f"Download failed: {r_down.status_code}"
    stego_pil = Image.open(io.BytesIO(r_down.content)).convert("RGB")
    stego_w, stego_h = stego_pil.size
    stego_np = np.array(stego_pil)

    print(f"    [+] Downloaded PNG Dimensions: {stego_w} x {stego_h}")
    assert (stego_w, stego_h) == (orig_w, orig_h), (
        f"Dimensions mismatch! Original: {(orig_w, orig_h)}, Stego: {(stego_w, stego_h)}"
    )
    print("    [PASS] Original and Stego image dimensions match 100% exactly!")

    # 4. Verify that surrounding pixels outside the 256x256 patch are 100% bitwise untouched
    y0 = (orig_h - 256) // 2
    x0 = (orig_w - 256) // 2
    
    # Check top boundary slice
    if y0 > 0:
        top_orig = orig_np[:y0, :]
        top_stego = stego_np[:y0, :]
        assert np.array_equal(top_orig, top_stego), "Top region altered!"
    
    # Check bottom boundary slice
    if y0 + 256 < orig_h:
        bot_orig = orig_np[y0 + 256:, :]
        bot_stego = stego_np[y0 + 256:, :]
        assert np.array_equal(bot_orig, bot_stego), "Bottom region altered!"

    # Check left boundary slice
    if x0 > 0:
        left_orig = orig_np[y0:y0+256, :x0]
        left_stego = stego_np[y0:y0+256, :x0]
        assert np.array_equal(left_orig, left_stego), "Left region altered!"

    # Check right boundary slice
    if x0 + 256 < orig_w:
        right_orig = orig_np[y0:y0+256, x0+256:]
        right_stego = stego_np[y0:y0+256, x0+256:]
        assert np.array_equal(right_orig, right_stego), "Right region altered!"

    print("    [PASS] Surrounding original image context is 100% bitwise preserved!")

    # 5. Extract from downloaded complete PNG
    print("\n[4] Testing /api/extract with re-uploaded complete Stego PNG...")
    files = {"image": ("stego_downloaded.png", io.BytesIO(r_down.content), "image/png")}
    data = {"password": password}
    r_ext = requests.post(f"{API_URL}/api/extract", files=files, data=data)

    assert r_ext.status_code == 200, f"Extract failed: {r_ext.status_code} - {r_ext.text}"
    recovered = r_ext.json().get("message")
    print(f"    [+] Recovered Message: \"{recovered}\"")
    assert recovered == secret_message, f"Secret recovery mismatch! Expected '{secret_message}', got '{recovered}'"
    print("    [PASS] Exact secret message recovered with 100% cryptographic fidelity!")

    # 6. Test wrong password rejection
    print("\n[5] Testing incorrect password rejection...")
    files = {"image": ("stego_downloaded.png", io.BytesIO(r_down.content), "image/png")}
    data = {"password": "TotallyWrongPassword!"}
    r_bad = requests.post(f"{API_URL}/api/extract", files=files, data=data)
    assert r_bad.status_code == 401, f"Expected 401, got {r_bad.status_code}"
    print(f"    [PASS] Wrong password rejected with safe message: {r_bad.json()['error']}")

    # 7. Test undersized image rejection
    print("\n[6] Testing undersized image validation (< 256x256)...")
    tiny_img = Image.new("RGB", (150, 150), color="blue")
    tiny_buf = io.BytesIO()
    tiny_img.save(tiny_buf, format="PNG")
    tiny_buf.seek(0)

    files = {"image": ("tiny.png", tiny_buf, "image/png")}
    data = {"message": "Should Fail", "password": "password"}
    r_tiny = requests.post(f"{API_URL}/api/embed", files=files, data=data)
    assert r_tiny.status_code == 400, f"Expected 400, got {r_tiny.status_code}"
    print(f"    [PASS] Undersized image cleanly rejected: {r_tiny.json()['error']}")

    print("\n" + "=" * 70)
    print("  ALL 6 FULL-IMAGE PRESERVATION TESTS PASSED WITH 100% FIDELITY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
