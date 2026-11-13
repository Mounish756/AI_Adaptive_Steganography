# Updated API server with blending mask functionality

"""
Flask API Server — AI-Based Adaptive Image Embedding Framework
Provides HTTP endpoints for the frontend dashboard with full-image preservation.
"""

import io
import os
import sys
import time
import base64
import logging
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Configure logging — never log passwords or plaintexts
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stego-api")

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.models.full_model import AdaptiveStegoModel
from ai.data.preprocessing import to_tensor, to_numpy, normalize_image, denormalize_image
from ai.metrics.psnr import calculate_psnr
from ai.metrics.ssim import calculate_ssim
from ai.metrics.mse import calculate_mse
from security.encryption import AuthenticationError
from security.payload import (
    PayloadCapacityError,
    CorruptedPayloadError,
    calculate_usable_capacity,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

_model: Optional[AdaptiveStegoModel] = None
_device: Optional[torch.device] = None

MAX_UPLOAD_BYTES = 25 * 1024 * 1024   # 25 MB per file
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
CARRIER_DIM = 256                      # Neural model native patch size
DEFAULT_BLEND_WIDTH = 16                # Pixels to feather edges

# ---------------------------------------------------------------------------
# Model loader (called once at startup)
# ---------------------------------------------------------------------------
def get_model() -> AdaptiveStegoModel:
    global _model, _device
    if _model is not None:
        return _model

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading neural steganography model on {_device} …")

    _model = AdaptiveStegoModel(
        base_dim=32,
        analyzer_checkpoint=os.path.join(PROJECT_ROOT, "models/adaptive_analyzer/best_model.pth"),
        freeze_analyzer=True,
        scaling_factor=0.13,
    ).to(_device)

    # Try best available checkpoint
    ckpt_candidates = [
        os.path.join(PROJECT_ROOT, "checkpoints/phase5_balanced/best_model.pth"),
        os.path.join(PROJECT_ROOT, "checkpoints/phase2b/phase2b_model.pth"),
        os.path.join(PROJECT_ROOT, "checkpoints/phase2a/best_model.pth"),
    ]
    for ckpt in ckpt_candidates:
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=_device, weights_only=False)
            _model.load_state_dict(state.get("model_state_dict", state))
            logger.info(f"Loaded checkpoint: {ckpt}")
            break
    else:
        logger.warning("No checkpoint found — using random weights.")

    _model.eval()
    return _model

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _pil_from_upload(file_storage) -> Image.Image:
    """Read uploaded file and return PIL Image in RGB mode."""
    data = file_storage.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Uploaded file exceeds 25 MB limit.")
    ext = os.path.splitext(file_storage.filename or "")[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(ALLOWED_EXTENSIONS)}")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img

def _pil_to_png_bytes(img: Image.Image) -> bytes:
    """Save PIL Image to in-memory PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _pil_to_b64_png(img: Image.Image) -> str:
    """Convert PIL Image to base64 data URI."""
    return "data:image/png;base64," + base64.b64encode(_pil_to_png_bytes(img)).decode("utf-8")

def _create_blend_mask(patch_dim: int, border: int) -> np.ndarray:
    """Create a 2‑D cosine feathering mask.

    Args:
        patch_dim: Size of the square patch (e.g., 256).
        border: Width of the feathering border in pixels. Must satisfy 0 <= border <= patch_dim // 2.

    Returns:
        A (patch_dim, patch_dim) float array with values in [0, 1].
        If border == 0, returns an all-ones mask (no blending).
        The interior region (distance > border) is 1.0, the outer ``border`` pixels transition smoothly
        to 0.0 using a cosine curve.
    """
    if border < 0 or border > patch_dim // 2:
        raise ValueError(f"Invalid border width {border}. Must be >= 0 and <= {patch_dim//2}.")
    if border == 0:
        return np.ones((patch_dim, patch_dim), dtype=np.float32)
    # Create 1‑D cosine ramp from 0 to 1 over 'border' pixels
    ramp = (1 - np.cos(np.linspace(0, np.pi, border))) / 2  # shape (border,)
    # Build 2‑D mask by applying ramp on each side
    mask = np.ones((patch_dim, patch_dim), dtype=np.float32)
    # Top and bottom
    mask[:border, :] = ramp[:, None]
    mask[-border:, :] = ramp[::-1][:, None]
    # Left and right (multiply to combine effects)
    mask[:, :border] *= ramp[None, :]
    mask[:, -border:] *= ramp[::-1][None, :]
    return mask

def _embed_into_full_image(
    cover_pil: Image.Image,
    message: str,
    password: str,
    blend_width: int = DEFAULT_BLEND_WIDTH,
) -> Tuple[Image.Image, dict, float]:
    """Embed secret message into a full cover image with optional cosine blending.

    Steps:
    1. Validate dimensions (must be at least 256×256).
    2. Extract the centre 256×256 patch.
    3. Run neural encoder on the patch.
    4. Optionally feather the patch edges using a cosine mask.
    5. Replace the patch back into the original image.
    6. Compute full‑image fidelity metrics.
    """
    orig_w, orig_h = cover_pil.size
    if orig_w < CARRIER_DIM or orig_h < CARRIER_DIM:
        raise ValueError(
            f"Input image dimensions ({orig_w}×{orig_h}) are smaller than the required "
            f"{CARRIER_DIM}×{CARRIER_DIM} minimum. Please upload an image at least {CARRIER_DIM}×{CARRIER_DIM} pixels."
        )

    # Original NumPy array (H, W, 3)
    orig_np = np.array(cover_pil)

    # Centre patch coordinates
    y0 = (orig_h - CARRIER_DIM) // 2
    x0 = (orig_w - CARRIER_DIM) // 2
    patch_np = orig_np[y0 : y0 + CARRIER_DIM, x0 : x0 + CARRIER_DIM]

    # Normalise patch for model
    patch_norm = normalize_image(patch_np)
    patch_tensor = to_tensor(patch_norm).unsqueeze(0).to(_device)

    # Model forward pass
    model = get_model()
    t0 = time.perf_counter()
    stego_patch_tensor, patch_metrics = model.embed_secret(patch_tensor, message, password)
    elapsed = time.perf_counter() - t0

    # Convert torch output to uint8 NumPy
    stego_patch_np = denormalize_image(to_numpy(stego_patch_tensor[0]))

    # ---------- Blending logic (experimental) ----------
    try:
        mask = _create_blend_mask(CARRIER_DIM, blend_width)[:, :, None]  # (H, W, 1)
        blended_patch = (mask * stego_patch_np.astype(np.float32) + (1 - mask) * patch_np.astype(np.float32)).astype(np.uint8)
        # Verify extraction on the blended patch before committing
        blended_tensor = to_tensor(normalize_image(blended_patch)).unsqueeze(0).to(_device)
        # Run decoder to ensure payload is recoverable
        recovered = model.extract_secret(blended_tensor, password, as_bytes=False)
        # If we reach here, blending preserved the payload
        final_patch = blended_patch
    except Exception as e:
        # Log the failure and fall back to the raw stego patch (no blending)
        logger.info(f"Blending failed (likely payload corruption): {e}. Falling back to raw patch.")
        final_patch = stego_patch_np
    # ----------------------------------------------------

    # Reconstruct the full stego image
    full_stego_np = orig_np.copy()
    full_stego_np[y0 : y0 + CARRIER_DIM, x0 : x0 + CARRIER_DIM] = final_patch
    full_stego_pil = Image.fromarray(full_stego_np, mode="RGB")

    # Ensure dimensions are preserved
    assert full_stego_pil.size == (orig_w, orig_h), (
        f"Dimension mismatch: expected {(orig_w, orig_h)}, got {full_stego_pil.size}"
    )

    # Compute metrics on the complete image
    c_norm = orig_np.astype(np.float64) / 255.0
    s_norm = full_stego_np.astype(np.float64) / 255.0
    mse_full = float(calculate_mse(c_norm, s_norm))
    psnr_full = float(calculate_psnr(c_norm, s_norm, max_val=1.0))
    ssim_full = float(calculate_ssim(c_norm, s_norm, max_val=1.0))

    metrics = {
        "psnr_db": psnr_full,
        "ssim": ssim_full,
        "mse": mse_full,
        "patch_psnr_db": patch_metrics["psnr_db"],
        "patch_ssim": patch_metrics["ssim"],
        "plaintext_bytes": patch_metrics["plaintext_bytes"],
        "ciphertext_bytes": patch_metrics["ciphertext_bytes"],
        "carrier_bytes": patch_metrics["carrier_bytes"],
        "width": orig_w,
        "height": orig_h,
        "aspect_ratio": round(orig_w / orig_h, 3),
    }

    return full_stego_pil, metrics, elapsed

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(_device or "not_loaded")})

@app.route("/api/capacity", methods=["GET"])
def capacity():
    """Return usable payload capacity and system config for 256×256 carrier map."""
    try:
        usable = calculate_usable_capacity(carrier_shape=(CARRIER_DIM, CARRIER_DIM), nsym=32)
        return jsonify({
            "carrier_dim": CARRIER_DIM,
            "carrier_pixels": CARRIER_DIM * CARRIER_DIM,
            "carrier_bytes": (CARRIER_DIM * CARRIER_DIM) // 8,
            "usable_payload_bytes": usable,
            "recommended_max_chars": min(usable - 100, 4000),
            "crypto_standard": "AES-256-GCM",
            "fec_scheme": "Reed-Solomon (32)",
            "key_derivation": "PBKDF2-HMAC-SHA256 (100k)",
            "model_architecture": "NeuralAdaptiveAnalyzer + UNet",
            "device": str(_device or "cpu"),
        })
    except Exception:
        return jsonify({"error": "Could not compute capacity."}), 500

@app.route("/api/embed", methods=["POST"])
def embed():
    """Embed a secret message into a complete cover image.

    Form-data:
      image      — cover image file (required)
      message    — plaintext secret message (required, max 4000 chars)
      password   — encryption password (required)
      blend_width — optional integer border width for feathering (default 16)
    """
    try:
        # Validate inputs
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400
        message = request.form.get("message", "").strip()
        password = request.form.get("password", "").strip()
        blend_width_raw = request.form.get("blend_width", None)
        blend_width = DEFAULT_BLEND_WIDTH
        if blend_width_raw is not None:
            try:
                blend_width = int(blend_width_raw)
            except ValueError:
                return jsonify({"error": "blend_width must be an integer."}), 400
            # Validate range
            if blend_width < 0 or blend_width > CARRIER_DIM // 2:
                return jsonify({"error": f"blend_width must be >= 0 and <= {CARRIER_DIM // 2}."}), 400

        if not message:
            return jsonify({"error": "Message cannot be empty."}), 400
        if not password:
            return jsonify({"error": "Password cannot be empty."}), 400
        if len(message) > 4000:
            return jsonify({"error": "Message exceeds 4,000 character limit."}), 400
        if len(password) < 4:
            return jsonify({"error": "Password must be at least 4 characters."}), 400

        # Load cover image
        try:
            cover_pil = _pil_from_upload(request.files["image"])
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Embed into complete image with blending
        full_stego_pil, metrics, elapsed = _embed_into_full_image(
            cover_pil, message, password, blend_width=blend_width
        )

        # Generate preview images (preserve aspect ratio, max 1024px)
        orig_w, orig_h = cover_pil.size
        max_preview_dim = 1024
        if max(orig_w, orig_h) > max_preview_dim:
            ratio = max_preview_dim / max(orig_w, orig_h)
            new_size = (int(orig_w * ratio), int(orig_h * ratio))
            preview_orig = cover_pil.resize(new_size, Image.LANCZOS)
            preview_stego = full_stego_pil.resize(new_size, Image.LANCZOS)
        else:
            preview_orig = cover_pil
            preview_stego = full_stego_pil

        original_b64 = _pil_to_b64_png(preview_orig)
        stego_b64 = _pil_to_b64_png(preview_stego)

        logger.info(
            f"embed OK | dims={orig_w}x{orig_h} | payload={metrics['plaintext_bytes']}B | "
            f"Full-PSNR={metrics['psnr_db']:.2f}dB | Full-SSIM={metrics['ssim']:.4f} | "
            f"Patch-PSNR={metrics['patch_psnr_db']:.2f}dB | t={elapsed:.3f}s"
        )

        return jsonify({
            "stego_image_b64": stego_b64,
            "original_b64": original_b64,
            "metrics": {
                "psnr_db": round(metrics["psnr_db"], 2),
                "ssim": round(metrics["ssim"], 4),
                "mse": round(metrics["mse"], 8),
                "patch_psnr_db": round(metrics["patch_psnr_db"], 2),
                "patch_ssim": round(metrics["patch_ssim"], 4),
                "plaintext_bytes": metrics["plaintext_bytes"],
                "ciphertext_bytes": metrics["ciphertext_bytes"],
                "carrier_bytes": metrics["carrier_bytes"],
                "width": orig_w,
                "height": orig_h,
                "dimensions": f"{orig_w} × {orig_h}",
            },
            "embedding_time_s": round(elapsed, 4),
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except PayloadCapacityError as e:
        return jsonify({"error": f"Payload too large: {e}"}), 422
    except Exception as e:
        logger.exception("Unexpected error in /api/embed")
        return jsonify({"error": "Embedding failed due to an internal error."}), 500

@app.route("/api/download_stego", methods=["POST"])
def download_stego():
    """Generate and download the complete stego image as a lossless PNG.
    The downloaded image preserves full original dimensions and aspect ratio.
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400
        message = request.form.get("message", "").strip()
        password = request.form.get("password", "").strip()
        blend_width_raw = request.form.get("blend_width", None)
        blend_width = DEFAULT_BLEND_WIDTH
        if blend_width_raw is not None:
            try:
                blend_width = int(blend_width_raw)
            except ValueError:
                return jsonify({"error": "blend_width must be an integer."}), 400
            if blend_width < 0 or blend_width > CARRIER_DIM // 2:
                return jsonify({"error": f"blend_width must be >= 0 and <= {CARRIER_DIM // 2}."}), 400

        if not message or not password:
            return jsonify({"error": "Message and password are required."}), 400
        if len(message) > 4000:
            return jsonify({"error": "Message exceeds 4,000 character limit."}), 400

        cover_pil = _pil_from_upload(request.files["image"])
        full_stego_pil, _, _ = _embed_into_full_image(
            cover_pil, message, password, blend_width=blend_width
        )

        png_bytes = _pil_to_png_bytes(full_stego_pil)
        return send_file(
            io.BytesIO(png_bytes),
            mimetype="image/png",
            as_attachment=True,
            download_name="stego_image.png",
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except PayloadCapacityError as e:
        return jsonify({"error": f"Payload too large: {e}"}), 422
    except Exception:
        logger.exception("Unexpected error in /api/download_stego")
        return jsonify({"error": "Download generation failed."}), 500

@app.route("/api/extract", methods=["POST"])
def extract() -> Tuple[dict, int]:
    """Extract a secret message from a stego image.
    Accepts full-dimension stego images (>= 256x256), extracts the centre carrier patch,
    verifies the AES-256-GCM authentication tag, and returns recovered plaintext.
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400
        password = request.form.get("password", "").strip()
        if not password:
            return jsonify({"error": "Password cannot be empty."}), 400

        try:
            stego_pil = _pil_from_upload(request.files["image"])
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        sw, sh = stego_pil.size
        if sw < CARRIER_DIM or sh < CARRIER_DIM:
            return jsonify({
                "error": f"Uploaded stego image ({sw}×{sh}) is smaller than required {CARRIER_DIM}×{CARRIER_DIM} minimum."
            }), 400

        stego_np = np.array(stego_pil)
        y0 = (sh - CARRIER_DIM) // 2
        x0 = (sw - CARRIER_DIM) // 2
        patch_np = stego_np[y0 : y0 + CARRIER_DIM, x0 : x0 + CARRIER_DIM]

        patch_norm = normalize_image(patch_np)
        stego_tensor = to_tensor(patch_norm).unsqueeze(0).to(_device)

        model = get_model()
        t0 = time.perf_counter()
        try:
            recovered = model.extract_secret(stego_tensor, password, as_bytes=False)
        except AuthenticationError:
            logger.info("extract FAILED: authentication error (wrong password or tampered image)")
            return jsonify({"error": "Authentication failed. Incorrect password or the image has been modified."}), 401
        except CorruptedPayloadError:
            logger.info("extract FAILED: corrupted payload")
            return jsonify({"error": "Payload is corrupted beyond recovery. The image may have been altered or compressed."}), 422
        except UnicodeDecodeError:
            return jsonify({"error": "Recovered payload is not valid UTF-8 text."}), 422

        elapsed = time.perf_counter() - t0
        logger.info(f"extract OK | dims={sw}x{sh} | payload={len(recovered)}B | t={elapsed:.3f}s")

        return jsonify({
            "message": recovered,
            "extraction_time_s": round(elapsed, 4),
            "width": sw,
            "height": sh,
        })

    except Exception:
        logger.exception("Unexpected error in /api/extract")
        return jsonify({"error": "Extraction failed due to an internal error."}), 500

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    get_model()
    port = int(os.environ.get("STEGO_PORT", 5050))
    logger.info(f"Starting Stego API server on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=False)
