import io
import base64
import json
import pytest
from PIL import Image
import numpy as np

# Import the Flask app and internal utilities from the server module
from api.server import app as flask_app, _create_blend_mask, CARRIER_DIM, DEFAULT_BLEND_WIDTH

flask_app.testing = True
client = flask_app.test_client()

def _create_test_image(width: int = 512, height: int = 512, color: tuple = (255, 255, 255)) -> Image.Image:
    """Create a solid color RGB image for testing."""
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")

def _image_to_file(img: Image.Image, filename: str = "test.png"):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    buf.name = filename
    return buf

def test_create_blend_mask_dimensions_and_range():
    mask = _create_blend_mask(CARRIER_DIM, DEFAULT_BLEND_WIDTH)
    assert mask.shape == (CARRIER_DIM, CARRIER_DIM)
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    # Ensure central region is exactly 1.0 when border < half size
    inner = mask[DEFAULT_BLEND_WIDTH:-DEFAULT_BLEND_WIDTH, DEFAULT_BLEND_WIDTH:-DEFAULT_BLEND_WIDTH]
    assert np.allclose(inner, 1.0)

def test_invalid_blend_width_raises():
    with pytest.raises(ValueError):
        _create_blend_mask(CARRIER_DIM, -5)
    with pytest.raises(ValueError):
        _create_blend_mask(CARRIER_DIM, CARRIER_DIM)  # too large

@pytest.mark.parametrize("blend_width", [0, 1, 8, 16])
def test_embedding_and_extraction_with_blending(blend_width):
    # Prepare a simple white image
    img = _create_test_image()
    img_file = _image_to_file(img, "cover.png")
    message = "Hello, blending test!"
    password = "securepwd"

    data = {
        "message": message,
        "password": password,
        "blend_width": str(blend_width),
    }
    response = client.post(
        "/api/embed",
        data={**data, "image": (img_file, img_file.name)},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, f"Embed failed: {response.get_data(as_text=True)}"
    resp_json = response.get_json()
    assert "stego_image_b64" in resp_json

    # Decode stego image and verify dimensions are unchanged
    stego_b64 = resp_json["stego_image_b64"].split(",", 1)[1]
    stego_bytes = base64.b64decode(stego_b64)
    stego_img = Image.open(io.BytesIO(stego_bytes))
    assert stego_img.size == img.size

    # Now extract
    stego_file = _image_to_file(stego_img, "stego.png")
    extract_resp = client.post(
        "/api/extract",
        data={"password": password, "image": (stego_file, stego_file.name)},
        content_type="multipart/form-data",
    )
    assert extract_resp.status_code == 200, f"Extract failed: {extract_resp.get_data(as_text=True)}"
    extract_json = extract_resp.get_json()
    assert extract_json["message"] == message

def test_embed_invalid_blend_width_returns_error():
    img = _create_test_image()
    img_file = _image_to_file(img, "cover.png")
    data = {
        "message": "test",
        "password": "pwd",
        "blend_width": "-10",
    }
    resp = client.post(
        "/api/embed",
        data={**data, "image": (img_file, img_file.name)},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "blend_width" in resp.get_json()["error"].lower()
