import pytest
import base64
from PIL import Image
import io
from fastapi import HTTPException
from backend.services.image_processor import validate_and_resize, image_to_base64


def make_image(width: int, height: int, mode: str = "RGB") -> bytes:
    img = Image.new(mode, (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG" if mode == "RGB" else "PNG")
    return buf.getvalue()


def test_validate_and_resize_small_image():
    img_bytes = make_image(320, 240)
    result = validate_and_resize(img_bytes)
    # Deve retornar bytes válidos de JPEG
    img = Image.open(io.BytesIO(result))
    assert img.format == "JPEG"
    assert img.size == (320, 240)  # sem redimensionamento


def test_validate_and_resize_large_image():
    img_bytes = make_image(2560, 1920)
    result = validate_and_resize(img_bytes)
    img = Image.open(io.BytesIO(result))
    assert img.format == "JPEG"
    # Deve ter sido redimensionada para caber em 1280x1280
    assert img.width <= 1280
    assert img.height <= 1280


def test_validate_and_resize_rgba():
    img = Image.new("RGBA", (200, 200), color=(50, 100, 150, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    result = validate_and_resize(img_bytes)
    out = Image.open(io.BytesIO(result))
    assert out.mode == "RGB"


def test_validate_invalid_bytes():
    with pytest.raises(HTTPException) as exc_info:
        validate_and_resize(b"isso nao e uma imagem")
    assert exc_info.value.status_code == 400


def test_image_to_base64():
    img_bytes = make_image(100, 100)
    b64 = image_to_base64(img_bytes)
    # Deve ser string base64 válida
    decoded = base64.b64decode(b64)
    assert decoded == img_bytes
