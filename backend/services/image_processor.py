import io
import base64
import logging
import numpy as np
from PIL import Image
from fastapi import HTTPException

logger = logging.getLogger(__name__)

MAX_SIZE = (1280, 1280)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    logger.warning("opencv-python-headless não encontrado — CLAHE desativado (degraded mode)")
    _CV2_AVAILABLE = False


def _apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def validate_and_resize(image_bytes: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Arquivo não é uma imagem válida")

    if img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail(MAX_SIZE, Image.LANCZOS)

    if _CV2_AVAILABLE:
        arr = np.array(img)
        arr = _apply_clahe(arr)
        img = Image.fromarray(arr)

    output = io.BytesIO()
    img.save(output, format="JPEG", quality=85)
    return output.getvalue()


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")
