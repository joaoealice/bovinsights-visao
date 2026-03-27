from PIL import Image
import io
import base64
from fastapi import HTTPException

MAX_SIZE = (1280, 1280)


def validate_and_resize(image_bytes: bytes) -> bytes:
    """
    Valida que é uma imagem válida, redimensiona para max 1280px
    e retorna bytes JPEG otimizados para inferência.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Arquivo não é uma imagem válida")

    # Converter para RGB (remove canal alpha se houver)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Redimensionar mantendo proporção
    img.thumbnail(MAX_SIZE, Image.LANCZOS)

    # Converter para JPEG bytes
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=85)
    return output.getvalue()


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")
