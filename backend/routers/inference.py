import base64
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from ..services.image_processor import validate_and_resize, image_to_base64
from ..services.roboflow_client import run_inference
from ..services.yolo_local import run_inference_local
from ..schemas.detection import DetectionResponse
from ..core.config import get_settings, Settings

router = APIRouter(prefix="/api/v1", tags=["inference"])


@router.post("/detect", response_model=DetectionResponse)
async def detect_behaviors(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    """
    Recebe um frame (imagem JPEG/PNG) e retorna os comportamentos
    detectados nos bovinos presentes na imagem.

    - Aceita: imagem até 5MB, formatos JPEG/PNG/WEBP
    - Retorna: contagem por comportamento + bounding boxes
    """
    content = await file.read()
    if len(content) > settings.max_image_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Imagem muito grande. Máximo: {settings.max_image_size_mb}MB"
        )

    processed = validate_and_resize(content)
    image_b64 = image_to_base64(processed)

    try:
        result = await run_inference(image_b64)
    except Exception as e:
        print(f"[WARN] Roboflow falhou: {e}. Usando modelo local.")
        result = await run_inference_local(processed)

    return result


@router.post("/detect/base64", response_model=DetectionResponse)
async def detect_behaviors_b64(
    payload: dict,
    settings: Settings = Depends(get_settings),
):
    """
    Alternativa: recebe imagem já em base64 (útil para PWA com canvas).
    Payload: {"image": "base64string..."}
    """
    try:
        image_bytes = base64.b64decode(payload["image"])
    except Exception:
        raise HTTPException(status_code=400, detail="Base64 inválido")

    processed = validate_and_resize(image_bytes)
    image_b64 = image_to_base64(processed)

    try:
        result = await run_inference(image_b64)
    except Exception as e:
        print(f"[WARN] Roboflow falhou: {e}. Usando modelo local.")
        result = await run_inference_local(processed)

    return result
