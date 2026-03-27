import base64
import httpx
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from ..core.config import get_settings
import requests as req_sync

router = APIRouter(prefix="/api/v1/training", tags=["training"])


class TrainingUploadRequest(BaseModel):
    image: str              # base64 da imagem
    label_hint: Optional[str] = None   # ex: "lying", "eating", "standing"
    batch_name: Optional[str] = "campo"


class TrainingUploadResponse(BaseModel):
    success: bool
    message: str
    roboflow_url: Optional[str] = None


@router.post("/upload", response_model=TrainingUploadResponse)
async def upload_training_frame(payload: TrainingUploadRequest):
    """
    Recebe um frame da câmera e faz upload para o Roboflow como imagem de treino.
    O label_hint é adicionado como tag para facilitar a anotação posterior.
    """
    settings = get_settings()

    if not settings.roboflow_workspace or not settings.roboflow_project:
        raise HTTPException(
            status_code=503,
            detail="ROBOFLOW_WORKSPACE e ROBOFLOW_PROJECT não configurados no .env"
        )

    try:
        image_bytes = base64.b64decode(payload.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Base64 inválido")

    tags = ["campo"]
    if payload.label_hint:
        tags.append(payload.label_hint)

    # API de upload do Roboflow (base64 + form-urlencoded)
    url = f"https://api.roboflow.com/dataset/{settings.roboflow_project}/upload"
    params = {
        "api_key": settings.roboflow_api_key,
        "batch": payload.batch_name,
        "split": "train",
    }
    for tag in tags:
        params.setdefault("tag", tag)

    try:
        def _upload():
            print(f"[DEBUG] URL: {url}")
            print(f"[DEBUG] Params: {params}")
            print(f"[DEBUG] Image len: {len(payload.image)}")
            r = req_sync.post(
                url,
                params=params,
                data=payload.image,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=20,
            )
            print(f"[DEBUG] Roboflow status: {r.status_code}")
            print(f"[DEBUG] Roboflow response: {r.text[:300]}")
            return r
        response = await asyncio.to_thread(_upload)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erro ao conectar no Roboflow: {e}")

    if response.status_code not in (200, 201):
        raise HTTPException(
            status_code=502,
            detail=f"Roboflow rejeitou o upload: {response.status_code} — {response.text[:200]}"
        )

    data = response.json()
    project_url = (
        f"https://app.roboflow.com/{settings.roboflow_workspace}/{settings.roboflow_project}"
    )

    return TrainingUploadResponse(
        success=True,
        message=f"Frame enviado com tag '{payload.label_hint or 'sem tag'}'. Anote em:",
        roboflow_url=project_url,
    )
