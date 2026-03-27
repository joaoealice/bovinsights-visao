from fastapi import APIRouter
from ..core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    settings = get_settings()
    return {
        "status": "ok",
        "model": "roboflow",
        "model_id": settings.roboflow_model_id,
        "version": "1.0.0",
        "env": settings.api_env,
    }
