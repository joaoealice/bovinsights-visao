from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..core.config import Settings, get_settings
from ..database import get_db
from ..models.sessao import Sessao
from ..schemas.detection import DetectionResponse
from ..schemas.sessao import SessaoOut
from ..services.image_processor import image_to_base64, validate_and_resize
from ..services.roboflow_client import run_inference
from ..services.yolo_local import run_inference_local

router = APIRouter(prefix="/api/v1/stream", tags=["stream"])


@router.post("/frame", response_model=DetectionResponse)
async def receber_frame(
    file: UploadFile = File(...),
    source_id: str = Form(...),
    location: str = Form(...),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
):
    content = await file.read()
    if len(content) > settings.max_image_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Imagem muito grande. Máximo: {settings.max_image_size_mb}MB",
        )

    # validate_and_resize aplica CLAHE quando cv2 disponível e HISTOGRAM_EQUALIZATION=true
    processed = validate_and_resize(content)
    image_b64 = image_to_base64(processed)

    try:
        result = await run_inference(image_b64)
    except Exception:
        result = await run_inference_local(processed)

    _upsert_sessao(db, source_id, location)

    return result


@router.get("/sources", response_model=List[SessaoOut])
def listar_fontes_ativas(db: Session = Depends(get_db)):
    return (
        db.query(Sessao)
        .filter(Sessao.ativa == True)  # noqa: E712
        .order_by(Sessao.ultima_atividade.desc())
        .all()
    )


def _upsert_sessao(db: Session, source_id: str, location: str) -> None:
    now = datetime.now(timezone.utc)
    sessao = db.query(Sessao).filter(Sessao.source_id == source_id).first()
    if sessao is None:
        db.add(Sessao(source_id=source_id, location=location, ativa=True, ultima_atividade=now))
    else:
        sessao.location = location
        sessao.ativa = True
        sessao.ultima_atividade = now
    db.commit()
