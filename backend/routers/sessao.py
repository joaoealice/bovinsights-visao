from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.sessao import Sessao
from ..schemas.sessao import SessaoIniciarRequest, SessaoOut

router = APIRouter(prefix="/api/v1/sessao", tags=["sessao"])


@router.post("/iniciar", response_model=SessaoOut)
def iniciar_sessao(payload: SessaoIniciarRequest, db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)
    sessao = db.query(Sessao).filter(Sessao.source_id == payload.source_id).first()
    if sessao is None:
        sessao = Sessao(
            source_id=payload.source_id,
            location=payload.location,
            ativa=True,
            ultima_atividade=now,
        )
        db.add(sessao)
    else:
        sessao.location = payload.location
        sessao.ativa = True
        sessao.iniciada_em = now
        sessao.ultima_atividade = now
    db.commit()
    db.refresh(sessao)
    return sessao
