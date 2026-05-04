from typing import List, Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.alerta import Alerta
from ..schemas.alerta import AlertaOut

router = APIRouter(prefix="/api/v1", tags=["alertas"])


@router.get("/alertas", response_model=List[AlertaOut])
def listar_alertas(
    source_id: Optional[str] = None,
    resolvido: Optional[bool] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Alerta)
    if source_id is not None:
        query = query.filter(Alerta.source_id == source_id)
    if resolvido is not None:
        query = query.filter(Alerta.resolvido == resolvido)
    return query.order_by(Alerta.criado_em.desc()).all()
