from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class AlertaOut(BaseModel):
    id: int
    source_id: str
    tipo: str
    descricao: str
    resolvido: bool
    criado_em: datetime
    resolvido_em: Optional[datetime] = None

    model_config = {"from_attributes": True}
