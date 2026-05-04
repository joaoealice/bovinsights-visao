from datetime import datetime
from pydantic import BaseModel


class SessaoIniciarRequest(BaseModel):
    source_id: str
    location: str


class SessaoOut(BaseModel):
    id: int
    source_id: str
    location: str
    ativa: bool
    iniciada_em: datetime
    ultima_atividade: datetime

    model_config = {"from_attributes": True}
