from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.sql import func
from ..database import Base


class Sessao(Base):
    __tablename__ = "sessoes"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(String, unique=True, index=True, nullable=False)
    location = Column(String, nullable=False)
    ativa = Column(Boolean, default=True, nullable=False)
    iniciada_em = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    ultima_atividade = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
