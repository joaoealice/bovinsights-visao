from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.sql import func
from ..database import Base


class Alerta(Base):
    __tablename__ = "alertas"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(String, index=True, nullable=False)
    tipo = Column(String, nullable=False)
    descricao = Column(String, nullable=False)
    resolvido = Column(Boolean, default=False, nullable=False)
    criado_em = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    resolvido_em = Column(DateTime(timezone=True), nullable=True)
