from pydantic import BaseModel
from typing import List, Optional


class BehaviorCount(BaseModel):
    eating: int = 0
    standing: int = 0
    deitado: int = 0
    running: int = 0
    refugo: int = 0
    isolado: int = 0
    postura_anormal: int = 0
    xibungo: int = 0
    urinando: int = 0
    unknown: int = 0


class DetectionBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    label: str


class DetectionResponse(BaseModel):
    success: bool
    total_animals: int
    behaviors: BehaviorCount
    detections: List[DetectionBox]
    inference_time_ms: float
    model_used: str  # "roboflow" ou "local"
    message: Optional[str] = None
