from pydantic import BaseModel
from typing import List, Optional


class BehaviorCount(BaseModel):
    eating: int = 0
    lying: int = 0
    standing: int = 0
    drinking: int = 0
    ruminating: int = 0
    running: int = 0
    unknown: int = 0
    not_identified: int = 0  # detectados pelo modelo de contagem mas sem comportamento classificado


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
