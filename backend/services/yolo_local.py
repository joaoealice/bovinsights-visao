"""
Fallback local usando YOLOv11 nano (sem GPU, sem internet).
Menos preciso para comportamentos, mas garante disponibilidade.
Detecta apenas "cow" (classe 19 do COCO) sem classes de comportamento.
Em produção (nuvem), este fallback não é usado — o Roboflow sempre está disponível.
"""
import io
import time
from PIL import Image
from functools import lru_cache
from ..schemas.detection import DetectionResponse, DetectionBox, BehaviorCount

try:
    import numpy as np
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


@lru_cache(maxsize=1)
def get_model():
    if not _YOLO_AVAILABLE:
        return None
    return YOLO("yolo11n.pt")


async def run_inference_local(image_bytes: bytes) -> DetectionResponse:
    if not _YOLO_AVAILABLE:
        return DetectionResponse(
            success=False,
            total_animals=0,
            behaviors=BehaviorCount(),
            detections=[],
            inference_time_ms=0,
            model_used="unavailable",
            message="Modelo local indisponível. Verifique a conexão com o Roboflow.",
        )
    model = get_model()
    img = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(img)

    start = time.time()
    results = model.predict(img_array, classes=[19], conf=0.4, verbose=False)
    elapsed_ms = (time.time() - start) * 1000

    boxes = []
    for r in results:
        for box in r.boxes:
            x, y, w, h = box.xywh[0].tolist()
            boxes.append(DetectionBox(
                x=x, y=y, width=w, height=h,
                confidence=float(box.conf[0]),
                label="standing",  # sem classe comportamental no fallback
            ))

    total = len(boxes)
    counts = BehaviorCount(standing=total)  # assume standing no fallback

    return DetectionResponse(
        success=True,
        total_animals=total,
        behaviors=counts,
        detections=boxes,
        inference_time_ms=round(elapsed_ms, 1),
        model_used="local_yolo11n",
        message="Modo offline: comportamentos não disponíveis, apenas contagem",
    )
