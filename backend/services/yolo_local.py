"""
Inferência local com modelo YOLOv11 treinado no dataset Bovinsights.
Classes: Comendo, Deitado, Em pe, Escondido, Pastando
mAP50: 0.861
"""
import io
import time
from pathlib import Path
from PIL import Image
from functools import lru_cache
from ..schemas.detection import DetectionResponse, DetectionBox, BehaviorCount

try:
    import numpy as np
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# Mapeamento: classes do modelo treinado → campos do BehaviorCount
LABEL_MAP = {
    "comendo":   "eating",
    "deitado":   "deitado",
    "em pe":     "standing",
    "em pé":     "standing",
    "escondido": "unknown",
    "pastando":  "eating",   # pastando = variação de comendo
}

MODEL_PATH = Path(__file__).parent.parent / "models" / "bovinsights_yolo11n_best.pt"


@lru_cache(maxsize=1)
def get_model():
    if not _YOLO_AVAILABLE:
        return None
    return YOLO(MODEL_PATH)


async def run_inference_local(image_bytes: bytes) -> DetectionResponse:
    if not _YOLO_AVAILABLE:
        return DetectionResponse(
            success=False,
            total_animals=0,
            behaviors=BehaviorCount(),
            detections=[],
            inference_time_ms=0,
            model_used="unavailable",
            message="Ultralytics não instalado.",
        )

    model = get_model()
    img = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(img)

    start = time.time()
    results = model.predict(img_array, conf=0.4, verbose=False)
    elapsed_ms = (time.time() - start) * 1000

    counts = BehaviorCount()
    boxes = []

    for r in results:
        for box in r.boxes:
            raw_label = model.names[int(box.cls[0])].lower()
            label = LABEL_MAP.get(raw_label, "unknown")

            current = getattr(counts, label, None)
            if current is not None:
                setattr(counts, label, current + 1)
            else:
                counts.unknown += 1

            x, y, w, h = box.xywh[0].tolist()
            boxes.append(DetectionBox(
                x=x, y=y, width=w, height=h,
                confidence=float(box.conf[0]),
                label=label,
            ))

    total = len(boxes)
    return DetectionResponse(
        success=True,
        total_animals=total,
        behaviors=counts,
        detections=boxes,
        inference_time_ms=round(elapsed_ms, 1),
        model_used="bovinsights_yolo11n",
        message=f"{total} animal(is) detectado(s)",
    )