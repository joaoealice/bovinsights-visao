import httpx
import asyncio
import time
from ..core.config import get_settings
from ..schemas.detection import DetectionResponse, DetectionBox, BehaviorCount

# Mapeamento: labels do Roboflow → classes padronizadas do Bovisights
LABEL_MAP = {
    "eating": "eating",
    "foraging": "eating",
    "feeding": "eating",
    "lying": "lying",
    "lying down": "lying",
    "lying_down": "lying",
    "resting": "lying",
    "standing": "standing",
    "standing up": "standing",
    "drinking": "drinking",
    "drinking water": "drinking",
    "ruminating": "ruminating",
    "rumination": "ruminating",
    "running": "running",
    "walking": "running",
    "falling": "unknown",
    "sitting": "unknown",
    "stand-eat-sit-fall": "unknown",
}


async def _call_model(client: httpx.AsyncClient, model_id: str, version: int,
                      api_key: str, image_b64: str, confidence: int = 20) -> list:
    """Chama um modelo Roboflow e retorna as predições."""
    url = f"https://detect.roboflow.com/{model_id}/{version}"
    response = await client.post(
        url,
        params={"api_key": api_key, "confidence": confidence, "overlap": 30},
        data=image_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if response.status_code != 200:
        raise Exception(f"Roboflow API error {model_id}: {response.status_code}")
    return response.json().get("predictions", [])


async def run_inference(image_b64: str) -> DetectionResponse:
    settings = get_settings()
    start = time.time()

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Rodar os dois modelos em paralelo
        behavior_task = _call_model(
            client,
            settings.roboflow_model_id,
            settings.roboflow_model_version,
            settings.roboflow_api_key,
            image_b64,
            confidence=20,
        )
        count_task = _call_model(
            client,
            settings.roboflow_count_model_id,
            settings.roboflow_count_model_version,
            settings.roboflow_api_key,
            image_b64,
            confidence=20,
        )
        behavior_preds, count_preds = await asyncio.gather(
            behavior_task, count_task, return_exceptions=True
        )

    elapsed_ms = (time.time() - start) * 1000

    # Se o modelo de comportamento falhou, usar lista vazia
    if isinstance(behavior_preds, Exception):
        print(f"[WARN] Modelo de comportamento falhou: {behavior_preds}")
        behavior_preds = []

    # Se o modelo de contagem falhou, usar lista vazia
    if isinstance(count_preds, Exception):
        print(f"[WARN] Modelo de contagem falhou: {count_preds}")
        count_preds = []

    # Processar comportamentos
    counts = BehaviorCount()
    boxes = []

    for pred in behavior_preds:
        raw_label = pred.get("class", "unknown").lower()
        label = LABEL_MAP.get(raw_label, "unknown")

        current = getattr(counts, label, None)
        if current is not None:
            setattr(counts, label, current + 1)
        else:
            counts.unknown += 1

        boxes.append(DetectionBox(
            x=pred["x"],
            y=pred["y"],
            width=pred["width"],
            height=pred["height"],
            confidence=pred["confidence"],
            label=label,
        ))

    behavior_total = sum([
        counts.eating, counts.lying, counts.standing,
        counts.drinking, counts.ruminating, counts.running, counts.unknown
    ])

    # Animais detectados pela contagem mas sem comportamento classificado
    count_total = len(count_preds)
    not_identified = max(0, count_total - behavior_total)
    counts.not_identified = not_identified

    # Total real = maior entre os dois modelos
    total = max(behavior_total + not_identified, behavior_total)

    return DetectionResponse(
        success=True,
        total_animals=total,
        behaviors=counts,
        detections=boxes,
        inference_time_ms=round(elapsed_ms, 1),
        model_used="roboflow+count",
    )
