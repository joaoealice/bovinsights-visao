"""
backend/services/roboflow_client.py  ← SUBSTITUI O ARQUIVO ATUAL

Correções aplicadas v2:
  1. Confidence: 20% → 45% (elimina detecções fantasmas)
  2. Removido cow-count paralelo (causava caixas gigantes e "Outro")
  3. LABEL_MAP atualizado para classes v2 do bovinos-demons
  4. Lógica de contagem simplificada e limpa
"""

import httpx
import time
from ..core.config import get_settings
from ..schemas.detection import DetectionResponse, DetectionBox, BehaviorCount

# ── Mapeamento: classes do modelo → campos do BehaviorCount ──────────────────
# Classes v2 (bovinos-demons) + aliases antigos para compatibilidade

LABEL_MAP = {
    # ── Classes v2 (bovinos-demons) ──────────────────────────────────────────
    "animal_em_pe":        "standing",
    "animal_comendo":      "eating",
    "animal_deitado":      "deitado",
    "animal_em_movimento": "running",
    "animal_urinando":     "urinando",
    "sodomia_xibungo":     "xibungo",
    "sodomia-xibungo":     "xibungo",
    "boi_refugo":          "refugo",
    "isolado":             "isolado",
    "postura_anormal":     "postura_anormal",

    # ── Zona do cocho (anotação de área, não é animal — ignorar) ─────────────
    "colcho":              None,
    "zona_cocho":          None,
    "zone_cocho":          None,

    # ── Classes antigas (modelo anterior — compatibilidade) ──────────────────
    "eating":              "eating",
    "foraging":            "eating",
    "feeding":             "eating",
    "lying":               "deitado",
    "lying down":          "deitado",
    "lying_down":          "deitado",
    "resting":             "deitado",
    "standing":            "standing",
    "standing up":         "standing",
    "drinking":            "eating",   # sem campo próprio — agrupa em eating
    "ruminating":          "standing", # ruminando em pé
    "rumination":          "standing",
    "running":             "running",
    "walking":             "running",
}


async def run_inference(image_b64: str) -> DetectionResponse:
    """
    Chama o modelo Roboflow de comportamento bovino (single model).

    Removido: cow-count paralelo
    Motivo:   O cow-count retornava caixas gigantes ao redor de grupos,
              inflando a contagem e gerando a categoria "Outro" no frontend.
              A contagem agora é feita somando as detecções individuais.

    Confidence: 45% (era 20%)
    Motivo:    20% gerava detecções fantasmas em sombras, postes e cercas.
               45% mantém sensibilidade sem falsos positivos excessivos.
    """
    settings = get_settings()
    start = time.time()

    url = f"https://detect.roboflow.com/{settings.roboflow_model_id}/{settings.roboflow_model_version}"

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            url,
            params={
                "api_key":    settings.roboflow_api_key,
                "confidence": 45,   # ← era 20, agora 45
                "overlap":    30,
            },
            data=image_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if response.status_code != 200:
        raise Exception(
            f"Roboflow API error {settings.roboflow_model_id}: {response.status_code}"
        )

    predictions = response.json().get("predictions", [])
    elapsed_ms = (time.time() - start) * 1000

    # ── Processar predições ───────────────────────────────────────────────────
    counts = BehaviorCount()
    boxes = []
    total = 0

    for pred in predictions:
        raw_label = pred.get("class", "unknown").lower()
        label = LABEL_MAP.get(raw_label)

        # Zona do cocho (label=None) → não conta, não desenha caixa
        if label is None:
            continue

        total += 1

        # Incrementa o contador correspondente (com fallback para unknown)
        current = getattr(counts, label, None)
        if current is not None:
            setattr(counts, label, current + 1)
        else:
            counts.unknown = getattr(counts, "unknown", 0) + 1

        boxes.append(DetectionBox(
            x=pred["x"],
            y=pred["y"],
            width=pred["width"],
            height=pred["height"],
            confidence=pred["confidence"],
            label=label,
        ))

    return DetectionResponse(
        success=True,
        total_animals=total,
        behaviors=counts,
        detections=boxes,
        inference_time_ms=round(elapsed_ms, 1),
        model_used="roboflow-v2",
    )
