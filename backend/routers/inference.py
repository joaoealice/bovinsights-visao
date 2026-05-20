"""
backend/routers/inference.py

Cadeia de fallback em 3 níveis:
  1º → Roboflow  (modelo especializado, rápido, na nuvem)
  2º → YOLO local (best.pt no servidor — sem internet)
  3º → Groq Vision (análise por IA geral — último recurso)

Por que 3 níveis?
  - Roboflow pode falhar: rate limit, timeout, serviço fora do ar
  - YOLO local pode falhar: modelo não carregado, sem memória
  - Groq quase nunca falha: tem SLA altíssimo
  - Com 3 níveis, o demo NUNCA retorna erro para o cliente

Logs gerados:
  [OK]    Roboflow respondeu em 320ms
  [WARN]  Roboflow falhou: timeout. Tentando YOLO local...
  [OK]    YOLO local respondeu em 890ms
  [WARN]  YOLO local falhou: model not loaded. Tentando Groq Vision...
  [OK]    Groq Vision respondeu em 1240ms
  [ERRO]  Todos os 3 níveis falharam. Retornando erro 503.
"""

import base64
import logging
import time

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException

from ..services.image_processor import validate_and_resize, image_to_base64
from ..services.roboflow_client import run_inference
from ..services.yolo_local import run_inference_local
from ..services.gemini_fallback import run_inference_gemini, gemini_disponivel
from ..schemas.detection import DetectionResponse
from ..core.config import get_settings, Settings

router = APIRouter(prefix="/api/v1", tags=["inference"])
logger = logging.getLogger(__name__)

# ── Throttle / cache simples ──────────────────────────────────────────────────
_ultimo_resultado: dict | None = None
_ultimo_tempo: float = 0.0


# ── Cadeia de fallback (núcleo da lógica) ────────────────────────────────────

async def _run_com_fallback(
    image_b64: str,
    image_bytes: bytes,
) -> DetectionResponse:
    """
    Tenta os 3 níveis em sequência.
    Retorna o resultado do primeiro que funcionar.
    Lança HTTPException 503 somente se os 3 falharem.
    """
    global _ultimo_resultado, _ultimo_tempo

    # ── Throttle: retorna cache se a última chamada foi há menos de 3s ─────────
    if _ultimo_resultado is not None and time.time() - _ultimo_tempo < 3.0:
        return DetectionResponse(**_ultimo_resultado)

    erros = []

    # ── NÍVEL 1: Roboflow ─────────────────────────────────────────────────────
    try:
        t = time.time()
        resultado = await run_inference(image_b64)
        if resultado.total_animals == 0:
            logger.warning("[WARN] Roboflow retornou 0 animais. Tentando YOLO local...")
            raise Exception("Roboflow retornou resultado vazio")
        logger.info(f"[OK] Roboflow respondeu em {(time.time()-t)*1000:.0f}ms")
        _ultimo_resultado = resultado.model_dump()
        _ultimo_tempo = time.time()
        return resultado
    except Exception as e:
        msg = f"Roboflow: {type(e).__name__} — {e}"
        erros.append(msg)
        logger.warning(f"[WARN] {msg}. Tentando YOLO local...")

    # ── NÍVEL 2: YOLO local ───────────────────────────────────────────────────
    try:
        t = time.time()
        resultado = await run_inference_local(image_bytes)
        if resultado.total_animals == 0:
            logger.warning("[WARN] YOLO local retornou 0 animais. Tentando Groq Vision...")
            raise Exception("YOLO local retornou resultado vazio")
        logger.info(f"[OK] YOLO local respondeu em {(time.time()-t)*1000:.0f}ms")
        _ultimo_resultado = resultado.model_dump()
        _ultimo_tempo = time.time()
        return resultado
    except Exception as e:
        msg = f"YOLO local: {type(e).__name__} — {e}"
        erros.append(msg)
        logger.warning(f"[WARN] {msg}. Tentando Groq Vision...")

    # ── NÍVEL 3: Groq Vision ──────────────────────────────────────────────────
    if not gemini_disponivel():
        erros.append("Groq: GROQ_API_KEY não configurada no Railway")
        logger.error("[WARN] Groq não configurado — pulando nível 3")
    else:
        try:
            t = time.time()
            dados = await run_inference_gemini(image_bytes)
            logger.info(
                f"[OK] Groq Vision respondeu em {(time.time()-t)*1000:.0f}ms"
            )
            # run_inference_gemini retorna dict — converte para DetectionResponse
            _ultimo_resultado = dados
            _ultimo_tempo = time.time()
            return DetectionResponse(**dados)
        except Exception as e:
            msg = f"Groq: {type(e).__name__} — {e}"
            erros.append(msg)
            logger.error(f"[ERRO] {msg}")

    # ── Todos falharam ────────────────────────────────────────────────────────
    logger.error(f"[ERRO] Todos os 3 níveis falharam: {erros}")
    raise HTTPException(
        status_code=503,
        detail={
            "erro": "Serviço temporariamente indisponível",
            "motivo": "Todos os modelos de detecção falharam",
            "detalhes": erros,
        },
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/detect", response_model=DetectionResponse)
async def detect_behaviors(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    """
    Recebe um frame (imagem JPEG/PNG) como upload de arquivo.

    - Aceita: imagem até 5MB, formatos JPEG/PNG/WEBP
    - Retorna: contagem por comportamento + bounding boxes (quando disponível)
    - Fallback: Roboflow → YOLO local → Gemini Vision
    """
    content = await file.read()

    if len(content) > settings.max_image_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Imagem muito grande. Máximo: {settings.max_image_size_mb}MB",
        )

    processed = validate_and_resize(content)
    image_b64 = image_to_base64(processed)

    return await _run_com_fallback(image_b64, processed)


@router.post("/detect/base64", response_model=DetectionResponse)
async def detect_behaviors_b64(
    payload: dict,
    settings: Settings = Depends(get_settings),
):
    """
    Alternativa: recebe imagem já em base64.
    Útil para o PWA que usa canvas para capturar da câmera.

    Payload esperado: {"image": "base64string..."}
    """
    try:
        image_bytes = base64.b64decode(payload["image"])
    except Exception:
        raise HTTPException(status_code=400, detail="Base64 inválido")

    processed = validate_and_resize(image_bytes)
    image_b64 = image_to_base64(processed)

    return await _run_com_fallback(image_b64, processed)


# ── Endpoint de status dos modelos ───────────────────────────────────────────

@router.get("/models/status")
async def status_modelos():
    """
    Retorna o status de disponibilidade de cada nível.
    Útil para debug e para o painel de admin.

    Exemplo de resposta:
    {
      "roboflow": true,
      "yolo_local": true,
      "gemini": false,
      "gemini_motivo": "GEMINI_API_KEY não configurada"
    }
    """
    import os

    groq_ok = gemini_disponivel()

    return {
        "roboflow": bool(os.environ.get("ROBOFLOW_API_KEY")),
        "yolo_local": True,   # Sempre tenta — falha só em runtime
        "groq": groq_ok,
        "groq_motivo": None if groq_ok else "GROQ_API_KEY não configurada no Railway",
        "ordem_fallback": ["roboflow", "yolo_local", "groq"],
    }
