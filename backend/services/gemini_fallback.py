"""
backend/services/gemini_fallback.py

Terceiro nível de fallback do Bovinsights.
Chamado quando Roboflow E o modelo local falham.

Fluxo completo:
  1º → Roboflow (modelo especializado bovino)
  2º → YOLO local (best.pt no servidor)
  3º → Gemini Vision (este arquivo) ← você está aqui

Como funciona:
  - Envia a imagem para o Gemini 2.0 Flash (gratuito)
  - O Gemini analisa visualmente e retorna JSON estruturado
  - O JSON é convertido para DetectionResponse (mesmo formato do Roboflow)
  - O frontend não percebe diferença — recebe sempre o mesmo schema

Instalação:
  pip install google-generativeai pillow

Variável de ambiente necessária no Railway:
  GEMINI_API_KEY=sua_chave_aqui
  (Chave gratuita em: https://aistudio.google.com/app/apikey)
"""

import os
import json
import time
import logging
import io
from typing import Optional

import google.generativeai as genai
from PIL import Image

logger = logging.getLogger(__name__)

# ── Configuração do modelo ────────────────────────────────────────────────────

_model: Optional[genai.GenerativeModel] = None


def _get_model() -> genai.GenerativeModel:
    """Inicializa o modelo Gemini uma única vez (singleton)."""
    global _model
    if _model is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY não encontrada. "
                "Adicione essa variável no Railway → Variables."
            )
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel("gemini-2.0-flash")
        logger.info("[Gemini] Modelo inicializado: gemini-2.0-flash")
    return _model


# ── Prompt do sistema ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
Você é um sistema de visão computacional especializado em análise de confinamento bovino.
Analise a imagem e identifique todos os bovinos visíveis.

Classifique cada animal em UMA das seguintes categorias:
- BOI_EM_PE: animal em pé com postura normal, sem sinais visíveis de problema
- ANIMAL_COMENDO: animal com cabeça dentro do cocho ou claramente se alimentando
- BOI_REFUGO: animal afastado do cocho enquanto os outros estão comendo
- ISOLADO: animal sozinho em canto afastado do grupo principal
- POSTURA_ANORMAL: animal em pé mas com postura suspeita (cabeça muito baixa travada, coluna arqueada, peso em 3 patas)
- DEITADO: animal deitado no chão
- XIBUNGO: dois animais em comportamento de monta/sodomia

Retorne APENAS um JSON válido, sem texto antes ou depois, no seguinte formato:
{
  "total_animais": <número inteiro>,
  "nivel_alerta": "<VERDE|ATENCAO|CRITICO>",
  "resumo": "<frase curta em português descrevendo a cena>",
  "contagem": {
    "BOI_EM_PE": <número>,
    "ANIMAL_COMENDO": <número>,
    "BOI_REFUGO": <número>,
    "ISOLADO": <número>,
    "POSTURA_ANORMAL": <número>,
    "DEITADO": <número>,
    "XIBUNGO": <número>
  },
  "alertas": [
    {
      "tipo": "<classe>",
      "quantidade": <número>,
      "descricao": "<descrição curta do que foi visto>"
    }
  ],
  "confianca_geral": <número de 0 a 100 indicando sua certeza na análise>
}

Regras para nivel_alerta:
- VERDE: todos BOI_EM_PE ou ANIMAL_COMENDO, sem anomalias
- ATENCAO: há ISOLADO, DEITADO em excesso, ou BOI_REFUGO
- CRITICO: há POSTURA_ANORMAL, XIBUNGO, ou BOI_REFUGO com mais de 2 animais

Se a imagem não mostrar bovinos claramente, retorne total_animais: 0 e nivel_alerta: VERDE.
"""


# ── Função principal ──────────────────────────────────────────────────────────

async def run_inference_gemini(image_bytes: bytes) -> dict:
    """
    Analisa uma imagem de confinamento usando Gemini Vision.

    Args:
        image_bytes: bytes da imagem (JPEG ou PNG)

    Returns:
        dict compatível com DetectionResponse do Bovinsights

    Raises:
        Exception: se a chamada ao Gemini falhar ou retornar JSON inválido
    """
    t_inicio = time.time()

    # Converte bytes para PIL Image (necessário para o Gemini)
    try:
        imagem_pil = Image.open(io.BytesIO(image_bytes))
        # Garante RGB (Gemini não aceita RGBA ou outros modos)
        if imagem_pil.mode != "RGB":
            imagem_pil = imagem_pil.convert("RGB")
    except Exception as e:
        raise ValueError(f"Imagem inválida: {e}")

    # Chama o Gemini Vision
    model = _get_model()
    logger.info("[Gemini] Enviando imagem para análise...")

    try:
        response = model.generate_content(
            contents=[imagem_pil, SYSTEM_PROMPT],
            generation_config=genai.GenerationConfig(
                temperature=0.1,        # Baixo = mais consistente e literal
                max_output_tokens=800,  # Suficiente para o JSON completo
            ),
        )
    except Exception as e:
        raise RuntimeError(f"Erro na chamada ao Gemini: {e}")

    # Extrai e valida o JSON da resposta
    texto_resposta = response.text.strip()

    # Remove markdown code blocks se o Gemini os incluir (```json ... ```)
    if texto_resposta.startswith("```"):
        linhas = texto_resposta.split("\n")
        texto_resposta = "\n".join(linhas[1:-1])

    try:
        dados = json.loads(texto_resposta)
    except json.JSONDecodeError as e:
        logger.error(f"[Gemini] JSON inválido recebido: {texto_resposta[:200]}")
        raise ValueError(f"Gemini retornou resposta não-JSON: {e}")

    tempo_ms = (time.time() - t_inicio) * 1000
    logger.info(
        f"[Gemini] Análise concluída em {tempo_ms:.0f}ms — "
        f"{dados.get('total_animais', 0)} animais, "
        f"alerta: {dados.get('nivel_alerta', 'VERDE')}"
    )

    # Converte para o formato DetectionResponse do Bovinsights
    return _converter_para_detection_response(dados, tempo_ms)


def _converter_para_detection_response(dados: dict, tempo_ms: float) -> dict:
    """
    Converte o JSON do Gemini para o formato DetectionResponse
    que o frontend já conhece.

    O frontend não sabe (nem precisa saber) que foi o Gemini
    que fez a análise — recebe o mesmo formato de sempre.
    """
    contagem = dados.get("contagem", {})

    return {
        "success": True,
        "total_animals": dados.get("total_animais", 0),
        "behaviors": {
            # Campos originais (compatibilidade com frontend v1)
            "eating":   contagem.get("ANIMAL_COMENDO", 0),
            "standing": contagem.get("BOI_EM_PE", 0),
            # Campos novos (classes v2)
            "refugo":          contagem.get("BOI_REFUGO", 0),
            "isolado":         contagem.get("ISOLADO", 0),
            "postura_anormal": contagem.get("POSTURA_ANORMAL", 0),
            "deitado":         contagem.get("DEITADO", 0),
            "xibungo":         contagem.get("XIBUNGO", 0),
        },
        # Detections vazio — Gemini não retorna bounding boxes
        # (apenas contagens e classificações)
        "detections": [],
        "inference_time_ms": round(tempo_ms, 1),
        # Identifica claramente que foi o Gemini — útil para debug e logs
        "model_used": "gemini-vision-fallback",
        # Campos extras do Bovinsights v2
        "nivel_alerta":    dados.get("nivel_alerta", "VERDE"),
        "resumo":          dados.get("resumo", ""),
        "alertas":         dados.get("alertas", []),
        "confianca_gemini": dados.get("confianca_geral", 0),
    }


# ── Verificação de disponibilidade ────────────────────────────────────────────

def gemini_disponivel() -> bool:
    """
    Verifica se o Gemini está configurado e pronto para uso.
    Útil para o endpoint /health mostrar o status do fallback.
    """
    return bool(os.environ.get("GEMINI_API_KEY"))
