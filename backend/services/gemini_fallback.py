import os, json, time, logging, base64
from groq import Groq

logger = logging.getLogger(__name__)
_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY não encontrada no Railway.")
        _client = Groq(api_key=api_key)
    return _client

SYSTEM_PROMPT = """Você é especialista em confinamento bovino. Analise a imagem e retorne APENAS JSON válido:
{"total_animais": 0, "nivel_alerta": "VERDE", "resumo": "", "contagem": {"BOI_EM_PE": 0, "ANIMAL_COMENDO": 0, "BOI_REFUGO": 0, "ISOLADO": 0, "POSTURA_ANORMAL": 0, "DEITADO": 0, "XIBUNGO": 0}, "ambiente": {"cocho_status": "NAO_VISIVEL", "fezes_cor": "NAO_VISIVEL", "fezes_consistencia": "NAO_VISIVEL"}, "alertas": [], "confianca_geral": 0}"""

async def run_inference_gemini(image_bytes: bytes) -> dict:
    t_inicio = time.time()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    client = _get_client()
    logger.info("[Groq] Enviando imagem para análise...")
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]}],
            temperature=0.1,
            max_tokens=800,
        )
    except Exception as e:
        raise RuntimeError(f"Erro na chamada ao Groq: {e}")

    texto = completion.choices[0].message.content.strip()
    if texto.startswith("```"):
        texto = "\n".join(texto.split("\n")[1:-1])
    try:
        dados = json.loads(texto)
    except json.JSONDecodeError as e:
        raise ValueError(f"Groq retornou resposta não-JSON: {e}")

    tempo_ms = (time.time() - t_inicio) * 1000
    contagem = dados.get("contagem", {})
    return {
        "success": True,
        "total_animals": dados.get("total_animais", 0),
        "behaviors": {
            "eating": contagem.get("ANIMAL_COMENDO", 0),
            "standing": contagem.get("BOI_EM_PE", 0),
            "refugo": contagem.get("BOI_REFUGO", 0),
            "isolado": contagem.get("ISOLADO", 0),
            "postura_anormal": contagem.get("POSTURA_ANORMAL", 0),
            "deitado": contagem.get("DEITADO", 0),
            "xibungo": contagem.get("XIBUNGO", 0),
        },
        "detections": [],
        "inference_time_ms": round(tempo_ms, 1),
        "model_used": "groq-vision-fallback",
        "nivel_alerta": dados.get("nivel_alerta", "VERDE"),
        "resumo": dados.get("resumo", ""),
        "alertas": dados.get("alertas", []),
        "confianca_gemini": dados.get("confianca_geral", 0),
    }

def gemini_disponivel() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))
