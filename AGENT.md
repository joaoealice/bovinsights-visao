# BOVISIGHTS — SERVIDOR DE VISÃO COMPUTACIONAL
## Guia Completo para o Agente Claude (VS Code)

> **Leia este arquivo inteiro antes de escrever qualquer código.**
> Este documento define arquitetura, decisões técnicas, ordem de implementação e todos os detalhes necessários para construir o sistema do zero.

---

## 1. CONTEXTO DO PROJETO

### O que é o Bovisights CV Server
Um servidor de visão computacional que detecta comportamentos de bovinos em confinamento em tempo real. O cliente abre o app no celular, aponta a câmera para o lote, e o sistema retorna: quais animais estão comendo, deitados, ruminando, em pé, bebendo.

### Fase Atual: MVP do Celular
**Escopo desta implementação:**
- Backend API (FastAPI) que recebe frames do celular e retorna comportamentos detectados
- Progressive Web App (PWA) que roda no celular do cliente — sem instalação de app
- Modelo de IA pré-treinado do Roboflow Universe (mAP 88%) — zero treinamento necessário agora
- Deploy simples na nuvem (Railway.app — gratuito para MVP)

**Fora do escopo agora (implementar depois):**
- Câmeras IP fixas nos piquetes
- Rastreamento contínuo com BoT-SORT
- Identificação individual de animais
- Banco de dados histórico
- Integração com sistema de gestão Bovisights

### Como o MVP funciona
```
[Celular do cliente]
  1. Abre https://cv.bovisights.com.br no browser
  2. Aceita acesso à câmera
  3. Aponta para o lote de bovinos
  4. App envia 1 frame a cada 3 segundos para a API
  5. API chama modelo Roboflow Universe
  6. Retorna: {"eating": 3, "lying": 2, "standing": 5, "drinking": 1}
  7. App exibe resultado na tela em tempo real
```

---

## 2. MODELO DE IA — ROBOFLOW UNIVERSE

### Modelo Selecionado
**Cow Behavior Tracking** — universe.roboflow.com/cow-detection-identifier/cow-behavior-tracking
- mAP: 88.0% | Precision: 78.0% | Recall: 81.1%
- Treinado com 908 imagens de bovinos reais
- Classes detectadas: eating, lying, standing, drinking (+ variações)
- API hospedada pelo Roboflow — sem precisar de GPU local

### Como usar a API do Roboflow
```python
import requests
import base64

def detect_behaviors(image_bytes: bytes, api_key: str) -> dict:
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    response = requests.post(
        "https://detect.roboflow.com/cow-behavior-tracking/1",
        params={"api_key": api_key},
        data=image_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    return response.json()
```

### Chave de API
- Criar conta gratuita em roboflow.com
- Free tier: 10.000 inferências/mês (suficiente para MVP)
- Variável de ambiente: `ROBOFLOW_API_KEY`

### Fallback: Ultralytics YOLOv11 (sem internet)
Se o Roboflow estiver indisponível, usar modelo local:
```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")  # nano — leve, roda sem GPU
results = model.predict(image, classes=[19])  # classe 19 = cow no COCO
```

---

## 3. STACK TECNOLÓGICO

| Componente | Tecnologia | Versão | Motivo |
|-----------|-----------|--------|--------|
| Backend API | FastAPI | 0.111+ | Async, rápido, mesmo ecossistema Python da IA |
| Servidor ASGI | Uvicorn | 0.30+ | Servidor de produção para FastAPI |
| Validação | Pydantic v2 | 2.7+ | Validação de dados automática |
| IA Inference | Roboflow API | hosted | Modelo pré-treinado, sem GPU necessária |
| IA Fallback | Ultralytics YOLOv11 | 8.2+ | Offline, CPU, modelo nano |
| Frontend | HTML + Vanilla JS | — | PWA simples, sem frameworks |
| Estilização | Tailwind CSS CDN | 3.x | Sem build necessário |
| Deploy | Railway.app | — | Gratuito, deploy via GitHub |
| Env vars | python-dotenv | 1.0+ | Variáveis de ambiente locais |
| HTTP client | httpx | 0.27+ | Async HTTP para chamar Roboflow |
| CORS | fastapi.middleware.cors | — | Celular pode chamar a API |
| Imagem | Pillow | 10.x | Processar frames recebidos |

---

## 4. ESTRUTURA DE PASTAS

```
bovisights-cv/
│
├── backend/
│   ├── main.py                  # Entrada FastAPI — define app e rotas
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── inference.py         # POST /detect — recebe frame, retorna comportamentos
│   │   └── health.py            # GET /health — status da API
│   ├── services/
│   │   ├── __init__.py
│   │   ├── roboflow_client.py   # Integração com Roboflow API
│   │   ├── yolo_local.py        # Fallback com Ultralytics local
│   │   └── image_processor.py   # Redimensionar, validar frames recebidos
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── detection.py         # Pydantic models: DetectionRequest, DetectionResponse
│   └── core/
│       ├── __init__.py
│       └── config.py            # Configurações via env vars
│
├── frontend/
│   ├── index.html               # PWA — única página, funciona no celular
│   ├── manifest.json            # PWA manifest — permite instalar no celular
│   ├── sw.js                    # Service Worker — cache offline
│   └── assets/
│       ├── icon-192.png         # Ícone do app (192x192)
│       └── icon-512.png         # Ícone do app (512x512)
│
├── tests/
│   ├── test_inference.py        # Testar endpoint /detect
│   └── test_image_processor.py  # Testar processamento de imagens
│
├── .env                         # NÃO commitar — variáveis locais
├── .env.example                 # Commitar — template sem valores
├── .gitignore
├── requirements.txt
├── Procfile                     # Para Railway.app: web: uvicorn backend.main:app
├── railway.json                 # Config de deploy Railway
└── README.md
```

---

## 5. CONFIGURAÇÃO DO AMBIENTE LOCAL

### Pré-requisitos
```bash
python --version   # Deve ser 3.11+
node --version     # Deve ser 20+
git --version      # Qualquer versão
```

### Setup inicial (executar uma vez)
```bash
# 1. Clonar o repositório (criar no GitHub primeiro)
git clone https://github.com/SEU_USUARIO/bovisights-cv.git
cd bovisights-cv

# 2. Criar ambiente virtual Python
python -m venv venv

# 3. Ativar o ambiente virtual
# No Windows:
venv\Scripts\activate
# No Mac/Linux:
source venv/bin/activate

# 4. Instalar dependências
pip install -r requirements.txt

# 5. Copiar e preencher variáveis de ambiente
cp .env.example .env
# Editar .env com suas chaves

# 6. Rodar o servidor localmente
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Testar se está funcionando
```bash
# No terminal, com o servidor rodando:
curl http://localhost:8000/health
# Esperado: {"status": "ok", "model": "roboflow", "version": "1.0.0"}
```

---

## 6. ARQUIVO requirements.txt

```txt
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-dotenv==1.0.1
httpx==0.27.0
Pillow==10.3.0
pydantic==2.7.1
python-multipart==0.0.9
ultralytics==8.2.18
```

---

## 7. ARQUIVO .env.example

```env
# Roboflow
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_MODEL_ID=cow-behavior-tracking
ROBOFLOW_MODEL_VERSION=1

# API Config
API_ENV=development
API_SECRET_KEY=change_this_in_production
MAX_IMAGE_SIZE_MB=5
INFERENCE_INTERVAL_SECONDS=3

# CORS — domínios que podem chamar a API
ALLOWED_ORIGINS=http://localhost:3000,https://cv.bovisights.com.br
```

---

## 8. IMPLEMENTAÇÃO DO BACKEND

### backend/core/config.py
```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    roboflow_api_key: str
    roboflow_model_id: str = "cow-behavior-tracking"
    roboflow_model_version: int = 1
    api_env: str = "development"
    max_image_size_mb: int = 5
    inference_interval_seconds: int = 3
    allowed_origins: str = "http://localhost:3000"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

### backend/schemas/detection.py
```python
from pydantic import BaseModel
from typing import Dict, List, Optional

class BehaviorCount(BaseModel):
    eating: int = 0
    lying: int = 0
    standing: int = 0
    drinking: int = 0
    ruminating: int = 0
    running: int = 0
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
```

### backend/services/image_processor.py
```python
from PIL import Image
import io
import base64
from fastapi import HTTPException

MAX_SIZE = (1280, 1280)

def validate_and_resize(image_bytes: bytes) -> bytes:
    """
    Valida que é uma imagem válida, redimensiona para max 1280px
    e retorna bytes JPEG otimizados para inferência.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Arquivo não é uma imagem válida")

    # Converter para RGB (remove canal alpha se houver)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Redimensionar mantendo proporção
    img.thumbnail(MAX_SIZE, Image.LANCZOS)

    # Converter para JPEG bytes
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=85)
    return output.getvalue()

def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")
```

### backend/services/roboflow_client.py
```python
import httpx
import time
from typing import Optional
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
    "ruminating": "ruminating",
    "rumination": "ruminating",
    "running": "running",
    "walking": "running",
}

async def run_inference(image_b64: str) -> DetectionResponse:
    settings = get_settings()
    url = (
        f"https://detect.roboflow.com/"
        f"{settings.roboflow_model_id}/{settings.roboflow_model_version}"
    )

    start = time.time()
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            url,
            params={"api_key": settings.roboflow_api_key},
            data=image_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    elapsed_ms = (time.time() - start) * 1000

    if response.status_code != 200:
        raise Exception(f"Roboflow API error: {response.status_code} — {response.text}")

    data = response.json()
    predictions = data.get("predictions", [])

    # Contar comportamentos
    counts = BehaviorCount()
    boxes = []

    for pred in predictions:
        raw_label = pred.get("class", "unknown").lower()
        label = LABEL_MAP.get(raw_label, "unknown")

        # Incrementar contador
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

    total = sum([
        counts.eating, counts.lying, counts.standing,
        counts.drinking, counts.ruminating, counts.running, counts.unknown
    ])

    return DetectionResponse(
        success=True,
        total_animals=total,
        behaviors=counts,
        detections=boxes,
        inference_time_ms=round(elapsed_ms, 1),
        model_used="roboflow",
    )
```

### backend/services/yolo_local.py
```python
"""
Fallback local usando YOLOv11 nano (sem GPU, sem internet).
Menos preciso para comportamentos, mas garante disponibilidade.
Detecta apenas "cow" (classe 19 do COCO) sem classes de comportamento.
"""
import io
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO
from functools import lru_cache
from ..schemas.detection import DetectionResponse, DetectionBox, BehaviorCount

@lru_cache(maxsize=1)
def get_model():
    return YOLO("yolo11n.pt")  # Baixa automaticamente na primeira execução

async def run_inference_local(image_bytes: bytes) -> DetectionResponse:
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
```

### backend/routers/inference.py
```python
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from ..services.image_processor import validate_and_resize, image_to_base64
from ..services.roboflow_client import run_inference
from ..services.yolo_local import run_inference_local
from ..schemas.detection import DetectionResponse
from ..core.config import get_settings, Settings

router = APIRouter(prefix="/api/v1", tags=["inference"])

@router.post("/detect", response_model=DetectionResponse)
async def detect_behaviors(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    """
    Recebe um frame (imagem JPEG/PNG) e retorna os comportamentos
    detectados nos bovinos presentes na imagem.

    - Aceita: imagem até 5MB, formatos JPEG/PNG/WEBP
    - Retorna: contagem por comportamento + bounding boxes
    """
    # Validar tamanho
    content = await file.read()
    if len(content) > settings.max_image_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Imagem muito grande. Máximo: {settings.max_image_size_mb}MB"
        )

    # Processar imagem
    processed = validate_and_resize(content)
    image_b64 = image_to_base64(processed)

    # Tentar Roboflow primeiro, fallback para local
    try:
        result = await run_inference(image_b64)
    except Exception as e:
        print(f"[WARN] Roboflow falhou: {e}. Usando modelo local.")
        result = await run_inference_local(processed)

    return result


@router.post("/detect/base64", response_model=DetectionResponse)
async def detect_behaviors_b64(
    payload: dict,
    settings: Settings = Depends(get_settings),
):
    """
    Alternativa: recebe imagem já em base64 (útil para PWA com canvas).
    Payload: {"image": "base64string..."}
    """
    import base64
    try:
        image_bytes = base64.b64decode(payload["image"])
    except Exception:
        raise HTTPException(status_code=400, detail="Base64 inválido")

    processed = validate_and_resize(image_bytes)
    image_b64 = image_to_base64(processed)

    try:
        result = await run_inference(image_b64)
    except Exception as e:
        print(f"[WARN] Roboflow falhou: {e}. Usando modelo local.")
        result = await run_inference_local(processed)

    return result
```

### backend/routers/health.py
```python
from fastapi import APIRouter
from ..core.config import get_settings

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    settings = get_settings()
    return {
        "status": "ok",
        "model": "roboflow",
        "model_id": settings.roboflow_model_id,
        "version": "1.0.0",
        "env": settings.api_env,
    }
```

### backend/main.py
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .core.config import get_settings
from .routers import inference, health
import os

settings = get_settings()

app = FastAPI(
    title="Bovisights CV Server",
    description="API de detecção de comportamento bovino por visão computacional",
    version="1.0.0",
)

# CORS — permite que o celular do cliente chame a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rotas da API
app.include_router(health.router)
app.include_router(inference.router)

# Servir o frontend (PWA) como arquivos estáticos
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
```

---

## 9. IMPLEMENTAÇÃO DO FRONTEND (PWA)

### frontend/index.html
```html
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
  <meta name="theme-color" content="#16a34a">
  <title>Bovisights — Detecção de Comportamento</title>
  <link rel="manifest" href="/manifest.json">
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    #video { transform: scaleX(-1); } /* espelhar câmera frontal */
    .behavior-badge { @apply inline-flex items-center px-3 py-1 rounded-full text-sm font-medium; }
    #overlay { position: absolute; top: 0; left: 0; pointer-events: none; }
  </style>
</head>
<body class="bg-gray-900 text-white min-h-screen flex flex-col">

  <!-- Header -->
  <header class="bg-green-700 px-4 py-3 flex items-center justify-between">
    <h1 class="text-lg font-bold">🐄 Bovisights CV</h1>
    <span id="status-badge" class="text-xs px-2 py-1 rounded-full bg-gray-700">Aguardando...</span>
  </header>

  <!-- Camera Container -->
  <div class="relative flex-1 flex flex-col items-center justify-center bg-black">
    <div class="relative w-full max-w-lg">
      <video id="video" class="w-full rounded-lg" autoplay playsinline muted></video>
      <canvas id="overlay" class="w-full rounded-lg"></canvas>
      <canvas id="canvas" class="hidden"></canvas>
    </div>

    <!-- Botão iniciar -->
    <button id="btn-start"
      class="mt-4 px-8 py-3 bg-green-600 hover:bg-green-500 rounded-full text-lg font-semibold transition">
      📷 Iniciar Detecção
    </button>
  </div>

  <!-- Resultados -->
  <div class="bg-gray-800 p-4">
    <div id="results-empty" class="text-center text-gray-400 py-4">
      Aponte a câmera para o lote e pressione Iniciar
    </div>
    <div id="results" class="hidden">
      <div class="flex items-center justify-between mb-3">
        <span class="font-semibold text-green-400">Animais detectados</span>
        <span id="total-count" class="text-2xl font-bold text-white">0</span>
      </div>
      <div id="behavior-counts" class="grid grid-cols-3 gap-2 mb-3"></div>
      <div class="text-xs text-gray-500 text-right">
        <span id="inference-time">—</span>ms · <span id="model-name">—</span>
      </div>
    </div>
  </div>

  <script>
    const API_URL = window.location.origin;
    const INTERVAL_MS = 3000; // enviar frame a cada 3 segundos

    const BEHAVIOR_CONFIG = {
      eating:     { label: "Comendo",    icon: "🌿", color: "bg-green-600" },
      lying:      { label: "Deitado",    icon: "💤", color: "bg-blue-600" },
      standing:   { label: "Em pé",      icon: "🐄", color: "bg-gray-600" },
      drinking:   { label: "Bebendo",    icon: "💧", color: "bg-cyan-600" },
      ruminating: { label: "Ruminando",  icon: "🔄", color: "bg-purple-600" },
      running:    { label: "Agitado",    icon: "⚡", color: "bg-red-600" },
      unknown:    { label: "Outro",      icon: "❓", color: "bg-yellow-600" },
    };

    let stream = null;
    let intervalId = null;
    let isRunning = false;

    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const overlay = document.getElementById("overlay");
    const btnStart = document.getElementById("btn-start");
    const statusBadge = document.getElementById("status-badge");
    const results = document.getElementById("results");
    const resultsEmpty = document.getElementById("results-empty");
    const totalCount = document.getElementById("total-count");
    const behaviorCounts = document.getElementById("behavior-counts");
    const inferenceTime = document.getElementById("inference-time");
    const modelName = document.getElementById("model-name");

    // Iniciar câmera
    btnStart.addEventListener("click", async () => {
      if (isRunning) {
        stopDetection();
        return;
      }

      try {
        // Tentar câmera traseira primeiro (melhor para apontar para o lote)
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } }
        });
        video.srcObject = stream;
        video.play();
        startDetection();
      } catch (err) {
        alert("Não foi possível acessar a câmera: " + err.message);
      }
    });

    function startDetection() {
      isRunning = true;
      btnStart.textContent = "⏹ Parar";
      btnStart.classList.replace("bg-green-600", "bg-red-600");
      setStatus("Detectando...", "bg-green-600");
      intervalId = setInterval(captureAndDetect, INTERVAL_MS);
      captureAndDetect(); // executar imediatamente
    }

    function stopDetection() {
      isRunning = false;
      clearInterval(intervalId);
      if (stream) { stream.getTracks().forEach(t => t.stop()); }
      btnStart.textContent = "📷 Iniciar Detecção";
      btnStart.classList.replace("bg-red-600", "bg-green-600");
      setStatus("Parado", "bg-gray-700");
      clearOverlay();
    }

    async function captureAndDetect() {
      if (!video.readyState || video.readyState < 2) return;

      // Capturar frame do vídeo
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);

      // Converter para base64 JPEG
      const imageB64 = canvas.toDataURL("image/jpeg", 0.8).split(",")[1];

      try {
        setStatus("Analisando...", "bg-yellow-600");
        const response = await fetch(`${API_URL}/api/v1/detect/base64`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: imageB64 }),
        });

        if (!response.ok) throw new Error("API retornou " + response.status);
        const data = await response.json();
        renderResults(data);
        drawBoxes(data.detections, canvas.width, canvas.height);
        setStatus("✓ Ativo", "bg-green-600");
      } catch (err) {
        console.error("Erro na detecção:", err);
        setStatus("Erro de conexão", "bg-red-700");
      }
    }

    function renderResults(data) {
      resultsEmpty.classList.add("hidden");
      results.classList.remove("hidden");
      totalCount.textContent = data.total_animals;
      inferenceTime.textContent = data.inference_time_ms.toFixed(0);
      modelName.textContent = data.model_used;

      behaviorCounts.innerHTML = "";
      for (const [key, cfg] of Object.entries(BEHAVIOR_CONFIG)) {
        const count = data.behaviors[key] || 0;
        if (count === 0) continue;
        const div = document.createElement("div");
        div.className = `${cfg.color} rounded-lg p-2 text-center`;
        div.innerHTML = `
          <div class="text-xl">${cfg.icon}</div>
          <div class="text-2xl font-bold">${count}</div>
          <div class="text-xs opacity-80">${cfg.label}</div>
        `;
        behaviorCounts.appendChild(div);
      }
    }

    function drawBoxes(detections, imgW, imgH) {
      const displayW = video.offsetWidth;
      const displayH = video.offsetHeight;
      overlay.width = displayW;
      overlay.height = displayH;
      const ctx = overlay.getContext("2d");
      ctx.clearRect(0, 0, displayW, displayH);

      const scaleX = displayW / imgW;
      const scaleY = displayH / imgH;

      for (const det of detections) {
        const cfg = BEHAVIOR_CONFIG[det.label] || BEHAVIOR_CONFIG.unknown;
        const x = (det.x - det.width / 2) * scaleX;
        const y = (det.y - det.height / 2) * scaleY;
        const w = det.width * scaleX;
        const h = det.height * scaleY;

        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = "rgba(22, 163, 74, 0.7)";
        ctx.fillRect(x, y - 20, w, 20);
        ctx.fillStyle = "white";
        ctx.font = "12px Arial";
        ctx.fillText(`${cfg.icon} ${cfg.label} ${(det.confidence * 100).toFixed(0)}%`, x + 4, y - 5);
      }
    }

    function clearOverlay() {
      const ctx = overlay.getContext("2d");
      ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    function setStatus(text, colorClass) {
      statusBadge.textContent = text;
      statusBadge.className = `text-xs px-2 py-1 rounded-full ${colorClass}`;
    }

    // Registrar Service Worker (PWA)
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(console.error);
    }
  </script>
</body>
</html>
```

### frontend/manifest.json
```json
{
  "name": "Bovisights CV",
  "short_name": "Bovisights",
  "description": "Detecção de comportamento bovino por visão computacional",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#111827",
  "theme_color": "#16a34a",
  "orientation": "portrait",
  "icons": [
    { "src": "/assets/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/assets/icon-512.png", "sizes": "512x512", "type": "image/png" }
  ]
}
```

### frontend/sw.js
```javascript
// Service Worker mínimo — apenas registra para habilitar o PWA
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", () => self.clients.claim());
// Cache offline pode ser adicionado futuramente
```

---

## 10. DEPLOY NO RAILWAY.APP

### O que é o Railway
Railway.app é uma plataforma de deploy gratuita para projetos pequenos. Você conecta o GitHub, ele faz o build e o deploy automaticamente a cada push.

### Pré-requisitos
- Conta no Railway.app (gratuita)
- Repositório no GitHub com o projeto

### Arquivos necessários para o deploy

**Procfile** (raiz do projeto):
```
web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

**railway.json** (raiz do projeto):
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn backend.main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

### Configurar variáveis de ambiente no Railway
No painel do Railway → seu projeto → Variables, adicionar:
```
ROBOFLOW_API_KEY = sua_chave_aqui
ROBOFLOW_MODEL_ID = cow-behavior-tracking
ROBOFLOW_MODEL_VERSION = 1
API_ENV = production
ALLOWED_ORIGINS = https://SEU_DOMINIO_RAILWAY.up.railway.app
```

### Deploy
```bash
# 1. Instalar Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Vincular projeto
railway link

# 4. Deploy
railway up
```

Após o deploy, Railway fornece uma URL como `https://bovisights-cv-production.up.railway.app`. Acesse pelo celular — o PWA já funciona.

---

## 11. TESTES

### tests/test_inference.py
```python
import pytest
from fastapi.testclient import TestClient
from backend.main import app
import base64
from PIL import Image
import io

client = TestClient(app)

def create_test_image() -> bytes:
    """Cria uma imagem verde 640x480 para testes."""
    img = Image.new("RGB", (640, 480), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_detect_base64():
    img_bytes = create_test_image()
    img_b64 = base64.b64encode(img_bytes).decode()
    response = client.post("/api/v1/detect/base64", json={"image": img_b64})
    assert response.status_code == 200
    data = response.json()
    assert "total_animals" in data
    assert "behaviors" in data
    assert "model_used" in data

def test_detect_upload():
    img_bytes = create_test_image()
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
```

### Rodar testes
```bash
pip install pytest
pytest tests/ -v
```

---

## 12. CHECKLIST DE IMPLEMENTAÇÃO

Use este checklist para acompanhar o progresso:

### Fase 1 — Setup
- [ ] Repositório criado no GitHub
- [ ] Ambiente virtual Python criado e ativado
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Arquivo `.env` configurado com chave do Roboflow
- [ ] Servidor rodando localmente (`uvicorn backend.main:app --reload`)
- [ ] Endpoint `/health` retornando `{"status": "ok"}`

### Fase 2 — Backend
- [ ] `config.py` funcionando (lê variáveis do `.env`)
- [ ] `image_processor.py` redimensionando imagens corretamente
- [ ] `roboflow_client.py` chamando API e recebendo detecções
- [ ] `yolo_local.py` funcionando como fallback
- [ ] Endpoint `POST /api/v1/detect` aceitando upload e retornando JSON
- [ ] Endpoint `POST /api/v1/detect/base64` funcionando
- [ ] Testes passando (`pytest tests/ -v`)

### Fase 3 — Frontend
- [ ] `index.html` servido pelo FastAPI na rota `/`
- [ ] Câmera abrindo no browser (desktop)
- [ ] Câmera abrindo no celular (Chrome Mobile)
- [ ] Frame sendo enviado para a API
- [ ] Resultados aparecendo na tela
- [ ] Bounding boxes sendo desenhados sobre o vídeo
- [ ] PWA instalável (ícone aparece na tela inicial do celular)

### Fase 4 — Deploy
- [ ] Projeto conectado ao Railway.app
- [ ] Variáveis de ambiente configuradas no Railway
- [ ] Deploy bem-sucedido (sem erros no log)
- [ ] URL pública acessível no celular
- [ ] Detecção funcionando via URL pública

---

## 13. PRÓXIMAS FASES (Não implementar agora)

### Fase 2 do Produto — Banco de Dados e Histórico
- PostgreSQL + TimescaleDB para séries temporais de comportamento
- Cadastro de fazendas e usuários
- Histórico por sessão de monitoramento
- Autenticação com JWT

### Fase 3 — Identificação Individual
- YOLOv8 para segmentar o focinho do animal
- InceptionV3 para extrair embeddings faciais
- SVM para classificar a identidade (0.999 de acurácia — validado UEMS 2023)
- Cadastro biométrico no onboarding (20+ fotos por animal)

### Fase 4 — Câmeras Fixas nos Piquetes
- MediaMTX para RTSP → HLS
- BoT-SORT para rastreamento contínuo
- Cloudflare Tunnel para acesso externo seguro
- Processamento 24h no servidor local da fazenda

### Fase 5 — Integração Bovisights Gestão
- Conectar alertas ao módulo de piquetes e animais
- Dashboard com mapa LEGO do confinamento
- Notificações push (Web Push API)

---

## 14. REFERÊNCIAS E RECURSOS

### Datasets Disponíveis
- **CBVD-5** (206.100 imagens, 5 comportamentos): kaggle.com/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset
- **Beef Cattle Behavior** (anotações YOLO prontas): kaggle.com/datasets/lucyfirst/beef-cattle-behavior-data-set
- **MmCows** (NeurIPS 2024, 4.8M frames): github.com/neis-lab/mmcows
- **BECA** (16.889 imagens para ID individual): nature.com/articles/s41597-025-06326-5

### Modelo Pré-Treinado em Uso
- **Cow Behavior Tracking v1** (mAP 88%): universe.roboflow.com/cow-detection-identifier/cow-behavior-tracking/model/1

### Documentação
- FastAPI: fastapi.tiangolo.com
- Ultralytics YOLOv11: docs.ultralytics.com
- Roboflow Inference: docs.roboflow.com/deploy/hosted-api
- Railway Deploy: docs.railway.app
- PWA Guide: web.dev/learn/pwa

### Estudos Científicos Validadores
- UFRN 2025 (Paulo Ricardo): YOLOv11 + BoT-SORT + Gemini, mAP@50=0.808
- UEMS 2023 (Pietro Claure): Identificação facial Nelore, 0.999 acurácia
- UTFPR 2024 (Vinícius Freitas): YOLOv8 vs YOLOv9 comportamentos bovinos
- UNESP 2025 (Maysa Gomes): Revisão de literatura CI na nutrição animal
- UNEMAT 2023 (Heber Nardes): Contagem aérea Nelore, 89.1% reconhecimento
