import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from .routers import inference, health, training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bovisights CV Server",
    description="API de detecção de comportamento bovino por visão computacional",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    try:
        from .core.config import get_settings
        settings = get_settings()
        logger.info(f"[OK] Configurações carregadas — env={settings.api_env}, model={settings.roboflow_model_id}/{settings.roboflow_model_version}")
    except Exception as e:
        logger.error(f"[ERRO] Falha ao carregar configurações: {e}")


# CORS — permite que o celular do cliente chame a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rotas da API (registrar ANTES do StaticFiles)
app.include_router(health.router)
app.include_router(inference.router)
app.include_router(training.router)

# Servir o frontend (PWA) como arquivos estáticos
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    logger.info(f"[OK] Frontend encontrado em: {frontend_path}")
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    logger.warning(f"[AVISO] Frontend não encontrado em: {frontend_path}")
