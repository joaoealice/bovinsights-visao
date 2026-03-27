import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .core.config import get_settings
from .routers import inference, health, training

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
app.include_router(training.router)

# Servir o frontend (PWA) como arquivos estáticos
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
