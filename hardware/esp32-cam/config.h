#pragma once

// ── WiFi ─────────────────────────────────────────────────────────────────────
#define WIFI_SSID       "SUA_REDE_WIFI"
#define WIFI_PASSWORD   "SUA_SENHA_WIFI"

// ── Servidor Bovinsights ──────────────────────────────────────────────────────
// Railway (produção) — substitua pela URL do seu deploy:
#define SERVER_URL      "https://SEU-APP.railway.app/api/v1/stream/frame"
// Orange Pi 5 na rede local (alternativa — descomente e ajuste o IP):
// #define SERVER_URL   "http://192.168.1.100:8000/api/v1/stream/frame"

// Identificação desta câmera (aparece nos logs e alertas da plataforma)
#define SOURCE_ID       "esp32cam-01"
#define LOCATION        "piquete-a"

// ── Captura ───────────────────────────────────────────────────────────────────
#define CAPTURE_INTERVAL_MS   5000   // intervalo entre frames em ms (padrão: 5s)
#define MAX_RETRIES           3      // tentativas por frame antes de descartar
#define HTTP_TIMEOUT_MS       10000  // timeout por requisição HTTP

// ── Resolução da câmera ───────────────────────────────────────────────────────
// FRAMESIZE_QVGA = 320x240  — mais rápido, payload menor (~15KB)
// FRAMESIZE_VGA  = 640x480  — recomendado (equilíbrio qualidade/velocidade)
// FRAMESIZE_SVGA = 800x600  — mais detalhes, payload maior (~60KB)
#define CAM_FRAMESIZE   FRAMESIZE_VGA
#define CAM_QUALITY     12  // JPEG quality: 0 (pior) a 63 (melhor). 10-15 = bom
