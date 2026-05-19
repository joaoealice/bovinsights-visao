# Bovinsights — Hardware

Firmware e configurações para os dispositivos de captura do sistema Bovinsights.

---

## Estrutura

```
hardware/
  esp32-cam/
    bovinsights_cam.ino   ← sketch principal (Arduino IDE)
    config.h              ← WiFi, URL do servidor, intervalo, resolução
  README.md               ← este arquivo
```

---

## ESP32-CAM (AI-Thinker) — Modo B

O ESP32-CAM **não processa** as imagens. Ele apenas captura frames JPEG e envia
via POST para o servidor (Railway ou Orange Pi 5 na rede local). Todo o YOLO
roda no servidor.

### Pré-requisitos

- Arduino IDE >= 2.0
- Board: **AI Thinker ESP32-CAM** (instale via `Gerenciador de Placas` → `ESP32 by Espressif`)
- Nenhuma biblioteca externa — usa `WiFi.h` e `HTTPClient.h` nativas

### Configuração antes de gravar

Edite `config.h`:

```cpp
#define WIFI_SSID     "sua_rede"
#define WIFI_PASSWORD "sua_senha"
#define SERVER_URL    "https://SEU-APP.railway.app/api/v1/stream/frame"
#define SOURCE_ID     "esp32cam-01"     // identifica esta câmera nos logs
#define LOCATION      "piquete-a"       // aparece nos alertas da plataforma
```

### Como gravar

1. Conecte **IO0 ao GND** (coloque um jumper ou pressione o botão de flash)
2. Ligue o ESP32-CAM via USB-Serial (adaptador FTDI ou similar)
3. No Arduino IDE: selecione a porta COM correta e grave
4. Remova o jumper IO0-GND e pressione Reset
5. Abra o Monitor Serial (115200 baud) para acompanhar os logs

### O que aparece no Monitor Serial

```
[Bovinsights] ESP32-CAM iniciando...
[CAM] Camera OK.
[WiFi] Conectando a 'MinhaRede'....
[WiFi] Conectado. IP: 192.168.1.105
[Bovinsights] Pronto. Capturando a cada 5s.
[CAM] Frame: 28432 bytes
[OK] Enviado. HTTP 200
[CAM] Frame: 27891 bytes
[OK] Enviado. HTTP 200
```

### Parâmetros ajustáveis em `config.h`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `CAPTURE_INTERVAL_MS` | 5000 | Intervalo entre frames (ms) |
| `MAX_RETRIES` | 3 | Tentativas por frame antes de descartar |
| `HTTP_TIMEOUT_MS` | 10000 | Timeout da requisição HTTP |
| `CAM_FRAMESIZE` | `FRAMESIZE_VGA` | Resolução (QVGA=320x240, VGA=640x480) |
| `CAM_QUALITY` | 12 | Qualidade JPEG (menor = mais leve; 10-15 recomendado) |

---

## Endpoint do servidor

O firmware faz POST para `/api/v1/stream/frame` com `multipart/form-data`:

| Campo | Tipo | Descrição |
|---|---|---|
| `file` | JPEG | Frame capturado pela câmera |
| `source_id` | string | ID desta câmera (ex: `esp32cam-01`) |
| `location` | string | Localização (ex: `piquete-a`) |

O servidor retorna o JSON de detecção, mas o ESP32 ignora a resposta (fire-and-forget).
