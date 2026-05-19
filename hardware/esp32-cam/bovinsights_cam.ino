/**
 * Bovinsights — Firmware ESP32-CAM (AI-Thinker)
 *
 * Função: captura frame JPEG e envia para o servidor Bovinsights via POST.
 * O servidor faz toda a inferência YOLO — o ESP32 só captura e transmite.
 *
 * Fluxo:
 *   Liga → conecta WiFi → loop: captura → POST → aguarda intervalo
 *
 * Requisitos Arduino IDE:
 *   - Board: "AI Thinker ESP32-CAM" (ESP32 by Espressif, versão >= 2.0)
 *   - Biblioteca: nenhuma externa necessária (usa HTTPClient e WiFi nativas)
 *
 * Para gravar:
 *   1. Conecte IO0 ao GND antes de ligar (modo flash)
 *   2. Grave via USB-Serial (baud 115200)
 *   3. Desconecte IO0 do GND e reinicie
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "config.h"

// ── Pinout AI-Thinker ESP32-CAM ──────────────────────────────────────────────
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  Serial.println("\n[Bovinsights] ESP32-CAM iniciando...");

  initCamera();
  connectWiFi();

  Serial.println("[Bovinsights] Pronto. Capturando a cada "
                 + String(CAPTURE_INTERVAL_MS / 1000) + "s.");
}

// ── Loop principal ────────────────────────────────────────────────────────────

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WARN] WiFi desconectado. Reconectando...");
    connectWiFi();
    return;
  }

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[ERRO] Falha ao capturar frame. Aguardando 1s...");
    delay(1000);
    return;
  }

  Serial.printf("[CAM] Frame: %d bytes\n", fb->len);

  bool enviado = false;
  for (int t = 1; t <= MAX_RETRIES && !enviado; t++) {
    enviado = enviarFrame(fb->buf, fb->len);
    if (!enviado && t < MAX_RETRIES) {
      Serial.printf("[WARN] Tentativa %d/%d falhou. Aguardando 2s...\n", t, MAX_RETRIES);
      delay(2000);
    }
  }

  if (!enviado) {
    Serial.println("[ERRO] Frame descartado após " + String(MAX_RETRIES) + " tentativas.");
  }

  esp_camera_fb_return(fb);
  delay(CAPTURE_INTERVAL_MS);
}

// ── Envio do frame ────────────────────────────────────────────────────────────

bool enviarFrame(uint8_t* buf, size_t len) {
  HTTPClient http;
  http.begin(SERVER_URL);
  http.setTimeout(HTTP_TIMEOUT_MS);

  const String boundary = "BovinsightsBoundary7MA4YWxkTrZu0gW";
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

  // Parte 1: cabeçalho do arquivo JPEG
  String head = "--" + boundary + "\r\n"
                "Content-Disposition: form-data; name=\"file\"; filename=\"frame.jpg\"\r\n"
                "Content-Type: image/jpeg\r\n\r\n";

  // Parte 2: campos de texto (source_id e location)
  String tail = "\r\n"
                "--" + boundary + "\r\n"
                "Content-Disposition: form-data; name=\"source_id\"\r\n\r\n"
                + String(SOURCE_ID) + "\r\n"
                "--" + boundary + "\r\n"
                "Content-Disposition: form-data; name=\"location\"\r\n\r\n"
                + String(LOCATION) + "\r\n"
                "--" + boundary + "--\r\n";

  size_t totalLen = head.length() + len + tail.length();
  uint8_t* body = (uint8_t*)malloc(totalLen);

  if (!body) {
    Serial.println("[ERRO] Sem RAM para montar body HTTP.");
    http.end();
    return false;
  }

  memcpy(body,                        head.c_str(), head.length());
  memcpy(body + head.length(),        buf,          len);
  memcpy(body + head.length() + len,  tail.c_str(), tail.length());

  int httpCode = http.POST(body, totalLen);
  free(body);
  http.end();

  if (httpCode == 200 || httpCode == 201) {
    Serial.printf("[OK] Enviado. HTTP %d\n", httpCode);
    return true;
  }

  Serial.printf("[WARN] Servidor retornou HTTP %d\n", httpCode);
  return false;
}

// ── Inicialização da câmera ───────────────────────────────────────────────────

void initCamera() {
  camera_config_t cfg;
  cfg.ledc_channel  = LEDC_CHANNEL_0;
  cfg.ledc_timer    = LEDC_TIMER_0;
  cfg.pin_d0        = Y2_GPIO_NUM;
  cfg.pin_d1        = Y3_GPIO_NUM;
  cfg.pin_d2        = Y4_GPIO_NUM;
  cfg.pin_d3        = Y5_GPIO_NUM;
  cfg.pin_d4        = Y6_GPIO_NUM;
  cfg.pin_d5        = Y7_GPIO_NUM;
  cfg.pin_d6        = Y8_GPIO_NUM;
  cfg.pin_d7        = Y9_GPIO_NUM;
  cfg.pin_xclk      = XCLK_GPIO_NUM;
  cfg.pin_pclk      = PCLK_GPIO_NUM;
  cfg.pin_vsync     = VSYNC_GPIO_NUM;
  cfg.pin_href      = HREF_GPIO_NUM;
  cfg.pin_sscb_sda  = SIOD_GPIO_NUM;
  cfg.pin_sscb_scl  = SIOC_GPIO_NUM;
  cfg.pin_pwdn      = PWDN_GPIO_NUM;
  cfg.pin_reset     = RESET_GPIO_NUM;
  cfg.xclk_freq_hz  = 20000000;
  cfg.pixel_format  = PIXFORMAT_JPEG;
  cfg.frame_size    = CAM_FRAMESIZE;
  cfg.jpeg_quality  = CAM_QUALITY;
  cfg.fb_count      = 1;

  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    Serial.printf("[ERRO] Camera init falhou: 0x%x — verifique a placa.\n", err);
    while (true) delay(1000);
  }
  Serial.println("[CAM] Camera OK.");
}

// ── Conexão WiFi ──────────────────────────────────────────────────────────────

void connectWiFi() {
  Serial.printf("[WiFi] Conectando a '%s'", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int tentativas = 0;
  while (WiFi.status() != WL_CONNECTED && tentativas < 20) {
    delay(500);
    Serial.print(".");
    tentativas++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] Conectado. IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[ERRO] WiFi falhou. Reiniciando em 5s...");
    delay(5000);
    ESP.restart();
  }
}
