# Pendências — Bovisights CV

## Em aberto

### 1. Melhorar detecção de comportamentos
- Modelo atual `cattle-dataset-behavior-cqtzu/1` não detecta todos os animais em campo
- **Próximos passos:**
  - [ ] Capturar frames reais do confinamento com o botão "Enviar pra Treino"
  - [ ] Acumular 30+ frames por comportamento (lying, eating, standing)
  - [ ] Anotar no Roboflow → app.roboflow.com/grupo-j-f-business/bovinos-confinamento
  - [ ] Treinar nova versão do modelo com imagens dos próprios animais
  - [ ] Atualizar `ROBOFLOW_MODEL_ID` e `ROBOFLOW_MODEL_VERSION` no Railway

### 2. Tracking de animais únicos
- Hoje o sistema conta detecções por frame, não animais únicos
- Relatório usa pico/média por frame como estimativa (honesto, mas não preciso)
- **Solução futura:** implementar ByteTrack/BoT-SORT via Ultralytics para rastrear IDs
- Requer câmera fixa no confinamento (não funciona bem com câmera em movimento)

### 3. Temperatura dos animais
- Não é possível via câmera comum
- **Opções futuras:**
  - Câmera térmica (FLIR) — R$ 3.000~30.000
  - Sensores IoT no animal (brinco/bolus) — R$ 300~800/animal
  - Integrar dados do sensor com o comportamento visual para alertas de doença

## Concluído em 2026-03-27
- [x] Troca para modelo `cattle-dataset-behavior-cqtzu/1` (mAP 81%, detecta lying down)
- [x] Modelo de contagem `cow-count/2` (mAP 98%) rodando em paralelo
- [x] LABEL_MAP atualizado com todos os comportamentos
- [x] Script `scripts/extrair_frames.py` para extrair frames de vídeo
- [x] Interface com abas: Ao vivo / Relatório da sessão
- [x] Relatório com pico, média por frame e % de comportamentos
- [x] Export do relatório em CSV e TXT com data e hora
- [x] Botão "Baixar Frame" e "Enviar pra Treino" com upload para Roboflow
- [x] Link direto para anotar no Roboflow após upload
- [x] Deploy no Railway — servidor 24h no ar
- [x] Domínio customizado: visao.bovinsights.com.br
- [x] Repositório GitHub: github.com/joaoealice/bovinsights-visao

## Concluído em 2026-03-26
- [x] Estrutura completa do projeto criada
- [x] Backend FastAPI com integração Roboflow
- [x] Fallback YOLOv11 local
- [x] Frontend PWA funcionando no browser
- [x] Servidor rodando localmente em `localhost:8000`
- [x] Câmera abrindo e retornando detecções (eating, standing)
- [x] Threshold de confiança ajustado para 25%
