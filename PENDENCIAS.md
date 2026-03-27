# Pendências — Bovisights CV

## Em aberto

### 1. Detecção de bovinos deitados
- ~~O modelo `cow-behavior-tracking/1` não está identificando animais deitados~~
- **[2026-03-27] Modelo trocado para `cattle-dataset-behavior-cqtzu/1`**
  - mAP 81.16% | Recall 74.40% | Precision 84.10%
  - Classes: eating, foraging, lying down, standing, drinking water, rumination, falling, sitting
  - 3811 imagens de treinamento — muito mais robusto
  - Testar em campo com vídeos reais do confinamento
- [ ] Treinar modelo próprio com imagens dos **seus** animais
  - Usar `python scripts/extrair_frames.py --video seu_video.mp4 --upload`
  - Anotar frames no Roboflow, marcar "lying" nos deitados
  - Treinar nova versão e atualizar `.env`

### 2. Deploy no Railway.app
- MVP local funcionando, falta publicar para acesso via celular em campo
- Ver Seção 10 do AGENT.md para instruções completas
- Variáveis de ambiente a configurar no Railway:
  - `ROBOFLOW_API_KEY`
  - `ROBOFLOW_MODEL_ID`
  - `ROBOFLOW_MODEL_VERSION`
  - `API_ENV=production`
  - `ALLOWED_ORIGINS=https://SEU_DOMINIO.up.railway.app`

## Concluído em 2026-03-27
- [x] Pesquisa de modelos no Roboflow Universe com classe "lying"
- [x] Troca para modelo `cattle-dataset-behavior-cqtzu/1` (mAP 81%, tem lying down)
- [x] LABEL_MAP atualizado com "drinking water", "falling", "sitting"
- [x] Script `scripts/extrair_frames.py` criado para extrair frames de vídeo

## Concluído em 2026-03-26
- [x] Estrutura completa do projeto criada
- [x] Backend FastAPI com integração Roboflow
- [x] Fallback YOLOv11 local
- [x] Frontend PWA funcionando no browser
- [x] Servidor rodando localmente em `localhost:8000`
- [x] Câmera abrindo e retornando detecções (eating, standing)
- [x] Threshold de confiança ajustado para 25%
