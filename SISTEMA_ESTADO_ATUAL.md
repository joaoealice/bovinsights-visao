# Bovinsights — Estado Atual do Sistema

**Data:** 2026-05-20  
**Branch:** main  
**Último commit:** `8abe699` — fix: substitui Gemini Vision por Groq Vision

---

## 1. Arquitetura Atual

### Backend (`backend/`)

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Ponto de entrada FastAPI — registra routers, CORS, StaticFiles e inicializa o banco |
| `database.py` | Engine SQLite (`bovinsights.db`), `SessionLocal`, `Base`, `init_db()` e `get_db()` |
| `core/config.py` | `Settings` via `pydantic-settings` — lê variáveis de ambiente com `@lru_cache` |
| `routers/inference.py` | Endpoints `/api/v1/detect` e `/api/v1/detect/base64` — orquestra a cadeia de 3 fallbacks |
| `routers/health.py` | Endpoint `GET /health` — retorna `{"status": "ok"}` |
| `routers/stream.py` | Endpoint `POST /api/v1/stream/frame` — recebe frames do ESP32-CAM com `source_id` e salva sessão |
| `routers/training.py` | Endpoint `POST /api/v1/training/upload` — envia frame para Roboflow como dado de treino |
| `routers/alertas.py` | Endpoint `GET /api/v1/alertas` — lista alertas do banco com filtros |
| `routers/sessao.py` | Endpoint `POST /api/v1/sessao/iniciar` — registra ou atualiza uma sessão de câmera |
| `services/roboflow_client.py` | Chama a API Roboflow (nível 1 da cadeia), mapeia classes v2 para `BehaviorCount` |
| `services/yolo_local.py` | Inferência local com `bovinsights_yolo11n_best.pt` usando Ultralytics (nível 2) |
| `services/gemini_fallback.py` | Fallback Groq Vision com `llama-4-scout-17b-16e-instruct` via base64 (nível 3) |
| `services/image_processor.py` | Valida imagem, redimensiona para 1280×1280 e aplica CLAHE via OpenCV |
| `models/alerta.py` | ORM SQLAlchemy para tabela `alertas` |
| `models/sessao.py` | ORM SQLAlchemy para tabela `sessoes` |
| `models/bovinsights_yolo11n_best.pt` | Pesos do modelo YOLO treinado — mAP50 0.861 |
| `schemas/detection.py` | `DetectionResponse`, `BehaviorCount`, `DetectionBox` |
| `schemas/alerta.py` | `AlertaOut` (Pydantic v2) |
| `schemas/sessao.py` | `SessaoIniciarRequest`, `SessaoOut` (Pydantic v2) |

### Frontend (`frontend/`)

PWA estático servido pelo próprio FastAPI via `StaticFiles`. Contém `index.html`, `manifest.json`, `sw.js` e `assets/`.

### Hardware (`hardware/esp32-cam/`)

Firmware Arduino para ESP32-CAM que captura frames e os envia para `POST /api/v1/stream/frame`.

---

## 2. Cadeia de Inferência

```
POST /api/v1/detect  (ou /detect/base64)
         │
         ▼
  image_processor.py
  ├── Valida formato (JPEG/PNG/WEBP, máx 5MB)
  ├── Converte para RGB
  ├── Redimensiona para ≤ 1280×1280
  └── Aplica CLAHE (equalização de histograma adaptativa)
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │            NÍVEL 1 — Roboflow API                   │
  │  modelo: cattle-dataset-behavior-cqtzu/v1           │
  │  confidence: 35%  |  overlap: 30%  |  timeout: 15s  │
  │  retorna bounding boxes + labels v2                  │
  └─────────────────────────────────────────────────────┘
         │  falha OU total_animals == 0
         ▼
  ┌─────────────────────────────────────────────────────┐
  │            NÍVEL 2 — YOLO local                     │
  │  modelo: bovinsights_yolo11n_best.pt                │
  │  confidence: 40%  |  executa no servidor Railway    │
  │  classes: Comendo, Deitado, Em pe, Escondido,       │
  │           Pastando                                   │
  └─────────────────────────────────────────────────────┘
         │  falha OU total_animals == 0
         ▼
  ┌─────────────────────────────────────────────────────┐
  │            NÍVEL 3 — Groq Vision                    │
  │  modelo: meta-llama/llama-4-scout-17b-16e-instruct  │
  │  imagem enviada em base64 (data:image/jpeg;base64)  │
  │  retorna JSON estruturado — sem bounding boxes      │
  │  model_used: "groq-vision-fallback"                 │
  └─────────────────────────────────────────────────────┘
         │  todos falharam
         ▼
      HTTP 503

ATENÇÃO: O router stream.py (usado pelo ESP32-CAM) usa apenas
2 níveis (Roboflow → YOLO local). Não chama o Groq.
```

---

## 3. O que Está Funcionando

- **Deploy no Railway** — servidor FastAPI sobe e responde `/health`
- **Frontend PWA** — servido como estático pelo próprio backend
- **CLAHE** — equalização de histograma aplicada automaticamente quando `opencv-python-headless` está disponível
- **Roboflow nível 1** — chamada funcionando com confidence 35%
- **YOLO local nível 2** — modelo `bovinsights_yolo11n_best.pt` carregado no servidor
- **Groq Vision nível 3** — substituiu Gemini com sucesso (commit `8abe699`); usa `llama-4-scout-17b-16e-instruct`
- **Banco SQLite** — tabelas `alertas` e `sessoes` criadas no startup
- **CORS aberto** — `allow_origins=["*"]` permite chamadas do celular/câmera
- **Fallback por `total_animals == 0`** — avança para o próximo nível quando o modelo retorna vazio

---

## 4. O que Está Quebrado

### BUG — `yolo_local.py`: mapeamento de label errado

`LABEL_MAP` mapeia `"deitado"` → `"lying"`, mas `BehaviorCount` não tem campo `lying` — tem `deitado`. Resultado: animais deitados detectados pelo YOLO local caem em `unknown`.

```python
# yolo_local.py linha 22 — ERRADO
"deitado": "lying",

# Correto
"deitado": "deitado",
```

### BUG — `stream.py`: fallback incompleto (apenas 2 níveis)

O router do ESP32-CAM (`POST /api/v1/stream/frame`) chama `run_inference` → `run_inference_local` mas nunca chama o Groq Vision. Se os dois falharem, o endpoint retorna erro em vez de usar o nível 3.

### PROBLEMA — `training.py`: prints de debug em produção

Linhas 63–66 têm `print("[DEBUG]...")` que aparecem nos logs do Railway sem utilidade em produção.

### INCONSISTÊNCIA — `roboflow_client.py`: docstring desatualizada

O docstring diz "Confidence: 45%" mas o código usa 35% (linha 79). Não afeta comportamento, mas gera confusão.

### LIMITAÇÃO — campos extras do Groq ignorados pelo schema

`gemini_fallback.py` retorna `nivel_alerta`, `resumo`, `alertas` e `confianca_geral`, mas `DetectionResponse` não tem esses campos. O Pydantic os descarta silenciosamente quando `inference.py` faz `DetectionResponse(**dados)`. O frontend nunca recebe essas informações.

### LIMITAÇÃO — `yolo_local.py`: caminho relativo frágil

`MODEL_PATH = "backend/models/bovinsights_yolo11n_best.pt"` depende do diretório de trabalho onde o servidor é iniciado. Se o Railway mudar o CWD, o modelo não carrega.

---

## 5. Variáveis de Ambiente Necessárias

| Variável | Obrigatória | Padrão | Status |
|---|---|---|---|
| `ROBOFLOW_API_KEY` | Sim | — | Configurada no Railway |
| `GROQ_API_KEY` | Sim (fallback nível 3) | — | Configurada no Railway |
| `ROBOFLOW_MODEL_ID` | Não | `cattle-dataset-behavior-cqtzu` | Usa padrão |
| `ROBOFLOW_MODEL_VERSION` | Não | `1` | Usa padrão |
| `ROBOFLOW_WORKSPACE` | Para training upload | — | Verificar |
| `ROBOFLOW_PROJECT` | Para training upload | — | Verificar |
| `API_ENV` | Não | `development` | Mudar para `production` no Railway |
| `API_SECRET_KEY` | Não | `change_this_in_production` | **Trocar urgente** |
| `MAX_IMAGE_SIZE_MB` | Não | `5` | Usa padrão |
| `INFERENCE_INTERVAL_SECONDS` | Não | `3` | Usa padrão |
| `HISTOGRAM_EQUALIZATION` | Não | `True` | Usa padrão |
| `USE_LOCAL_MODEL` | Não | `False` | Usa padrão |

---

## 6. Princípios para o Agente

Regras que devem ser seguidas em **todo commit** feito neste projeto:

1. **Sempre confirmar push concluído** — nunca declarar "feito" sem a saída `git push` confirmando o remote atualizado.

2. **Nunca declarar que um arquivo foi salvo sem verificar** — após `Write` ou `Edit`, confirmar que o conteúdo está correto antes de reportar sucesso.

3. **Sempre testar caminho de arquivos antes de referenciar no código** — usar `os.path.exists()` ou caminho absoluto via `__file__` em vez de caminhos relativos que dependem do CWD.

4. **Sempre incluir `!pip install` no início de células Colab** — notebooks devem ser autocontidos e executáveis em qualquer ambiente sem setup manual.

5. **git add explícito por arquivo** — nunca usar `git add .` ou `git add -A` para evitar commitar arquivos não intencionais (`.env`, `*.db`, pesos de modelo grandes).

6. **Nunca commitar chaves de API** — variáveis como `GROQ_API_KEY` e `ROBOFLOW_API_KEY` pertencem exclusivamente às variáveis de ambiente do Railway, jamais ao código ou a arquivos no repositório.

---

## 7. Próximas Ações Prioritárias

Ordenadas por impacto:

1. **[CRÍTICO] Corrigir mapeamento `yolo_local.py`** — `"deitado": "lying"` → `"deitado": "deitado"`. Animais deitados estão sendo perdidos no nível 2.

2. **[ALTO] Adicionar Groq ao fallback do `stream.py`** — o router do ESP32-CAM usa apenas 2 níveis; unificar com a mesma cadeia de `inference.py` para consistência.

3. **[ALTO] Trocar `API_SECRET_KEY`** — o padrão `"change_this_in_production"` está exposto; gerar uma chave aleatória e configurar no Railway.

4. **[MÉDIO] Adicionar `nivel_alerta` e `resumo` ao `DetectionResponse`** — os campos retornados pelo Groq são descartados pelo Pydantic; expô-los no schema para o frontend poder exibir alertas contextuais.

5. **[MÉDIO] Corrigir caminho do modelo YOLO** — substituir o caminho relativo por `Path(__file__).parent.parent / "models" / "bovinsights_yolo11n_best.pt"` para ser robusto ao CWD.

6. **[BAIXO] Remover prints de debug do `training.py`** — substituir por `logger.debug(...)`.

7. **[BAIXO] Atualizar docstring do `roboflow_client.py`** — alinhar a documentação interna com a confidence real (35%).
