#Markdown CHATCIELO - Adaptive Core

Bem-vindo ao repositório oficial do **CHATCIELO**, um sistema de Machine Learning de missão crítica projetado para a Cielo Brasil. 

Este projeto implementa um modelo de **Pairwise Preference Ranking** (inspirado no SOTA do Kaggle LMSYS) para personalizar o atendimento a lojistas, adaptando respostas de LLMs com base no perfil do usuário (MEI, Varejo, Corporate) e garantindo uma experiência conversacional de alta conversão e baixo churn.

---

## 🏗️ 1. Árvore de Diretórios

O projeto segue padrões estritos de MLOps, separando responsabilidades entre engenharia de dados, treinamento, inferência e API.

```text
CHATCIELO/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Pydantic BaseSettings — todas as configs via .env
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # ⚠️ CORAÇÃO da injeção de contexto — pairwise dataset
│   │   └── preprocessing.py     # PII scrubbing (LGPD), normalização, tokenização
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py  # Features auxiliares (segmento, delta de tamanho, Jaccard)
│   │   └── embeddings.py           # Sentence-transformers para cold-start e similarity
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── preference_model.py  # DeBERTa-v3 cross-encoder + AuxFusionHead
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py             # CLI de treino (entry point)
│   │   ├── trainer.py           # Loop de treino: grad accum, early stopping, checkpoint
│   │   └── losses.py            # LabelSmoothing + MarginRanking + CombinedLoss
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── infer.py             # CLI de inferência (entry point)
│   │   └── predictor.py         # Stateless Predictor: Redis cache, P99 < 300ms
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py          # CLI de avaliação (entry point)
│   │   └── metrics.py           # AUC, Accuracy, Log-loss, NDCG, ECE
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app factory
│   │   ├── middleware.py        # Logging, Prometheus, PII audit
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── preference.py    # POST /preference, POST /preference/batch
│   │   │   ├── feedback.py      # POST /feedback  → Redis stream (feedback loop)
│   │   │   └── health.py        # GET /health
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── requests.py      # Pydantic request models
│   │       └── responses.py     # Pydantic response models
│   │
│   └── __init__.py
│
├── tests/
│   ├── unit/                    # pytest — um test_ por módulo em src/
│   ├── integration/             # Fluxo ponta a ponta: Dataloader→Model→Scoring
│   ├── ml/                      # hypothesis — property-based tests (score ∈ [0,1])
│   ├── security/                # LGPD: CPF/cartão em logs e outputs
│   └── benchmarks/              # pytest-benchmark — P99 < 300ms enforced
│
├── scripts/
│   ├── lgpd_audit.py            # Varre logs/outputs buscando PII → exit 1 se encontrar
│   └── validate_dataset.py      # Re-valida pipeline de dados após mudanças em dataset.py
│
├── notebooks/
│   └── 01_eda_lmsys.ipynb       # Exploração dos dados do Kaggle LMSYS
│
├── docker/
│   ├── Dockerfile               # Multi-stage: builder + runtime
│   └── docker-compose.yml       # API + Redis + Postgres + Prometheus + Grafana
│
├── intents/
│   └── intents.json             # Domínios: Jurídico/LGPD, Recomendação, Suporte Técnico
│
├── artifacts/
│   └── models/                  # Checkpoints salvos (gitignored)
│
├── data/                        # Dados de treino/val (gitignored)
│
├── .env.example
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
└── CLAUDE.md
🏛️ 2. Arquitetura TécnicaA arquitetura foi desenhada para altíssima disponibilidade (latência P99 < 300ms) e conformidade rigorosa com a LGPD.Plaintext╔══════════════════════════════════════════════════════════════════════════════╗
║                        CHATCIELO — SYSTEM ARCHITECTURE                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│  CAMADA 1 — CANAIS DE ENTRADA                                               │
│                                                                             │
│   Web Chat (Next.js)      WhatsApp (360dialog)      Mobile App (React Native)│
│   Latência: ~80ms         Latência: ~150ms           Latência: ~100ms       │
│         │                        │                          │               │
│         └────────────────────────┴──────────────────────────┘               │
│                                  │                                          │
│                          API Gateway (nginx)                                │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│  CAMADA 2 — SERVING LAYER                          Target: P99 < 300ms      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  FastAPI Application (uvicorn, 4 workers, stateless)                │    │
│  │                                                                     │    │
│  │  POST /preference  ──►  [1] Validate + PII scrub                    │    │
│  │                         [2] Redis cache lookup  ─────► HIT: return  │    │
│  │                         [3] MISS: Feature engineering               │    │
│  │                         [4] PreferenceModel.forward()               │    │
│  │                         [5] Cache write (TTL 1h)                    │    │
│  │                         [6] Return PreferenceResponse               │    │
│  │                                                                     │    │
│  │  POST /feedback    ──►  Redis XADD → stream "chatcielo:feedback"    │    │
│  │  GET  /health      ──►  Liveness + readiness probe                  │    │
│  │  GET  /metrics     ──►  Prometheus exposition                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Redis (cache + feedback stream)    Prometheus + Grafana (observability)    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│  CAMADA 3 — MODEL LAYER                                                     │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │  PreferenceModel  (DeBERTa-v3-large cross-encoder)               │       │
│  │                                                                  │       │
│  │  Input:  [SEG] <prompt>p</prompt>                                │       │
│  │          <response_a>r_a</response_a>                            │       │
│  │          <response_b>r_b</response_b>                            │       │
│  │                                                                  │       │
│  │  Encoder → pooler_output (d=1024)                                │       │
│  │  AuxFusionHead: concat([CLS_emb, aux_features(12)]) → Linear(3)  │       │
│  │  Output: softmax([P(A wins), P(B wins), P(tie)])                 │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  Model Registry: artifacts/models/{best.pt, latest.pt, v{n}.pt}             │
│  Versionamento: hash do dataset + hyperparams no nome do artefato           │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│  CAMADA 4 — TRAINING PIPELINE                                               │
│                                                                             │
│  Dados Brutos (Parquet/CSV)                                                 │
│       │                                                                     │
│       ▼                                                                     │
│  scripts/validate_dataset.py  ── se falhar → bloqueia pipeline              │
│       │                                                                     │
│       ▼                                                                     │
│  src/data/dataset.py          ── PII scrub → tokenize → ChatCieloDataset    │
│       │                                                                     │
│       ▼                                                                     │
│  src/features/feature_engineering.py  ── aux features (12-dim)              │
│       │                                                                     │
│       ▼                                                                     │
│  src/training/trainer.py      ── AdamW + Cosine LR + grad accum             │
│       │                         CombinedLoss (0.7·cls + 0.3·rank)           │
│       ▼                                                                     │
│  src/evaluation/evaluate.py   ── AUC ≥ 0.70 + ECE < 0.05 → salva best.pt    │
│       │                                                                     │
│       ▼                                                                     │
│  artifacts/models/best.pt  ──► serving layer                                │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│  CAMADA 5 — FEEDBACK LOOP & MONITORAMENTO                                   │
│                                                                             │
│  Redis Stream "chatcielo:feedback"                                          │
│       │                                                                     │
│       ▼  (consumer worker — roda como cron/k8s job)                         │
│  scripts/retrain_trigger.py                                                 │
│       │  detecta drift (PSI > 0.2 ou AUC degradação > 5%)                   │
│       ▼                                                                     │
│  Dispara novo ciclo de training pipeline (acima)                            │
│                                                                             │
│  A/B Testing: feature flag por merchant_segment no Redis                    │
│  Drift detection: PSI no distribution de scores a cada 24h                  │
│  Métricas: Prometheus → Grafana (accuracy, P99 latency, PII leak count=0)   │
└─────────────────────────────────────────────────────────────────────────────┘
🛠️ 3. Stack de TecnologiasCamadaTecnologiaJustificativaAPIFastAPI + uvicornASGI assíncrono, ideal para manter a latência < 300ms.Cache & FilasRedis 7Sub-ms lookup, TTL nativo e suporte a streams para o feedback loop.Banco de DadosPostgreSQL 15Retenção do histórico de conversas e audit log (LGPD) em schema tipado.ML RuntimePyTorch 2.2 + TransformersSuporte nativo ao DeBERTa-v3 com aceleração via CUDA/MPS/CPU.ContainerizaçãoDocker + ComposeBuild isolado multi-stage, garantindo imagens leves (slim) para produção.ObservabilityPrometheus + GrafanaRastreio de latência P99 e contadores críticos de segurança (PII leak = 0).StreamingRedis StreamsIngestão leve de feedback. Pode ser escalado para Kafka se > 10k req/s.
