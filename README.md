# CHATCIELO - Core Adaptativo de Preferência

## 1. Visão Geral do Projeto

O CHATCIELO é um sistema de Machine Learning de missão crítica, desenvolvido para a Cielo Brasil, que emprega um modelo avançado de **Pairwise Preference Ranking**. Inspirado nas abordagens de ponta da competição Kaggle LMSYS, este sistema visa personalizar e otimizar as interações com lojistas, adaptando as respostas de Large Language Models (LLMs) com base no perfil específico do usuário (MEI, Varejo, Corporate). O objetivo primordial é proporcionar uma experiência conversacional de alta conversão, minimizando o *churn* e maximizando a satisfação do cliente.

Este projeto se destaca pela sua aderência a rigorosos padrões de MLOps, garantindo a separação clara de responsabilidades entre engenharia de dados, treinamento de modelos, inferência em produção e a exposição via API. A arquitetura foi concebida para assegurar altíssima disponibilidade, com uma latência P99 inferior a 300 milissegundos, e conformidade estrita com a Lei Geral de Proteção de Dados (LGPD).

## 2. Arquitetura do Sistema

A arquitetura do CHATCIELO é organizada em camadas distintas, projetadas para otimizar o desempenho, a escalabilidade e a manutenibilidade do sistema.

### 2.1. Camada de Entrada

Esta camada compreende os diversos canais pelos quais os usuários interagem com o sistema, incluindo Web Chat (desenvolvido com Next.js), WhatsApp (via 360dialog) e aplicativos móveis (React Native). Todas as requisições são roteadas através de um API Gateway (nginx), que atua como ponto de entrada unificado, garantindo segurança e balanceamento de carga.

### 2.2. Camada de Serviço (Serving Layer)

Implementada com **FastAPI** e **Uvicorn**, esta camada é responsável por orquestrar a inferência do modelo em produção. Ela é projetada para ser *stateless* e de alta performance, visando a meta de latência P99 < 300ms. As principais funcionalidades incluem:

*   **Endpoints de Preferência**: `POST /preference` e `POST /preference/batch` para obter as predições do modelo.
*   **Cache de Resultados**: Utiliza **Redis** para armazenar resultados de inferência, reduzindo a latência para requisições repetidas. O cache possui um Time-To-Live (TTL) configurável de 1 hora.
*   **Scrubbing de PII**: Realiza a remoção de informações de identificação pessoal (PII) dos *prompts* e respostas antes do processamento pelo modelo, garantindo a conformidade com a LGPD.
*   **Endpoints de Monitoramento**: `GET /health` para verificações de liveness e readiness, e `GET /metrics` para exposição de métricas Prometheus.
*   **Feedback Loop**: `POST /feedback` para ingestão de feedback do usuário em um stream Redis, alimentando o ciclo de retreinamento contínuo.

### 2.3. Camada de Modelo (Model Layer)

O cerne da inteligência do CHATCIELO reside no `PreferenceModel`, um **cross-encoder DeBERTa-v3-large** que incorpora uma `AuxFusionHead`. Este cabeçote de fusão tardia combina o *embedding* CLS do encoder com um vetor de 12 dimensões de *features* auxiliares, que incluem características estruturais das respostas (comprimento, média de palavras por frase, Type-Token Ratio), métricas de comparação entre respostas (delta de comprimento, similaridade Jaccard, delta de sentenças) e um *one-hot encoding* do segmento do lojista (MEI, Varejo, Corporate). Esta abordagem híbrida permite ao modelo considerar tanto o contexto textual quanto informações estruturadas do usuário para uma tomada de decisão mais precisa. Os modelos treinados são versionados e armazenados em um Model Registry (`artifacts/models/`).

### 2.4. Pipeline de Treinamento (Training Pipeline)

Este pipeline automatiza todo o processo de desenvolvimento e atualização do modelo, desde a ingestão de dados brutos até a disponibilização de novos *checkpoints*:

1.  **Validação de Dados**: O script `scripts/validate_dataset.py` garante a integridade e consistência dos dados de entrada, bloqueando o pipeline em caso de falhas.
2.  **Preparação de Dados**: `src/data/dataset.py` realiza o scrubbing de PII, tokenização e a criação do `ChatCieloDataset`, que organiza os dados em triplas (prompt, response_a, response_b) com condicionamento de segmento.
3.  **Engenharia de Features**: `src/features/feature_engineering.py` calcula as 12 *features* auxiliares que são injetadas no modelo.
4.  **Treinamento do Modelo**: `src/training/trainer.py` orquestra o loop de treinamento, utilizando otimizadores como AdamW, *learning rate schedules* (CosineAnnealingWarmup), acumulação de gradientes, *early stopping* baseado em AUC de validação e salvamento de *checkpoints* (`best.pt`, `latest.pt`). A função de perda combinada (`CombinedPreferenceLoss`) otimiza tanto a classificação quanto o ranking.
5.  **Avaliação**: `src/evaluation/evaluate.py` calcula métricas como AUC, Accuracy, Log-loss e ECE, com critérios de aceitação para o salvamento do melhor modelo.

### 2.5. Ciclo de Feedback e Monitoramento

Para garantir a adaptabilidade e a robustez do modelo em produção, o CHATCIELO incorpora um ciclo de feedback contínuo e um sistema de monitoramento abrangente:

*   **Feedback Loop**: O feedback dos usuários é ingerido via um stream **Redis** (`chatcielo:feedback`). Um *consumer worker* (`scripts/retrain_trigger.py`) monitora este stream para detectar *drift* na distribuição dos dados (e.g., PSI > 0.2) ou degradação da AUC, disparando automaticamente um novo ciclo do pipeline de treinamento.
*   **A/B Testing**: Suporta testes A/B com *feature flags* por segmento de lojista, permitindo a avaliação de modelos desafiantes em produção.
*   **Monitoramento**: **Prometheus** e **Grafana** são utilizados para rastrear métricas críticas, incluindo latência P99, acurácia do modelo e contagem de vazamentos de PII (com meta de zero).

## 3. Stack de Tecnologias

A tabela a seguir detalha as principais tecnologias empregadas no desenvolvimento do CHATCIELO e suas respectivas justificativas:

| Camada / Componente | Tecnologia | Justificativa |
| :------------------ | :--------- | :------------ |
| API                 | FastAPI + Uvicorn | Framework ASGI assíncrono, ideal para manter a latência de inferência abaixo de 300ms. |
| Cache & Filas       | Redis 7    | Oferece *lookup* em sub-milissegundos, TTL nativo para cache e suporte a *streams* para o *feedback loop*. |
| Banco de Dados      | PostgreSQL 15 | Utilizado para retenção do histórico de conversas e *audit logs* (LGPD) em um esquema tipado e robusto. |
| ML Runtime          | PyTorch 2.2 + Transformers | Suporte nativo ao DeBERTa-v3 com aceleração via CUDA/MPS/CPU, otimizado para modelos de *deep learning*. |
| Containerização     | Docker + Docker Compose | Permite *builds* isolados e multi-stage, garantindo imagens leves e portáveis para produção. |
| Observabilidade     | Prometheus + Grafana | Ferramentas padrão da indústria para rastreamento de latência P99, métricas de modelo e contadores críticos de segurança (e.g., PII leak = 0). |
| Streaming           | Redis Streams | Solução leve para ingestão de feedback, escalável para Kafka em cenários de alta vazão (> 10k req/s). |

## 4. Estrutura de Diretórios

A organização do projeto segue um padrão rigoroso de MLOps, com responsabilidades bem definidas para cada diretório:

```
CHATCIELO/
├── src/                          # Código-fonte principal da aplicação
│   ├── config/                   # Configurações globais (Pydantic BaseSettings)
│   ├── data/                     # Módulos de processamento de dados (PII scrubbing, dataset)
│   ├── features/                 # Engenharia de features auxiliares e embeddings
│   ├── models/                   # Definição do PreferenceModel (DeBERTa-v3 + AuxFusionHead)
│   ├── training/                 # Lógica de treinamento (loop, otimizadores, perdas)
│   ├── inference/                # Motor de inferência em produção (Predictor, cache)
│   ├── evaluation/               # Módulos de avaliação de métricas e CLI
│   └── api/                      # API FastAPI (endpoints, middleware, schemas)
├── tests/                        # Suíte de testes (unitários, integração, ML, segurança, benchmarks)
├── scripts/                      # Scripts utilitários (LGPD audit, validação de dataset)
├── notebooks/                    # Notebooks de exploração de dados e prototipagem
├── docker/                       # Definições Docker (Dockerfile, docker-compose.yml)
├── intents/                      # Definições de domínios de intenção (e.g., Jurídico/LGPD)
├── artifacts/                    # Artefatos gerados (modelos treinados, gitignored)
├── data/                         # Dados de treinamento e validação (gitignored)
├── .env.example                  # Exemplo de arquivo de variáveis de ambiente
├── .pre-commit-config.yaml       # Configurações para hooks de pre-commit
├── pyproject.toml                # Configurações de projeto (Poetry/PEP 621)
├── requirements.txt              # Dependências do projeto
└── CLAUDE.md                     # Documentação específica do modelo CLAUDE (se aplicável)
```

## 5. Instalação e Uso

Para configurar e executar o projeto localmente, siga os passos abaixo:

1.  **Pré-requisitos**: Certifique-se de ter Docker e Docker Compose instalados.
2.  **Clonar o Repositório**:
    ```bash
    git clone https://github.com/seu-usuario/CHATCIELO.git
    cd CHATCIELO
    ```
3.  **Configurar Variáveis de Ambiente**: Crie um arquivo `.env` na raiz do projeto, baseado no `.env.example`, e preencha com as configurações necessárias.
4.  **Construir e Iniciar os Serviços**: Utilize Docker Compose para levantar a API, Redis, PostgreSQL, Prometheus e Grafana.
    ```bash
    docker-compose up --build -d
    ```
5.  **Treinar o Modelo (Opcional)**: Para treinar um novo modelo, execute o script de treinamento. Certifique-se de ter os dados de treinamento (`data/train.parquet`) disponíveis.
    ```bash
    python -m src.training.train
    ```
6.  **Acessar a API**: A API estará disponível em `http://localhost:8000`. A documentação interativa (Swagger UI) pode ser acessada em `http://localhost:8000/docs`.

## 6. Conformidade e Segurança (LGPD)

O CHATCIELO foi projetado com a LGPD em mente, implementando medidas robustas para a proteção de dados pessoais. Isso inclui:

*   **Scrubbing de PII**: Todas as entradas e saídas de texto são processadas para remover ou anonimizar PII antes do armazenamento ou processamento pelo modelo.
*   **Auditoria de PII**: Scripts de auditoria (`scripts/lgpd_audit.py`) varrem logs e saídas para garantir a ausência de PII.
*   **Retenção de Dados**: Políticas de retenção de dados são aplicadas para limitar o período de armazenamento de informações.
*   **Testes de Segurança**: Uma suíte de testes dedicada verifica a conformidade com as diretrizes de LGPD, incluindo a ausência de PII em logs e respostas da API.

## 7. Contribuição

Contribuições são bem-vindas! Por favor, consulte as diretrizes de contribuição (CONTRIBUTING.md, a ser criado) para mais detalhes sobre como propor melhorias, reportar bugs ou submeter *pull requests*.

## 8. Licença

Este projeto está licenciado sob a **LICENÇA MIT**.
