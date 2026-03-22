# 🏀 CHATCIELO - Core Adaptativo de Preferência

**Pipeline de Machine Learning para Personalização de LLMs com Ranking de Preferência e Conformidade LGPD**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/Transformers-4.x%2B-red)](https://huggingface.co/docs/transformers/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Sobre o Projeto

**CHATCIELO** é um sistema de Machine Learning de missão crítica, desenvolvido para a Cielo Brasil, que implementa um modelo de **Pairwise Preference Ranking** para personalizar o atendimento a lojistas. Inspirado nas abordagens *state-of-the-art* da competição Kaggle LMSYS, este pipeline adapta as respostas de Large Language Models (LLMs) com base no perfil do usuário (MEI, Varejo, Corporate), visando uma experiência conversacional de alta conversão e baixo *churn*.

O projeto adere a rigorosos padrões de MLOps, garantindo alta disponibilidade (latência P99 < 300ms) e conformidade estrita com a Lei Geral de Proteção de Dados (LGPD) através de mecanismos robustos de *scrubbing* de PII e auditoria.

### 🎯 Principais Características

*   ✅ **Modelo de Preferência Híbrido** — Cross-encoder DeBERTa-v3 com fusão de *features* auxiliares (segmento do lojista, características das respostas).
*   📊 **Função de Perda Combinada** — Otimização conjunta de *Label Smoothing Cross-Entropy* e *Margin Ranking Loss* para classificação e ordenação de preferências.
*   🔒 **Conformidade LGPD** — *Scrubbing* de PII em tempo real e pseudonimização unidirecional para dados sensíveis.
*   ⚡ **API de Baixa Latência** — Implementação com FastAPI e Uvicorn, suportada por cache Redis para inferência sub-300ms.
*   🔄 **Ciclo de Feedback Contínuo** — Detecção de *drift* de modelo via Redis Streams e retreinamento automatizado.
*   🧪 **Testes Abrangentes** — Suíte de testes unitários, de integração, de ML, de segurança (LGPD) e de *benchmarks* de latência.
*   🐳 **Containerização** — Imagens Docker otimizadas para implantação em produção.

---

## 🏗️ Estrutura do Projeto

A organização do projeto segue um padrão rigoroso de MLOps, com responsabilidades bem definidas para cada diretório, facilitando a manutenção e escalabilidade:

```
CHATCIELO/
├── src/                           # Código-fonte principal da aplicação
│   ├── config/                    # Configurações globais (Pydantic BaseSettings)
│   ├── data/                      # Módulos de processamento de dados (PII scrubbing, dataset)
│   ├── features/                  # Engenharia de features auxiliares e embeddings
│   ├── models/                    # Definição do PreferenceModel (DeBERTa-v3 + AuxFusionHead)
│   ├── training/                  # Lógica de treinamento (loop, otimizadores, perdas)
│   ├── inference/                 # Motor de inferência em produção (Predictor, cache)
│   ├── evaluation/                # Módulos de avaliação de métricas e CLI
│   └── api/                       # API FastAPI (endpoints, middleware, schemas)
├── tests/                         # Suíte de testes (unitários, integração, ML, segurança, benchmarks)
├── scripts/                       # Scripts utilitários (LGPD audit, validação de dataset)
├── notebooks/                     # Notebooks de exploração de dados e prototipagem
├── docker/                        # Definições Docker (Dockerfile, docker-compose.yml)
├── intents/                       # Definições de domínios de intenção (e.g., Jurídico/LGPD)
├── artifacts/                     # Artefatos gerados (modelos treinados, gitignored)
├── data/                          # Dados de treinamento e validação (gitignored)
├── .env.example                   # Exemplo de arquivo de variáveis de ambiente
├── .pre-commit-config.yaml        # Configurações para hooks de pre-commit
├── pyproject.toml                 # Configurações de projeto (Poetry/PEP 621)
├── requirements.txt               # Dependências do projeto
└── AUDITORIA_TECNICA.md           # Relatório de auditoria técnica detalhada
```

---

## 🚀 Quick Start

### Instalação

Para configurar o ambiente de desenvolvimento e executar o projeto localmente, siga os passos abaixo:

```bash
git clone https://github.com/joaopedropassostocantins/CHATCIELO.git # Assumindo que o repositório está aqui
cd CHATCIELO
pip install -r requirements.txt
```

### Configuração de Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto, baseado no `.env.example`, e preencha com as configurações necessárias para o ambiente (e.g., `REDIS_URL`, `DATABASE_URL`, `APP_SECRET_KEY`).

### Executar Serviços com Docker Compose

Utilize Docker Compose para levantar a API, Redis, PostgreSQL, Prometheus e Grafana:

```bash
docker-compose -f docker/docker-compose.yml up --build -d
```

### Treinar o Modelo (Opcional)

Para treinar um novo modelo, execute o script de treinamento. Certifique-se de ter os dados de treinamento (`data/train.parquet`) disponíveis:

```bash
python -m src.training.train
```

### Acessar a API

A API estará disponível em `http://localhost:8000`. A documentação interativa (Swagger UI) pode ser acessada em `http://localhost:8000/docs`.

---

## 📊 Metodologia Técnica

### 1. **Engenharia de Features**

O sistema emprega um conjunto de 12 *features* auxiliares, calculadas em `src/features/feature_engineering.py`, para enriquecer a representação textual e contextual. Estas *features* são categorizadas em:

*   **Features Estruturais por Resposta**: Comprimento do texto, média de palavras por frase (proxy de formalidade) e *Type-Token Ratio* (diversidade lexical) para cada resposta candidata.
*   **Features Comparativas**: Delta de comprimento entre as respostas, similaridade Jaccard (sobreposição lexical) e delta de média de palavras por frase.
*   **Segmento do Lojista**: Representação *one-hot encoded* do segmento do lojista (MEI, Varejo, Corporate), injetando informações cruciais sobre o perfil do usuário.

### 2. **Modelagem de Preferência**

O coração do CHATCIELO é o `PreferenceModel`, um *cross-encoder* baseado em **DeBERTa-v3-large**. Este modelo é otimizado para a tarefa de *pairwise preference ranking* através de uma `AuxFusionHead` que concatena o *pooler output* do DeBERTa com as *features* auxiliares. A saída do modelo são probabilidades para três classes: A vence, B vence, ou empate ($P(A \text{ wins}), P(B \text{ wins}), P(\text{tie})$).

### 3. **Função de Perda**

Para otimizar o modelo, utiliza-se uma `CombinedPreferenceLoss`, que é uma combinação ponderada de duas funções de perda:

*   **Label Smoothing Cross-Entropy**: Previne o *overfitting* e melhora a generalização, suavizando os rótulos verdadeiros com um fator $\epsilon$.
*   **Margin Ranking Loss**: Reforça a ordem de preferência entre as respostas A e B, excluindo casos de empate, garantindo que o modelo aprenda a distinguir entre as opções preferidas.

### 4. **Validação e Monitoramento**

A validação do modelo é realizada através de métricas como AUC, Accuracy, Log-loss e ECE (Expected Calibration Error). O sistema incorpora um robusto *feedback loop* com detecção de *drift* de dados (PSI - Population Stability Index) e degradação de performance (e.g., queda na AUC), que dispara automaticamente o retreinamento do modelo. O monitoramento contínuo é feito via Prometheus e Grafana, rastreando latência P99, acurácia e, criticamente, a contagem de vazamentos de PII (com meta de zero).

---

## 📚 Stack Tecnológico

A tabela a seguir detalha as principais tecnologias empregadas no desenvolvimento do CHATCIELO e suas respectivas justificativas técnicas:

| Camada / Componente | Tecnologia | Versão | Propósito |
| :------------------ | :--------- | :----- | :-------- |
| **API**             | FastAPI + Uvicorn | 0.100+ | Framework ASGI assíncrono para alta performance e baixa latência de inferência. |
| **Cache & Filas**   | Redis      | 7+     | *Lookup* em sub-milissegundos, TTL nativo e suporte a *streams* para o *feedback loop* e cache de resultados. |
| **Banco de Dados**  | PostgreSQL | 15+    | Retenção robusta do histórico de conversas e *audit logs* (LGPD) em esquema tipado. |
| **ML Runtime**      | PyTorch    | 2.2+   | Suporte nativo ao DeBERTa-v3 com aceleração via CUDA/MPS/CPU para treinamento e inferência de modelos de *deep learning*. |
| **Transformers**    | HuggingFace | 4.x+   | Biblioteca para carregamento e utilização do modelo DeBERTa-v3 e tokenizadores. |
| **Containerização** | Docker + Docker Compose | Latest | Permite *builds* isolados e multi-stage, garantindo imagens leves e portáveis para produção. |
| **Observabilidade** | Prometheus + Grafana | Latest | Ferramentas padrão da indústria para rastreamento de latência P99, métricas de modelo e contadores críticos de segurança (e.g., PII leak = 0). |
| **Processamento de Dados** | Pandas + NumPy | 1.3+ / 1.20+ | Manipulação eficiente de dados e operações numéricas para engenharia de *features* e pré-processamento. |

---

## 📖 Documentação Completa

Para uma compreensão aprofundada dos aspectos técnicos, decisões de design e detalhes de implementação, consulte a documentação complementar:

*   **[AUDITORIA_TECNICA.md](AUDITORIA_TECNICA.md)** — Relatório de auditoria técnica detalhada, incluindo formulações matemáticas, análise de componentes e recomendações de otimização.
*   **[notebooks/01_eda_lmsys.ipynb](notebooks/01_eda_lmsys.ipynb)** — Notebook de exploração de dados e prototipagem inicial.
*   **[CLAUDE.md](CLAUDE.md)** — Documentação específica do modelo CLAUDE (se aplicável).

---

## 🎓 Uso em Notebooks

O diretório `notebooks/` contém exemplos práticos de como interagir com os componentes do CHATCIELO. O notebook `01_eda_lmsys.ipynb` demonstra:

*   Carregamento e pré-processamento de dados.
*   Cálculo de *features* auxiliares.
*   Exemplos de treinamento e avaliação de modelos.
*   Visualização de resultados e métricas.

---

## 📝 Licença

Este projeto é licenciado sob a **[MIT License](LICENSE)**. Você é livre para usar, modificar e distribuir, com ou sem fins comerciais, conforme os termos da licença.

---

## 👤 Autoria & Créditos

**Autor Principal:** João Pedro Passos Tocantins

**Contribuições Técnicas:**
*   Arquitetura e design do pipeline de Machine Learning.
*   Desenvolvimento do sistema de *Pairwise Preference Ranking*.
*   Implementação de *features* auxiliares e funções de perda combinadas.
*   Framework de *backtesting* temporal e validação de modelos.
*   Auditoria e documentação técnica.

**Referências:**
*   Kaggle LMSYS Chatbot Arena Competition
*   Documentação oficial do PyTorch, HuggingFace Transformers e FastAPI.
*   Artigos de pesquisa sobre *Label Smoothing* e *Margin Ranking Loss*.

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Para reportar bugs, sugerir melhorias ou submeter *pull requests*, por favor, consulte as diretrizes de contribuição (CONTRIBUTING.md, a ser criado) e siga o fluxo padrão do GitHub:

1.  Abra uma [issue](https://github.com/joaopedropassostocantins/CHATCIELO/issues) para discutir a mudança proposta.
2.  Faça *fork* do repositório e crie um *branch* para sua *feature* ou correção.
3.  Realize suas modificações e *commit* as alterações com mensagens claras.
4.  Envie suas alterações (*push*) para o seu *fork*.
5.  Abra um *Pull Request* para o repositório principal.

---

## 📧 Contato

*   **GitHub:** [@joaopedropassostocantins](https://github.com/joaopedropassostocantins)

---

## 🔗 Links Úteis

*   [Documentação FastAPI](https://fastapi.tiangolo.com/)
*   [Documentação HuggingFace Transformers](https://huggingface.co/docs/transformers/)
*   [Documentação PyTorch](https://pytorch.org/docs/stable/index.html)

---

**Última Atualização:** Março de 2026 | **Versão:** 1.0 | **Status:** Produção Pronta ✅
