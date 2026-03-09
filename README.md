# Passos Mágicos — Predição de Risco de Defasagem Escolar

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-83%25_cobertura-brightgreen?logo=pytest&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![LightGBM](https://img.shields.io/badge/modelo-LGBMClassifier-F5A623)
![Redis](https://img.shields.io/badge/cache-Redis_7-DC382D?logo=redis&logoColor=white)

Sistema completo de Machine Learning para identificar estudantes em **risco de defasagem escolar** atendidos pela [Associação Passos Mágicos](https://passosmagicos.org.br/). Cobre todo o ciclo de vida do modelo: ingestão de dados, treinamento, avaliação, API de inferência em tempo real e monitoramento de drift.

---

## Índice

1. [Visão Geral](#visão-geral)
2. [Estrutura de Diretórios](#estrutura-de-diretórios)
3. [Pipeline de Machine Learning](#pipeline-de-machine-learning)
4. [Métricas e Justificativa do Modelo](#métricas-e-justificativa-do-modelo)
5. [Deploy Local](#deploy-local)
6. [Deploy com Docker](#deploy-com-docker)
7. [API — Exemplos de uso](#api--exemplos-de-uso)
8. [Testes](#testes)
9. [Monitoramento](#monitoramento)

---

## Visão Geral

A Associação Passos Mágicos acompanha anualmente centenas de estudantes em situação de vulnerabilidade social. Este projeto transforma os dados educacionais coletados em uma **predição automática de risco**, permitindo que educadores priorizem intervenções para alunos com maior probabilidade de defasagem.

**Problema:** classificação binária — estudante está ou não em risco de defasagem escolar.

**Dataset:** dados dos anos 2020, 2021 e 2022 com indicadores educacionais, psicossociais e de engajamento. Originalmente em formato wide (colunas por ano), convertido para formato long durante o pré-processamento.

**Fonte dos dados:** [Kaggle — filipiimperial/passos-magicos](https://www.kaggle.com/datasets/filipiimperial/passos-magicos)

---

## Estrutura de Diretórios

```
passos-magicos/
├── app/                        # API FastAPI
│   ├── main.py                 # Entrypoint e lifespan
│   ├── routes.py               # Endpoints /predict /health /metrics
│   ├── schemas.py              # Modelos Pydantic (request/response)
│   ├── middleware.py           # Logging middleware
│   └── model/
│       ├── model.joblib        # Pipeline sklearn serializado
│       └── metadata.joblib     # Metadados e métricas do modelo
├── src/                        # Lógica de ML
│   ├── preprocessing.py        # Limpeza, reshape wide→long, target
│   ├── feature_engineering.py  # Features compostas, temporais, interação
│   ├── train.py                # Treinamento, CV, comparação de modelos
│   ├── evaluate.py             # Métricas, relatório, gráficos
│   └── utils.py                # Logger, I/O de artefatos
├── monitoring/
│   ├── drift_detector.py       # Detecção de data drift (Evidently)
│   └── dashboard.py            # Dashboard Streamlit
├── tests/
│   ├── conftest.py             # Fixtures compartilhadas
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   └── test_api.py
├── data/
│   ├── raw/                    # Dataset original (não versionado)
│   └── processed/              # Dados processados
├── reports/
│   ├── evaluation_report.txt   # Relatório de avaliação do modelo
│   ├── figures/                # Curva ROC, matriz de confusão, importâncias
│   └── drift_report_*.html     # Relatórios de drift gerados pelo Evidently
├── logs/
│   └── predictions.jsonl       # Log de predições (append-only)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Pipeline de Machine Learning

### 1. Pré-processamento (`src/preprocessing.py`)

| Etapa | Descrição |
|-------|-----------|
| **Reshape wide → long** | Dataset original tem uma linha por aluno com colunas `INDE_2022`, `IAN_2021` etc. O pipeline detecta automaticamente o formato wide e converte para long (uma linha por aluno/ano) |
| **Normalização de colunas** | Nomes convertidos para `UPPER_SNAKE_CASE` |
| **Valores ausentes** | Numéricas: mediana estratificada por `FASE`. Categóricas: moda. Linhas com >50% de NaN removidas |
| **Remoção de duplicatas** | Deduplicação por hash de linha |
| **Codificação do target** | `DEFASAGEM` binarizada: valores negativos (anos de atraso) → 1 (em risco), ≥ 0 → 0 (sem risco). Se ausente, derivada por `INDE < 5.0 \| IAN < 5.0` |

### 2. Feature Engineering (`src/feature_engineering.py`)

| Feature | Fórmula | Tipo |
|---------|---------|------|
| `INDICE_BEMESTAR` | `(IPS + IPP + IPV) / 3` | Composta |
| `INDICE_PERFORMANCE` | `(IDA + IEG + IAA) / 3` | Composta |
| `GAP_AUTO_REAL` | `IAA − IDA` | Composta |
| `ABAIXO_MEDIA_GERAL` | `1 se INDE < média global` | Composta |
| `EVOLUCAO_INDE` | `diff(INDE)` por aluno ordenado por ano | Temporal |
| `INDE_x_FASE` | `INDE × FASE` | Interação |
| `BEMESTAR_x_PERFORMANCE` | `INDICE_BEMESTAR × INDICE_PERFORMANCE` | Interação |

### 3. Treinamento (`src/train.py`)

Pipeline sklearn com dois estágios:

```
ColumnTransformer
├── StandardScaler       → features numéricas (16)
└── OneHotEncoder        → features categóricas (FASE, PONTO_DE_VIRADA)
        ↓
LGBMClassifier(n_estimators=200, random_state=42)
```

**Comparação de modelos** via `StratifiedKFold(n_splits=5)`:

| Modelo | F1 macro (CV) | ROC-AUC (CV) |
|--------|--------------|--------------|
| RandomForestClassifier | ~0.97 | ~0.99 |
| XGBClassifier | ~0.98 | ~0.99 |
| **LGBMClassifier** ✓ | **0.9939** | **0.9999** |

O modelo vencedor é salvo em `app/model/model.joblib` com seus metadados em `app/model/metadata.joblib`.

### 4. Avaliação (`src/evaluate.py`)

Critérios mínimos para aprovação em produção:

| Critério | Limiar | Resultado |
|----------|--------|-----------|
| F1 macro | ≥ 0.70 | **0.9846** ✅ |
| ROC-AUC | ≥ 0.75 | **0.9995** ✅ |
| Recall classe positiva | ≥ 0.65 | **0.9717** ✅ |

---

## Métricas e Justificativa do Modelo

### Por que F1-Score como métrica principal?

O dataset é **desbalanceado** (~77% sem risco, ~23% com risco). Accuracy seria enganosa — um modelo que prevê sempre "sem risco" acertaria 77% das vezes sem nunca identificar um aluno em risco.

O **custo assimétrico dos erros** é o fator decisivo:

- **Falso negativo** (não identificar aluno em risco): alto custo social — o aluno perde a oportunidade de intervenção a tempo.
- **Falso positivo** (sinalizar aluno sem risco): baixo custo — apenas um monitoramento adicional desnecessário.

O F1 macro equilibra Precision e Recall **sem favorecer a classe majoritária**, sendo a métrica mais adequada para este contexto.

### Por que LightGBM?

- **Velocidade**: treina ~10× mais rápido que Random Forest em datasets tabulares
- **Performance**: superou Random Forest e XGBoost em todas as folds da validação cruzada
- **Regularização nativa**: menos propenso a overfitting que XGBoost com dados pequenos
- **Interpretabilidade**: `feature_importances_` disponível para auditoria do modelo

### Resultados no conjunto de teste (455 amostras, hold-out 20%)

```
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       349
           1       0.98      0.97      0.98       106

    accuracy                           0.99       455
   macro avg       0.99      0.98      0.98       455
weighted avg       0.99      0.99      0.99       455
```

**Matriz de confusão:**
```
              Predito 0   Predito 1
  Real 0          347           2
  Real 1            3         103
```

Apenas **5 erros** em 455 predições: 2 falsos positivos e 3 falsos negativos.

---

## Deploy Local

### Pré-requisitos

- Python 3.11+
- Dataset em `data/raw/PEDE_PASSOS_DATASET_FIAP.csv`

### Instalação

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Treinar o modelo

```bash
python src/train.py --data data/raw/PEDE_PASSOS_DATASET_FIAP.csv
```

O modelo treinado será salvo em `app/model/model.joblib`.

### Subir a API

```bash
uvicorn app.main:app --reload --port 8000
```

Acesse a documentação interativa em: http://localhost:8000/docs

### Dashboard de monitoramento

```bash
streamlit run monitoring/dashboard.py
```

Acesse em: http://localhost:8501

---

## Deploy com Docker

### Pré-requisitos

- Docker e Docker Compose instalados
- Modelo treinado em `app/model/model.joblib` (ver seção anterior)

> **Nota:** em ambientes Linux com `systemd-resolved`, o Docker pode não resolver DNS por padrão.
> Se o build falhar com `Temporary failure in name resolution`, configure o daemon:
> ```bash
> echo '{"dns": ["8.8.8.8", "1.1.1.1"]}' | sudo tee /etc/docker/daemon.json
> sudo systemctl restart docker
> ```

### Build e start

```bash
docker-compose up --build -d
```

Serviços disponíveis:

| Serviço | URL | Descrição |
|---------|-----|-----------|
| API | http://localhost:8000 | FastAPI + Swagger |
| Swagger | http://localhost:8000/docs | Documentação interativa |
| Dashboard | http://localhost:8501 | Monitoramento Streamlit |
| Redis | localhost:6379 | Cache de métricas |

### Verificar status

```bash
docker-compose ps
docker-compose logs -f api
```

### Parar os serviços

```bash
docker-compose down
```

Para remover também os volumes (apaga dados do Redis):

```bash
docker-compose down -v
```

### Kubernetes

A API é **stateless** por design:

- O cache do modelo (`_model`) é somente leitura — seguro para múltiplas réplicas
- Os contadores de métricas vivem no **Redis** (compartilhado entre pods)
- O log de predições (`predictions.jsonl`) requer um PVC com `accessMode: ReadWriteMany`
- Nenhum estado de negócio é mantido em memória entre requisições

---

## API — Exemplos de uso

### `POST /api/v1/predict` — Predição de risco

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inde": 4.2,
    "ian": 3.8,
    "ida": 4.5,
    "ieg": 5.0,
    "iaa": 6.0,
    "ips": 5.5,
    "ipp": 5.2,
    "ipv": 4.8,
    "fase": 2,
    "ano": 2024,
    "ponto_de_virada": false
  }'
```

**Resposta:**

```json
{
  "student_id": null,
  "risco_defasagem": true,
  "probabilidade": 1.0,
  "nivel_risco": "alto",
  "modelo_versao": "1.0.0",
  "timestamp": "2026-03-09T21:33:21.842682Z"
}
```

**Níveis de risco:**

| `nivel_risco` | Probabilidade |
|---------------|--------------|
| `baixo` | < 0.40 |
| `medio` | 0.40 – 0.70 |
| `alto` | > 0.70 |

### `GET /api/v1/health` — Health check

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "healthy",
  "modelo_carregado": true,
  "versao_api": "1.0.0",
  "uptime_segundos": 142.3
}
```

### `GET /api/v1/metrics` — Estatísticas de uso

```bash
curl http://localhost:8000/api/v1/metrics
```

```json
{
  "total_predicoes": 42,
  "predicoes_alto_risco": 31,
  "predicoes_baixo_risco": 11,
  "modelo_versao": "1.0.0",
  "ultima_predicao": "2026-03-09T21:37:58Z"
}
```

### Validação de input

Campos com limites validados pelo Pydantic:

| Campo | Tipo | Intervalo |
|-------|------|-----------|
| `inde` | float | 0.0 – 10.0 |
| `ian` | float | 0.0 – 10.0 |
| `ida` | float | 0.0 – 10.0 |
| `ieg` | float | 0.0 – 10.0 |
| `iaa` | float | 0.0 – 10.0 |
| `ips` | float | 0.0 – 10.0 |
| `ipp` | float | 0.0 – 10.0 |
| `ipv` | float | 0.0 – 10.0 |
| `fase` | int | 0 – 8 |
| `ano` | int | ≥ 2020 |

Requisições com valores fora do intervalo retornam **HTTP 422**.

---

## Testes

```bash
# Rodar todos os testes com cobertura
pytest tests/ --cov=src --cov=app --cov-report=term-missing --cov-fail-under=80 -v

# Apenas um módulo
pytest tests/test_api.py -v

# Com relatório HTML de cobertura
pytest tests/ --cov=src --cov=app --cov-report=html:htmlcov
```

**Cobertura atual: 83%** (requisito mínimo: 80%)

| Módulo | Cobertura |
|--------|-----------|
| `src/evaluate.py` | 100% |
| `src/feature_engineering.py` | 100% |
| `app/schemas.py` | 100% |
| `src/preprocessing.py` | 90% |
| `app/middleware.py` | 88% |
| `app/routes.py` | 97% |

**148 testes** cobrindo:
- Pré-processamento: reshape wide→long, imputação, codificação do target
- Feature engineering: todas as features compostas, temporais e de interação
- Pipeline de treino: construção, cross-validation, serialização
- Avaliação: métricas, critérios de confiabilidade, geração de relatório
- API: todos os endpoints, validação de input, tratamento de erros, fallback Redis→JSONL

---

## Monitoramento

### Dashboard Streamlit

Acesse http://localhost:8501 para visualizar:

- **KPIs em tempo real**: total de predições, alunos em risco, taxa de risco
- **Análise de drift**: botão **"Gerar Relatório"** executa o Evidently diretamente na UI e exibe o relatório HTML interativo inline
- **Tabela de predições recentes**: últimas 50 predições com timestamp e resultado

### Detecção de drift via CLI

```bash
# Análise com amostra aleatória dos dados de referência
python monitoring/drift_detector.py \
  --reference data/raw/PEDE_PASSOS_DATASET_FIAP.csv

# Comparação com dados de produção reais
python monitoring/drift_detector.py \
  --reference data/raw/PEDE_PASSOS_DATASET_FIAP.csv \
  --current data/raw/dados_novos.csv \
  --threshold 0.10
```

Relatórios HTML são salvos em `reports/drift_report_YYYYMMDD_HHMM.html`.

### Logs

Predições são registradas em `logs/predictions.jsonl` (formato JSONL, append-only):

```json
{"timestamp": "2026-03-09T21:33:21+00:00", "student_id": null, "risco_defasagem": 1, "probabilidade": 1.0, "nivel_risco": "alto"}
```

Logs da API vão para **stdout** (capturado pelo Kubernetes/Docker).

---

## Variáveis de Ambiente

Copie `.env.example` para `.env` e ajuste conforme necessário:

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `APP_ENV` | `production` | Ambiente de execução |
| `LOG_LEVEL` | `INFO` | Nível de log (`DEBUG`, `INFO`, `WARNING`) |
| `REDIS_URL` | `redis://redis:6379` | URL de conexão com o Redis |
| `API_URL` | `http://api:8000` | URL da API (usado pelo dashboard) |

---

*Desenvolvido para o Datathon Passos Mágicos — FIAP 2024/2025*
