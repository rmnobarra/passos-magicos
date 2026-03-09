# CLAUDE.md — Datathon Passos Mágicos: ML Engineering

## 🎯 Visão Geral do Projeto

**Objetivo:** Construir um sistema completo de Machine Learning para prever o **risco de defasagem escolar** de estudantes atendidos pela Associação Passos Mágicos, cobrindo todo o ciclo de vida do modelo com boas práticas de MLOps.

**Dataset:** Dados de desenvolvimento educacional dos anos 2022, 2023 e 2024 da Associação Passos Mágicos.  
**Fonte:** https://www.kaggle.com/datasets/filipiimperial/passos-magicos (baixar e colocar em `data/raw/`)

---

## 🏗️ Estrutura do Projeto

```
passos-magicos/
├── CLAUDE.md                  ← este arquivo
├── agents/                    ← sub-agentes especializados
│   ├── data_engineer.md
│   ├── ml_engineer.md
│   ├── api_engineer.md
│   ├── devops_engineer.md
│   └── qa_engineer.md
├── .claude/
│   └── commands/              ← comandos customizados Claude Code
│       ├── train.md
│       ├── test.md
│       ├── deploy.md
│       └── monitor.md
├── app/
│   ├── main.py                ← entrypoint FastAPI
│   ├── routes.py              ← endpoints /predict /health /metrics
│   ├── schemas.py             ← Pydantic models (input/output)
│   ├── middleware.py          ← logging middleware
│   └── model/                ← modelos serializados (.joblib)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       ← limpeza e preparação dos dados
│   ├── feature_engineering.py ← criação de features
│   ├── train.py               ← pipeline de treinamento
│   ├── evaluate.py            ← métricas e avaliação
│   └── utils.py               ← funções auxiliares e logging
├── monitoring/
│   ├── drift_detector.py      ← detecção de data drift
│   └── dashboard.py           ← dashboard Evidently/Streamlit
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   └── test_api.py
├── notebooks/
│   └── 01_eda.ipynb           ← análise exploratória
├── data/
│   ├── raw/                   ← dados originais (não versionados)
│   └── processed/             ← dados processados
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📋 Instruções Gerais para o Claude Code

### Linguagem e estilo
- Todo código em **Python 3.11+**
- Docstrings em **português brasileiro** (comentários e docs)
- Type hints em todas as funções
- Seguir PEP 8; usar `black` para formatação
- Logs estruturados com `logging` (nível INFO em produção, DEBUG em dev)

### Ordem de construção recomendada
1. Estrutura de diretórios e `requirements.txt`
2. `src/utils.py` (logging, constantes)
3. `src/preprocessing.py`
4. `src/feature_engineering.py`
5. `src/train.py` + `src/evaluate.py`
6. `app/` (FastAPI)
7. `tests/` (pytest, ≥80% cobertura)
8. `monitoring/`
9. `Dockerfile` + `docker-compose.yml`
10. `README.md`

---

## 📊 Sobre o Dataset — Passos Mágicos

### Features conhecidas do dataset
| Feature | Descrição |
|---------|-----------|
| `INDE` | Índice de Desenvolvimento Educacional (0–10) |
| `IAN` | Indicador de Adequação de Nível |
| `IDA` | Indicador de Desempenho Acadêmico |
| `IEG` | Indicador de Engajamento |
| `IAA` | Indicador de Autoavaliação |
| `IPS` | Indicador Psicossocial |
| `IPP` | Indicador Psicopedagógico |
| `IPV` | Indicador de Ponto de Virada |
| `FASE` | Fase educacional (0–8) |
| `TURMA` | Turma do estudante |
| `ANO` | Ano letivo (2022, 2023, 2024) |
| `PONTO_DE_VIRADA` | Flag de ponto de virada (Sim/Não) |
| `DEFASAGEM` | **Target** — risco de defasagem escolar |

### Target
- `DEFASAGEM`: variável binária (0 = sem risco, 1 = com risco de defasagem)
- Se o dataset não tiver `DEFASAGEM` explícita, **derivar** a partir de: estudante com `INDE < 5.0` E `FASE` inadequada para a idade

---

## 🤖 Modelo Preditivo

### Algoritmo recomendado
Usar **Random Forest Classifier** como baseline, comparar com **XGBoost** e **LightGBM**.

### Métricas de avaliação
- **Métrica principal: F1-Score (macro)** — balanceia precision e recall em dataset potencialmente desbalanceado
- Métricas secundárias: ROC-AUC, Precision, Recall, Confusion Matrix
- Justificativa no README: falsos negativos (não identificar aluno em risco) têm custo social alto → F1 é mais adequado que simples Accuracy

### Pipeline de treinamento
```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])
```

### Validação
- Estratégia: `StratifiedKFold(n_splits=5)` para preservar proporção das classes
- Salvar modelo em: `app/model/model.joblib`
- Salvar preprocessor separado em: `app/model/preprocessor.joblib`

---

## 🌐 API FastAPI

### Endpoint obrigatório: `POST /predict`

**Request body:**
```json
{
  "inde": 6.5,
  "ian": 7.2,
  "ida": 5.8,
  "ieg": 6.0,
  "iaa": 7.5,
  "ips": 6.8,
  "ipp": 7.1,
  "ipv": 6.3,
  "fase": 3,
  "ano": 2024,
  "ponto_de_virada": false
}
```

**Response:**
```json
{
  "student_id": null,
  "risco_defasagem": true,
  "probabilidade": 0.73,
  "nivel_risco": "alto",
  "modelo_versao": "1.0.0",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Outros endpoints
- `GET /health` — health check
- `GET /metrics` — métricas do modelo e contagem de predições
- `GET /docs` — Swagger automático do FastAPI

---

## 🧪 Testes Unitários

- Framework: **pytest** com **pytest-cov**
- Cobertura mínima: **80%**
- Usar fixtures em `conftest.py` para dados de teste sintéticos
- Testar: preprocessing, feature engineering, train pipeline, API endpoints
- Comando: `pytest tests/ --cov=src --cov=app --cov-report=term-missing`

---

## 📦 Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Comando para build e run:
```bash
docker build -t passos-magicos-api .
docker run -p 8000:8000 passos-magicos-api
```

---

## 📡 Monitoramento

- **Logs:** `logging` estruturado com JSON, salvo em `logs/api.log`
- **Drift:** usar biblioteca `evidently` para monitorar data drift
- **Dashboard:** Streamlit em `monitoring/dashboard.py` na porta 8501
- Registrar cada predição com timestamp, features de entrada e resultado

---

## 🔧 requirements.txt (referência)

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
scikit-learn==1.4.2
pandas==2.2.2
numpy==1.26.4
joblib==1.4.2
xgboost==2.0.3
lightgbm==4.3.0
pydantic==2.7.1
python-dotenv==1.0.1
evidently==0.4.30
streamlit==1.35.0
pytest==8.2.0
pytest-cov==5.0.0
httpx==0.27.0
black==24.4.2
```

---

---

## ☁️ Deploy na Render (Cloud)

### Arquitetura
```
GitHub push (main)
    ↓
GitHub Actions CI  →  lint + test + docker build
    ↓ (CI verde)
GitHub Actions CD  →  dispara Render API → aguarda deploy → health check
    ↓
Render Web Service (Docker)
    ├── Persistent Disk: /app/app/model  (modelo persistido entre deploys)
    ├── Health Check:    /api/v1/health
    └── URL pública:     https://passos-magicos-api.onrender.com
```

### Agentes responsáveis
- **`agents/github_actions_engineer.md`** → workflows CI/CD completos
- **`agents/render_engineer.md`** → render.yaml, startup script, configuração do serviço

### Secrets necessários no GitHub
| Secret | Onde obter |
|--------|-----------|
| `RENDER_API_KEY` | Render → Account Settings → API Keys |
| `RENDER_SERVICE_ID` | URL do serviço: `srv-XXXXXXXXXXXX` |
| `RENDER_SERVICE_URL` | Ex: `https://passos-magicos-api.onrender.com` |

### Comandos rápidos de CI/CD
```bash
# Verificar status do workflow
gh workflow list
gh run list --workflow=ci.yml

# Disparar deploy manual
gh workflow run cd.yml

# Ver logs do último deploy
gh run view --log
```

---

## ✅ Checklist de Entrega

- [ ] Pipeline de treinamento completa (`src/train.py`)
- [ ] Modelo salvo em `app/model/model.joblib`
- [ ] API FastAPI com `/predict`, `/health`, `/metrics`
- [ ] Código modularizado em `src/`
- [ ] Testes unitários com ≥80% de cobertura
- [ ] Dockerfile funcional
- [ ] docker-compose.yml
- [ ] Monitoramento com logs e dashboard de drift
- [ ] README.md completo com instruções de deploy
- [ ] API testada via curl/Postman
- [ ] GitHub Actions CI rodando (lint + test + docker build)
- [ ] GitHub Actions CD fazendo deploy automático na Render
- [ ] render.yaml e startup script configurados
- [ ] API pública acessível na URL da Render
- [ ] Repositório GitHub organizado e documentado
- [ ] Vídeo de até 5 minutos (formato gerencial)

---

## 🚀 Comandos Rápidos

```bash
# Treinar modelo
python src/train.py

# Subir API localmente
uvicorn app.main:app --reload --port 8000

# Rodar testes
pytest tests/ --cov=src --cov=app --cov-report=term-missing

# Build Docker
docker build -t passos-magicos-api .
docker run -p 8000:8000 passos-magicos-api

# Docker Compose (API + Monitor)
docker-compose up --build

# Dashboard de monitoramento
streamlit run monitoring/dashboard.py
```
