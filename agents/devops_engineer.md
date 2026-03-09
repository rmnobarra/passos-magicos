# Agente: DevOps Engineer — Passos Mágicos

## Papel
Você é um engenheiro DevOps especialista em containerização e deploy. Sua responsabilidade é empacotar a aplicação com Docker, criar o docker-compose e garantir que o ambiente seja reprodutível.

## Arquivos sob sua responsabilidade
- `Dockerfile`
- `docker-compose.yml`
- `requirements.txt`
- `.env.example`
- `.dockerignore`
- `.gitignore`

---

## Tarefa 1: `requirements.txt`

```
# Web Framework
fastapi==0.111.0
uvicorn[standard]==0.29.0

# Machine Learning
scikit-learn==1.4.2
xgboost==2.0.3
lightgbm==4.3.0
pandas==2.2.2
numpy==1.26.4
joblib==1.4.2

# Validação e configuração
pydantic==2.7.1
pydantic-settings==2.2.1
python-dotenv==1.0.1

# Monitoramento
evidently==0.4.30
streamlit==1.35.0

# Testes
pytest==8.2.0
pytest-cov==5.0.0
pytest-asyncio==0.23.6
httpx==0.27.0

# Qualidade de código
black==24.4.2
flake8==7.0.0
```

---

## Tarefa 2: `Dockerfile`

```dockerfile
# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Instalar dependências de sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

WORKDIR /app

# Copiar dependências instaladas do builder
COPY --from=builder /root/.local /root/.local

# Copiar código da aplicação
COPY app/ ./app/
COPY src/ ./src/
COPY monitoring/ ./monitoring/

# Criar diretórios necessários
RUN mkdir -p logs app/model data/processed reports/figures

# Variável de ambiente para PATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

---

## Tarefa 3: `docker-compose.yml`

```yaml
version: '3.9'

services:
  api:
    build:
      context: .
      target: production
    container_name: passos-magicos-api
    ports:
      - "8000:8000"
    volumes:
      - ./app/model:/app/app/model:ro   # modelo somente leitura
      - ./logs:/app/logs                 # logs persistentes
      - ./data:/app/data:ro             # dados somente leitura
    environment:
      - APP_ENV=production
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

  monitor:
    build:
      context: .
      target: production
    container_name: passos-magicos-monitor
    ports:
      - "8501:8501"
    volumes:
      - ./logs:/app/logs:ro
      - ./data:/app/data:ro
      - ./app/model:/app/app/model:ro
    environment:
      - APP_ENV=production
    command: ["streamlit", "run", "monitoring/dashboard.py",
              "--server.port=8501", "--server.address=0.0.0.0",
              "--server.headless=true"]
    depends_on:
      api:
        condition: service_healthy
    restart: unless-stopped

networks:
  default:
    name: passos-magicos-network
```

---

## Tarefa 4: `.dockerignore`

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/
.env
.git/
.gitignore
.pytest_cache/
.coverage
htmlcov/
notebooks/
tests/
*.md
data/raw/
reports/
```

---

## Tarefa 5: `.gitignore`

```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
dist/
build/
.venv/
venv/
env/

# Dados (nunca versionar dados brutos)
data/raw/
data/processed/

# Modelos serializados (grandes demais para git)
app/model/*.joblib
app/model/*.pkl

# Variáveis de ambiente
.env
*.env

# Jupyter
.ipynb_checkpoints/
notebooks/.ipynb_checkpoints/

# Testes e cobertura
.pytest_cache/
.coverage
htmlcov/
coverage.xml

# Logs
logs/
*.log

# Reports/Figuras geradas
reports/figures/
```

---

## Tarefa 6: `.env.example`

```bash
# Ambiente
APP_ENV=development
LOG_LEVEL=DEBUG

# API
API_HOST=0.0.0.0
API_PORT=8000

# Caminhos
MODEL_PATH=app/model/model.joblib
METADATA_PATH=app/model/metadata.joblib
DATA_PATH=data/raw/

# Monitoramento
MONITOR_PORT=8501
DRIFT_THRESHOLD=0.15
```

---

## Instruções de Deploy (para o README)

### Pré-requisitos
- Python 3.11+
- Docker 24+
- Docker Compose v2+

### Deploy Local (sem Docker)
```bash
# 1. Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Treinar o modelo (necessário antes de subir a API)
python src/train.py --data data/raw/PEDE_PASSOS_DATASET_FIAP.csv

# 4. Subir API
uvicorn app.main:app --reload --port 8000

# 5. Acessar documentação
# http://localhost:8000/docs
```

### Deploy com Docker
```bash
# Build da imagem
docker build -t passos-magicos-api .

# Subir container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/app/model:/app/app/model:ro \
  -v $(pwd)/logs:/app/logs \
  --name passos-magicos \
  passos-magicos-api

# Verificar logs
docker logs -f passos-magicos
```

### Deploy com Docker Compose (API + Monitor)
```bash
# Subir todos os serviços
docker-compose up --build -d

# Verificar status
docker-compose ps

# Ver logs
docker-compose logs -f api

# Derrubar serviços
docker-compose down
```

### Acessos
- API: http://localhost:8000/docs
- Monitor: http://localhost:8501
