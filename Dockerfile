# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Copiar e instalar dependências Python (todas possuem wheels pré-compilados)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

WORKDIR /app

# Instalar dependência de runtime do LightGBM (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

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
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
