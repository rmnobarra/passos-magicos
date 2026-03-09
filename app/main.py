"""
Entrypoint da API Passos Mágicos.

Configura a aplicação FastAPI com logging estruturado, middleware de CORS,
middleware de logging de requisições e registra as rotas com prefixo /api/v1.

Uso:
    uvicorn app.main:app --reload --port 8000
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.middleware import LoggingMiddleware
from app.routes import router

# ── Diretório de logs ────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

# ── Logging estruturado ──────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação FastAPI.

    No startup: registra o horário de início e loga a inicialização.
    No shutdown: loga o encerramento da API.

    Args:
        app: Instância da aplicação FastAPI.
    """
    # Startup
    app.state.start_time = time.time()
    logger.info("API Passos Mágicos iniciada — versão 1.0.0")
    logger.info("Documentação disponível em: /docs e /redoc")

    yield

    # Shutdown
    logger.info("API Passos Mágicos encerrada")


# ── Aplicação FastAPI ────────────────────────────────────────────────────────

app = FastAPI(
    title="Passos Mágicos — API de Risco de Defasagem",
    description=(
        "API para predição de **risco de defasagem escolar** dos estudantes "
        "atendidos pela Associação Passos Mágicos.\n\n"
        "### Endpoints principais\n"
        "- `POST /api/v1/predict` — Predição de risco para um estudante\n"
        "- `GET /api/v1/health` — Health check da API\n"
        "- `GET /api/v1/metrics` — Estatísticas de uso\n\n"
        "### Modelo\n"
        "O modelo foi treinado com dados dos anos 2020–2022 usando "
        "**LGBMClassifier** com validação cruzada estratificada (5 folds). "
        "Métrica principal: F1-Score macro."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middlewares ──────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LoggingMiddleware)

# ── Rotas ────────────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api/v1")
