# Agente: API Engineer — Passos Mágicos

## Papel
Você é um engenheiro de software especialista em APIs. Sua responsabilidade é construir a API FastAPI para servir o modelo de predição de risco de defasagem escolar.

## Arquivos sob sua responsabilidade
- `app/main.py`
- `app/routes.py`
- `app/schemas.py`
- `app/middleware.py`
- `tests/test_api.py`

---

## Tarefa 1: `app/schemas.py`

Definir os Pydantic models para request e response:

```python
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime
from enum import Enum

class NivelRisco(str, Enum):
    baixo = "baixo"       # probabilidade < 0.40
    medio = "medio"       # probabilidade 0.40–0.70
    alto = "alto"         # probabilidade > 0.70

class PredictionRequest(BaseModel):
    student_id: Optional[str] = Field(None, description="ID opcional do estudante")
    inde: float = Field(..., ge=0, le=10, description="Índice de Desenvolvimento Educacional")
    ian: float = Field(..., ge=0, le=10, description="Indicador de Adequação de Nível")
    ida: float = Field(..., ge=0, le=10, description="Indicador de Desempenho Acadêmico")
    ieg: float = Field(..., ge=0, le=10, description="Indicador de Engajamento")
    iaa: float = Field(..., ge=0, le=10, description="Indicador de Autoavaliação")
    ips: float = Field(..., ge=0, le=10, description="Indicador Psicossocial")
    ipp: float = Field(..., ge=0, le=10, description="Indicador Psicopedagógico")
    ipv: float = Field(..., ge=0, le=10, description="Indicador de Ponto de Virada")
    fase: int = Field(..., ge=0, le=8, description="Fase educacional (0–8)")
    ano: int = Field(..., ge=2020, le=2030, description="Ano letivo")
    ponto_de_virada: bool = Field(False, description="Atingiu ponto de virada")

    class Config:
        json_schema_extra = {
            "example": {
                "student_id": "EST-001",
                "inde": 6.5, "ian": 7.2, "ida": 5.8,
                "ieg": 6.0, "iaa": 7.5, "ips": 6.8,
                "ipp": 7.1, "ipv": 6.3, "fase": 3,
                "ano": 2024, "ponto_de_virada": False
            }
        }

class PredictionResponse(BaseModel):
    student_id: Optional[str]
    risco_defasagem: bool
    probabilidade: float = Field(..., ge=0.0, le=1.0)
    nivel_risco: NivelRisco
    modelo_versao: str
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    modelo_carregado: bool
    versao_api: str
    uptime_segundos: float

class MetricsResponse(BaseModel):
    total_predicoes: int
    predicoes_alto_risco: int
    predicoes_baixo_risco: int
    modelo_versao: str
    ultima_predicao: Optional[datetime]
```

---

## Tarefa 2: `app/routes.py`

```python
from fastapi import APIRouter, HTTPException, Request
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
from pathlib import Path

router = APIRouter()
logger = logging.getLogger(__name__)

MODEL_PATH = Path("app/model/model.joblib")
METADATA_PATH = Path("app/model/metadata.joblib")

# Cache do modelo em memória
_model = None
_metadata = None
_metrics_counter = {"total": 0, "alto": 0, "baixo": 0, "ultima": None}

def get_model():
    """Carrega modelo com lazy loading e cache."""
    global _model, _metadata
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Modelo não encontrado. Execute o treinamento primeiro.")
        _model = joblib.load(MODEL_PATH)
        _metadata = joblib.load(METADATA_PATH) if METADATA_PATH.exists() else {"version": "unknown"}
        logger.info(f"Modelo carregado: versão {_metadata.get('version')}")
    return _model, _metadata

@router.post("/predict", response_model=PredictionResponse, tags=["Predição"])
async def predict(request: PredictionRequest):
    """
    Recebe dados de um estudante e retorna o risco de defasagem escolar.
    """
    model, metadata = get_model()
    
    # Montar DataFrame com as mesmas features do treinamento
    input_data = pd.DataFrame([{
        'INDE': request.inde, 'IAN': request.ian, 'IDA': request.ida,
        'IEG': request.ieg, 'IAA': request.iaa, 'IPS': request.ips,
        'IPP': request.ipp, 'IPV': request.ipv, 'FASE': request.fase,
        'ANO': request.ano, 'PONTO_DE_VIRADA': int(request.ponto_de_virada),
        # Features engenheiradas
        'INDICE_BEMESTAR': (request.ips + request.ipp + request.ipv) / 3,
        'INDICE_PERFORMANCE': (request.ida + request.ieg + request.iaa) / 3,
        'GAP_AUTO_REAL': request.iaa - request.ida,
    }])
    
    proba = float(model.predict_proba(input_data)[0][1])
    pred = bool(proba >= 0.5)
    
    # Classificar nível de risco
    if proba < 0.40:
        nivel = "baixo"
    elif proba < 0.70:
        nivel = "medio"
    else:
        nivel = "alto"
    
    # Atualizar contadores
    _metrics_counter["total"] += 1
    _metrics_counter["alta" if pred else "baixo"] += 1
    _metrics_counter["ultima"] = datetime.utcnow()
    
    logger.info(f"Predição | student={request.student_id} | risco={pred} | proba={proba:.3f}")
    
    return PredictionResponse(
        student_id=request.student_id,
        risco_defasagem=pred,
        probabilidade=round(proba, 4),
        nivel_risco=nivel,
        modelo_versao=metadata.get("version", "1.0.0"),
        timestamp=datetime.utcnow()
    )

@router.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health():
    """Health check da API."""
    modelo_ok = MODEL_PATH.exists()
    return HealthResponse(
        status="healthy" if modelo_ok else "degraded",
        modelo_carregado=modelo_ok,
        versao_api="1.0.0",
        uptime_segundos=0  # implementar com time.time() na inicialização
    )

@router.get("/metrics", response_model=MetricsResponse, tags=["Sistema"])
async def metrics():
    """Retorna estatísticas de uso da API."""
    _, metadata = get_model()
    return MetricsResponse(
        total_predicoes=_metrics_counter["total"],
        predicoes_alto_risco=_metrics_counter["alto"],
        predicoes_baixo_risco=_metrics_counter["baixo"],
        modelo_versao=metadata.get("version", "1.0.0"),
        ultima_predicao=_metrics_counter["ultima"]
    )
```

---

## Tarefa 3: `app/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
import logging
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log", mode="a")
    ]
)

START_TIME = time.time()

app = FastAPI(
    title="Passos Mágicos — API de Risco de Defasagem",
    description="API para predição de risco de defasagem escolar dos estudantes da Associação Passos Mágicos",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    import os
    os.makedirs("logs", exist_ok=True)
    logging.getLogger(__name__).info("API Passos Mágicos iniciada.")
```

---

## Tarefa 4: `tests/test_api.py`

Usar `httpx.AsyncClient` com `pytest-asyncio`:

```python
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.fixture
def valid_payload():
    return {
        "inde": 6.5, "ian": 7.2, "ida": 5.8,
        "ieg": 6.0, "iaa": 7.5, "ips": 6.8,
        "ipp": 7.1, "ipv": 6.3, "fase": 3,
        "ano": 2024, "ponto_de_virada": False
    }

# Testes:
# test_health_returns_200
# test_predict_returns_200_with_valid_payload
# test_predict_probabilidade_between_0_and_1
# test_predict_nivel_risco_valid_value
# test_predict_invalid_inde_above_10  → espera 422
# test_predict_invalid_fase_negative  → espera 422
# test_metrics_returns_counts
```

---

## Exemplos de chamadas cURL (para o README)

```bash
# Health check
curl -X GET http://localhost:8000/api/v1/health

# Predição
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "EST-001",
    "inde": 4.2, "ian": 3.8, "ida": 4.5,
    "ieg": 5.0, "iaa": 6.0, "ips": 5.5,
    "ipp": 5.2, "ipv": 4.8, "fase": 2,
    "ano": 2024, "ponto_de_virada": false
  }'
```

---

## Padrões de qualidade
- Todas as rotas documentadas com `tags`, `description` e `response_model`
- Validação automática via Pydantic (nunca validar manualmente no handler)
- Tratar exceções com `HTTPException` com mensagens em PT-BR
- Logar toda predição com student_id, probabilidade e resultado
