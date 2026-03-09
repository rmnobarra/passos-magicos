"""
Schemas Pydantic da API Passos Mágicos.

Define os modelos de request e response usados nos endpoints da API,
com validação automática de tipos e valores via Pydantic v2.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class NivelRisco(str, Enum):
    """Classificação do nível de risco de defasagem escolar."""

    baixo = "baixo"  # probabilidade < 0.40
    medio = "medio"  # probabilidade 0.40–0.70
    alto = "alto"  # probabilidade > 0.70


class PredictionRequest(BaseModel):
    """Dados de entrada de um estudante para predição de risco de defasagem."""

    student_id: Optional[str] = Field(
        default=None,
        description="Identificador opcional do estudante",
    )
    inde: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Índice de Desenvolvimento Educacional (0–10)",
    )
    ian: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Indicador de Adequação de Nível (0–10)",
    )
    ida: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Indicador de Desempenho Acadêmico (0–10)",
    )
    ieg: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Indicador de Engajamento (0–10)",
    )
    iaa: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Indicador de Autoavaliação (0–10)",
    )
    ips: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Indicador Psicossocial (0–10)",
    )
    ipp: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Indicador Psicopedagógico (0–10)",
    )
    ipv: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Indicador de Ponto de Virada (0–10)",
    )
    fase: int = Field(
        ...,
        ge=0,
        le=8,
        description="Fase educacional do estudante (0–8)",
    )
    ano: int = Field(
        ...,
        ge=2020,
        le=2030,
        description="Ano letivo de referência",
    )
    ponto_de_virada: bool = Field(
        default=False,
        description="Indica se o estudante atingiu ponto de virada",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "student_id": "EST-001",
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
                "ponto_de_virada": False,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Resposta da predição de risco de defasagem escolar."""

    student_id: Optional[str] = Field(
        default=None,
        description="Identificador do estudante (espelhado do request)",
    )
    risco_defasagem: bool = Field(
        ...,
        description="True se o modelo classifica o estudante em risco",
    )
    probabilidade: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade estimada de risco (0.0–1.0)",
    )
    nivel_risco: NivelRisco = Field(
        ...,
        description="Nível de risco: baixo (<0.40), medio (0.40–0.70), alto (>0.70)",
    )
    modelo_versao: str = Field(
        ...,
        description="Versão do modelo que gerou a predição",
    )
    timestamp: datetime = Field(
        ...,
        description="Momento UTC da predição",
    )


class HealthResponse(BaseModel):
    """Resposta do health check da API."""

    status: str = Field(
        ...,
        description="Estado da API: 'healthy' ou 'degraded'",
    )
    modelo_carregado: bool = Field(
        ...,
        description="Indica se o arquivo do modelo está acessível",
    )
    versao_api: str = Field(
        ...,
        description="Versão da API",
    )
    uptime_segundos: float = Field(
        ...,
        description="Tempo em segundos desde o início da API",
    )


class MetricsResponse(BaseModel):
    """Resposta com estatísticas de uso da API."""

    total_predicoes: int = Field(
        ...,
        description="Total de chamadas ao endpoint /predict desde o início",
    )
    predicoes_alto_risco: int = Field(
        ...,
        description="Predições classificadas como alto risco",
    )
    predicoes_baixo_risco: int = Field(
        ...,
        description="Predições classificadas como baixo ou médio risco",
    )
    modelo_versao: str = Field(
        ...,
        description="Versão do modelo em uso",
    )
    ultima_predicao: Optional[datetime] = Field(
        default=None,
        description="Timestamp UTC da última predição realizada",
    )
