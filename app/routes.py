"""
Rotas da API Passos Mágicos.

Define os endpoints /predict, /health e /metrics com documentação
automática via FastAPI/Swagger. O modelo é carregado com lazy loading
e mantido em cache em memória durante o ciclo de vida da API.

Design stateless: nenhum estado de negócio é mantido em memória entre
requisições. As métricas são derivadas do arquivo predictions.jsonl,
que deve residir em um PersistentVolume compartilhado em ambientes
Kubernetes com múltiplas réplicas.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import redis as redis_lib
from fastapi import APIRouter, HTTPException, Request

from app.schemas import (
    HealthResponse,
    MetricsResponse,
    NivelRisco,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)

MODEL_PATH = Path("app/model/model.joblib")
METADATA_PATH = Path("app/model/metadata.joblib")
PREDICTIONS_LOG = Path("logs/predictions.jsonl")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

# Prefixo de chaves Redis
_RK_TOTAL = "pm:total"
_RK_ALTO = "pm:alto"
_RK_BAIXO_MEDIO = "pm:baixo_medio"
_RK_ULTIMA = "pm:ultima"

# Cache do modelo em memória (somente leitura — seguro para múltiplas réplicas)
_model = None
_metadata: Optional[dict] = None
_redis: Optional[redis_lib.Redis] = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def get_redis() -> Optional[redis_lib.Redis]:
    """
    Retorna cliente Redis com conexão lazy e cache em módulo.

    Tenta conectar uma única vez; se falhar, retorna None e o sistema
    opera em modo degradado usando o JSONL como fallback.

    Returns:
        Cliente Redis conectado ou None se indisponível.
    """
    global _redis
    if _redis is None:
        try:
            client = redis_lib.from_url(REDIS_URL, decode_responses=True)
            client.ping()
            _redis = client
            logger.info(f"Redis conectado em {REDIS_URL}")
        except Exception as exc:
            logger.warning(f"Redis indisponível — operando sem cache: {exc}")
    return _redis


def get_model():
    """
    Carrega o modelo e os metadados com lazy loading e cache em memória.

    Na primeira chamada, desserializa os arquivos .joblib. Nas chamadas
    subsequentes, retorna os objetos já carregados.

    Returns:
        Tupla (pipeline_sklearn, dict_metadata).

    Raises:
        HTTPException 503: Se o arquivo do modelo não existir.
    """
    global _model, _metadata

    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Modelo não encontrado. "
                    "Execute 'python src/train.py --data <csv>' primeiro."
                ),
            )
        _model = joblib.load(MODEL_PATH)
        _metadata = (
            joblib.load(METADATA_PATH)
            if METADATA_PATH.exists()
            else {"version": "desconhecida", "features": []}
        )
        logger.info(
            f"Modelo carregado — versão: {_metadata.get('version')} | "
            f"algoritmo: {_metadata.get('algorithm', 'N/A')}"
        )

    return _model, _metadata


def _classificar_nivel(probabilidade: float) -> NivelRisco:
    """
    Classifica o nível de risco com base na probabilidade predita.

    Faixas:
    - baixo : probabilidade < 0.40
    - medio : 0.40 <= probabilidade <= 0.70
    - alto  : probabilidade > 0.70

    Args:
        probabilidade: Valor entre 0.0 e 1.0 retornado pelo modelo.

    Returns:
        Membro do enum NivelRisco correspondente à faixa.
    """
    if probabilidade < 0.40:
        return NivelRisco.baixo
    if probabilidade <= 0.70:
        return NivelRisco.medio
    return NivelRisco.alto


def _build_input_df(request: PredictionRequest, metadata: dict) -> pd.DataFrame:
    """
    Monta o DataFrame de entrada com as features que o modelo espera.

    Computa todas as features derivadas (compostas, de interação) usando
    os valores do request. Features que dependem de dados históricos ou
    populacionais (EVOLUCAO_INDE, ABAIXO_MEDIA_GERAL) recebem valor 0,
    que é o valor neutro/default definido durante o treinamento.

    A lista final de colunas é determinada pela chave 'features' dos
    metadados do modelo, garantindo compatibilidade com o pipeline treinado.

    Args:
        request: Dados do estudante recebidos na requisição.
        metadata: Metadados do modelo carregado (contém lista de features).

    Returns:
        DataFrame de uma linha com as colunas na ordem correta.
    """
    inde = request.inde
    ian = request.ian
    ida = request.ida
    ieg = request.ieg
    iaa = request.iaa
    ips = request.ips
    ipp = request.ipp
    ipv = request.ipv
    fase = float(request.fase)  # OHE foi treinado com float
    ano = request.ano

    indice_bemestar = (ips + ipp + ipv) / 3
    indice_performance = (ida + ieg + iaa) / 3
    gap_auto_real = iaa - ida

    # PONTO_DE_VIRADA: OHE foi treinado com strings 'Sim'/'Não'
    ponto_de_virada = "Sim" if request.ponto_de_virada else "Não"

    todas_features = {
        "INDE": inde,
        "IAN": ian,
        "IDA": ida,
        "IEG": ieg,
        "IAA": iaa,
        "IPS": ips,
        "IPP": ipp,
        "IPV": ipv,
        "INDICE_BEMESTAR": indice_bemestar,
        "INDICE_PERFORMANCE": indice_performance,
        "GAP_AUTO_REAL": gap_auto_real,
        # Não disponível em inferência (requer todos os alunos) → default 0
        "ABAIXO_MEDIA_GERAL": 0,
        # Não disponível em inferência (requer histórico) → default 0
        "EVOLUCAO_INDE": 0.0,
        "INDE_x_FASE": inde * fase,
        "BEMESTAR_x_PERFORMANCE": indice_bemestar * indice_performance,
        "ANO": ano,
        "FASE": fase,
        "PONTO_DE_VIRADA": ponto_de_virada,
    }

    # Usar apenas as features na ordem que o modelo espera
    features_modelo = metadata.get("features", list(todas_features.keys()))
    linha = {k: todas_features[k] for k in features_modelo if k in todas_features}

    return pd.DataFrame([linha])


def _ler_metricas_do_log() -> dict:
    """
    Deriva as métricas de uso lendo o histórico persistido em predictions.jsonl.

    Não mantém estado em memória — cada chamada relê o arquivo, garantindo
    consistência em ambientes com múltiplas réplicas (stateless).

    Em Kubernetes, o arquivo deve residir em um PersistentVolume
    com accessMode ReadWriteMany para que todas as réplicas o compartilhem.

    Returns:
        dict com total, alto, baixo_medio e ultima_predicao.
    """
    total = alto = baixo_medio = 0
    ultima = None
    if PREDICTIONS_LOG.exists():
        try:
            with PREDICTIONS_LOG.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    registro = json.loads(line)
                    total += 1
                    if registro.get("nivel_risco") == "alto":
                        alto += 1
                    else:
                        baixo_medio += 1
                    ts = registro.get("timestamp")
                    if ts and (ultima is None or ts > ultima):
                        ultima = ts
        except Exception as exc:
            logger.warning(f"Falha ao ler métricas do log: {exc}")
    return {
        "total": total,
        "alto": alto,
        "baixo_medio": baixo_medio,
        "ultima": datetime.fromisoformat(ultima) if ultima else None,
    }


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predição"],
    summary="Prediz risco de defasagem escolar",
    responses={
        200: {"description": "Predição realizada com sucesso"},
        422: {"description": "Dados de entrada inválidos"},
        503: {"description": "Modelo não disponível"},
    },
)
async def predict(payload: PredictionRequest) -> PredictionResponse:
    """
    Recebe indicadores educacionais de um estudante e retorna a predição
    de risco de defasagem escolar.

    O modelo classifica o estudante em **sem risco** (0) ou **com risco** (1)
    e fornece a probabilidade associada e o nível de risco (baixo/medio/alto).

    **Regra de negócio:** falsos negativos têm custo social alto — um aluno
    em risco não identificado pode não receber intervenção a tempo. Por isso
    o modelo prioriza alto Recall para a classe positiva.
    """
    model, metadata = get_model()

    try:
        input_df = _build_input_df(payload, metadata)
        proba = float(model.predict_proba(input_df)[0][1])
    except Exception as exc:
        logger.error(f"Erro na predição | student={payload.student_id} | {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno ao processar a predição: {exc}",
        )

    risco = proba >= 0.5
    nivel = _classificar_nivel(proba)
    agora = datetime.now(timezone.utc)

    # Persistir predição em JSONL (tabela do dashboard)
    registro = {
        "timestamp": agora.isoformat(),
        "student_id": payload.student_id,
        "risco_defasagem": int(risco),
        "probabilidade": round(proba, 4),
        "nivel_risco": nivel.value,
    }
    try:
        PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PREDICTIONS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(registro) + "\n")
    except Exception as exc:
        logger.warning(f"Falha ao gravar JSONL: {exc}")

    # Incrementar contadores no Redis (atômico, compartilhado entre réplicas)
    r = get_redis()
    if r:
        try:
            pipe = r.pipeline()
            pipe.incr(_RK_TOTAL)
            pipe.incr(_RK_ALTO if nivel == NivelRisco.alto else _RK_BAIXO_MEDIO)
            pipe.set(_RK_ULTIMA, agora.isoformat())
            pipe.execute()
        except Exception as exc:
            logger.warning(f"Falha ao atualizar Redis: {exc}")

    logger.info(
        f"Predição | student={payload.student_id} | "
        f"risco={risco} | proba={proba:.4f} | nivel={nivel.value}"
    )

    return PredictionResponse(
        student_id=payload.student_id,
        risco_defasagem=risco,
        probabilidade=round(proba, 4),
        nivel_risco=nivel,
        modelo_versao=metadata.get("version", "1.0.0"),
        timestamp=agora,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Sistema"],
    summary="Health check da API",
    responses={
        200: {"description": "Estado atual da API"},
    },
)
async def health(request: Request) -> HealthResponse:
    """
    Verifica o estado da API e a disponibilidade do modelo.

    Retorna **healthy** quando o arquivo do modelo está acessível,
    ou **degraded** quando o modelo não foi encontrado em disco.
    """
    import time

    modelo_disponivel = MODEL_PATH.exists()
    start_time = getattr(request.app.state, "start_time", time.time())

    return HealthResponse(
        status="healthy" if modelo_disponivel else "degraded",
        modelo_carregado=modelo_disponivel,
        versao_api="1.0.0",
        uptime_segundos=round(time.time() - start_time, 2),
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Sistema"],
    summary="Métricas de uso da API",
    responses={
        200: {"description": "Estatísticas de predições realizadas"},
        503: {"description": "Modelo não disponível"},
    },
)
async def metrics() -> MetricsResponse:
    """
    Retorna estatísticas acumuladas de uso do endpoint /predict.

    As métricas são derivadas diretamente do arquivo predictions.jsonl,
    sem estado em memória — consistente entre reinicializações e réplicas.
    """
    _, metadata = get_model()

    r = get_redis()
    if r:
        try:
            total = int(r.get(_RK_TOTAL) or 0)
            alto = int(r.get(_RK_ALTO) or 0)
            baixo_medio = int(r.get(_RK_BAIXO_MEDIO) or 0)
            ultima_str = r.get(_RK_ULTIMA)
            ultima = datetime.fromisoformat(ultima_str) if ultima_str else None
        except Exception as exc:
            logger.warning(f"Falha ao ler Redis, usando fallback JSONL: {exc}")
            r = None

    if not r:
        contadores = _ler_metricas_do_log()
        total = contadores["total"]
        alto = contadores["alto"]
        baixo_medio = contadores["baixo_medio"]
        ultima = contadores["ultima"]

    return MetricsResponse(
        total_predicoes=total,
        predicoes_alto_risco=alto,
        predicoes_baixo_risco=baixo_medio,
        modelo_versao=metadata.get("version", "1.0.0"),
        ultima_predicao=ultima,
    )
