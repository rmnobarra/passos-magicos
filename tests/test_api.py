"""
Testes de integração para a API FastAPI — app/routes.py.

Usa httpx.AsyncClient com ASGITransport para testar os endpoints
/predict, /health e /metrics sem depender de servidor HTTP real.
O modelo é mockado para isolamento total dos testes da API.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

# ── Fixtures ─────────────────────────────────────────────────────────────────

FEATURES_MODELO = [
    "INDE",
    "IAN",
    "IDA",
    "IEG",
    "IAA",
    "IPS",
    "IPP",
    "IPV",
    "INDICE_BEMESTAR",
    "INDICE_PERFORMANCE",
    "GAP_AUTO_REAL",
    "ABAIXO_MEDIA_GERAL",
    "EVOLUCAO_INDE",
    "INDE_x_FASE",
    "BEMESTAR_x_PERFORMANCE",
    "ANO",
    "FASE",
    "PONTO_DE_VIRADA",
]

META_MOCK = {
    "version": "1.0.0",
    "features": FEATURES_MODELO,
    "algorithm": "MockClassifier",
}


@pytest.fixture
def mock_model_alto_risco():
    """Mock que retorna probabilidade de 0.85 (alto risco)."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.15, 0.85]])
    return model


@pytest.fixture
def mock_model_baixo_risco():
    """Mock que retorna probabilidade de 0.10 (baixo risco)."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.90, 0.10]])
    return model


class FakeRedis:
    """Redis em memória para isolamento dos testes."""

    def __init__(self):
        self._store: dict = {}

    def ping(self) -> bool:
        return True

    def get(self, key: str):
        v = self._store.get(key)
        return str(v) if v is not None else None

    def set(self, key: str, value) -> None:
        self._store[key] = value

    def incr(self, key: str) -> int:
        self._store[key] = int(self._store.get(key, 0)) + 1
        return self._store[key]

    def pipeline(self):
        return self  # pipeline auto-executado

    def execute(self):
        return []


@pytest.fixture(autouse=True)
def reset_redis(monkeypatch):
    """Injeta FakeRedis limpo para cada teste e reseta ao final."""
    import app.routes as routes

    fake = FakeRedis()
    monkeypatch.setattr(routes, "_redis", fake)
    yield
    monkeypatch.setattr(routes, "_redis", None)


@pytest.fixture
async def client():
    """Cliente HTTP assíncrono conectado à aplicação de teste."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────


async def test_health_retorna_200(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200


async def test_health_campos_obrigatorios(client):
    response = await client.get("/api/v1/health")
    body = response.json()
    for campo in ["status", "modelo_carregado", "versao_api", "uptime_segundos"]:
        assert campo in body, f"Campo '{campo}' ausente na resposta de /health"


async def test_health_versao_api_correta(client):
    response = await client.get("/api/v1/health")
    assert response.json()["versao_api"] == "1.0.0"


async def test_health_uptime_e_numero(client):
    response = await client.get("/api/v1/health")
    uptime = response.json()["uptime_segundos"]
    assert isinstance(uptime, (int, float))
    assert uptime >= 0


# ── /predict ──────────────────────────────────────────────────────────────────


async def test_predict_retorna_200_com_payload_valido(
    client, valid_api_payload, mock_model_alto_risco
):
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        response = await client.post("/api/v1/predict", json=valid_api_payload)
    assert response.status_code == 200


async def test_predict_probabilidade_entre_0_e_1(
    client, valid_api_payload, mock_model_alto_risco
):
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        response = await client.post("/api/v1/predict", json=valid_api_payload)
    prob = response.json()["probabilidade"]
    assert 0.0 <= prob <= 1.0


async def test_predict_risco_defasagem_e_booleano(
    client, valid_api_payload, mock_model_alto_risco
):
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        response = await client.post("/api/v1/predict", json=valid_api_payload)
    assert isinstance(response.json()["risco_defasagem"], bool)


async def test_predict_nivel_risco_valor_valido(
    client, valid_api_payload, mock_model_alto_risco
):
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        response = await client.post("/api/v1/predict", json=valid_api_payload)
    nivel = response.json()["nivel_risco"]
    assert nivel in {"baixo", "medio", "alto"}


async def test_predict_alto_risco_classifica_nivel_alto(
    client, high_risk_payload, mock_model_alto_risco
):
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        response = await client.post("/api/v1/predict", json=high_risk_payload)
    assert response.json()["nivel_risco"] == "alto"
    assert response.json()["risco_defasagem"] is True


async def test_predict_baixo_risco_classifica_nivel_baixo(
    client, valid_api_payload, mock_model_baixo_risco
):
    with patch(
        "app.routes.get_model", return_value=(mock_model_baixo_risco, META_MOCK)
    ):
        response = await client.post("/api/v1/predict", json=valid_api_payload)
    assert response.json()["nivel_risco"] == "baixo"
    assert response.json()["risco_defasagem"] is False


async def test_predict_student_id_espelhado(
    client, valid_api_payload, mock_model_alto_risco
):
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        response = await client.post("/api/v1/predict", json=valid_api_payload)
    assert response.json()["student_id"] == valid_api_payload["student_id"]


async def test_predict_modelo_versao_presente(
    client, valid_api_payload, mock_model_alto_risco
):
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        response = await client.post("/api/v1/predict", json=valid_api_payload)
    assert "modelo_versao" in response.json()


async def test_predict_timestamp_presente(
    client, valid_api_payload, mock_model_alto_risco
):
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        response = await client.post("/api/v1/predict", json=valid_api_payload)
    assert "timestamp" in response.json()


# ── Validação de input (/predict → 422) ──────────────────────────────────────


@pytest.mark.parametrize(
    "campo,valor_invalido",
    [
        ("inde", 15.0),  # acima do máximo (10)
        ("inde", -1.0),  # abaixo do mínimo (0)
        ("fase", -1),  # fase negativa
        ("fase", 9),  # fase acima do máximo (8)
        ("ano", 2019),  # ano abaixo do mínimo (2020)
        ("ian", 11.0),  # ian acima do máximo
    ],
)
async def test_predict_campo_invalido_retorna_422(
    client, valid_api_payload, campo, valor_invalido
):
    payload = {**valid_api_payload, campo: valor_invalido}
    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422, (
        f"Campo '{campo}'={valor_invalido} deveria retornar 422, "
        f"mas retornou {response.status_code}"
    )


async def test_predict_payload_vazio_retorna_422(client):
    response = await client.post("/api/v1/predict", json={})
    assert response.status_code == 422


async def test_predict_sem_campo_obrigatorio_retorna_422(client, valid_api_payload):
    payload = {k: v for k, v in valid_api_payload.items() if k != "inde"}
    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


# ── /metrics ──────────────────────────────────────────────────────────────────


async def test_metrics_retorna_200(client, mock_model_alto_risco):
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        response = await client.get("/api/v1/metrics")
    assert response.status_code == 200


async def test_metrics_campos_obrigatorios(client, mock_model_alto_risco):
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        response = await client.get("/api/v1/metrics")
    body = response.json()
    for campo in [
        "total_predicoes",
        "predicoes_alto_risco",
        "predicoes_baixo_risco",
        "modelo_versao",
    ]:
        assert campo in body


async def test_metrics_total_predicoes_incrementa(
    client, valid_api_payload, mock_model_alto_risco
):
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        await client.post("/api/v1/predict", json=valid_api_payload)
        await client.post("/api/v1/predict", json=valid_api_payload)
        response = await client.get("/api/v1/metrics")
    assert response.json()["total_predicoes"] == 2


async def test_metrics_conta_alto_risco_corretamente(
    client, valid_api_payload, mock_model_alto_risco
):
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        await client.post("/api/v1/predict", json=valid_api_payload)
        response = await client.get("/api/v1/metrics")
    assert response.json()["predicoes_alto_risco"] == 1


async def test_metrics_ultima_predicao_e_null_sem_predicoes(
    client, mock_model_alto_risco
):
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        response = await client.get("/api/v1/metrics")
    assert response.json()["ultima_predicao"] is None


async def test_metrics_ultima_predicao_preenchida_apos_predict(
    client, valid_api_payload, mock_model_alto_risco
):
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        await client.post("/api/v1/predict", json=valid_api_payload)
        response = await client.get("/api/v1/metrics")
    assert response.json()["ultima_predicao"] is not None


# ── get_redis ─────────────────────────────────────────────────────────────────


def test_get_redis_retorna_cliente_quando_disponivel(monkeypatch):
    import app.routes as routes

    monkeypatch.setattr(routes, "_redis", None)
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    with patch("app.routes.redis_lib.from_url", return_value=mock_client):
        r = routes.get_redis()
    assert r is mock_client


def test_get_redis_retorna_none_quando_falha(monkeypatch):
    import app.routes as routes

    monkeypatch.setattr(routes, "_redis", None)
    with patch("app.routes.redis_lib.from_url", side_effect=Exception("conn refused")):
        r = routes.get_redis()
    assert r is None


# ── get_model ─────────────────────────────────────────────────────────────────


async def test_predict_sem_modelo_retorna_503(client, monkeypatch, valid_api_payload):
    import app.routes as routes
    from pathlib import Path

    monkeypatch.setattr(routes, "_model", None)
    monkeypatch.setattr(routes, "_metadata", None)
    monkeypatch.setattr(routes, "MODEL_PATH", Path("/nao/existe/model.joblib"))
    response = await client.post("/api/v1/predict", json=valid_api_payload)
    assert response.status_code == 503


def test_get_model_carrega_de_disco(monkeypatch, tmp_path):
    import joblib
    import app.routes as routes
    from pathlib import Path
    from sklearn.dummy import DummyClassifier

    model_path = tmp_path / "model.joblib"
    joblib.dump(DummyClassifier(), str(model_path))
    monkeypatch.setattr(routes, "_model", None)
    monkeypatch.setattr(routes, "_metadata", None)
    monkeypatch.setattr(routes, "MODEL_PATH", model_path)
    monkeypatch.setattr(routes, "METADATA_PATH", tmp_path / "nao_existe.joblib")
    model, metadata = routes.get_model()
    assert model is not None
    assert metadata["version"] == "desconhecida"


# ── _classificar_nivel (medio) ─────────────────────────────────────────────────


async def test_predict_nivel_medio(client, valid_api_payload):
    mock_medio = MagicMock()
    mock_medio.predict_proba.return_value = np.array([[0.45, 0.55]])
    with patch("app.routes.get_model", return_value=(mock_medio, META_MOCK)):
        response = await client.post("/api/v1/predict", json=valid_api_payload)
    assert response.json()["nivel_risco"] == "medio"


# ── _ler_metricas_do_log ──────────────────────────────────────────────────────


def test_ler_metricas_do_log_conta_corretamente(monkeypatch, tmp_path):
    import app.routes as routes

    jsonl = tmp_path / "predictions.jsonl"
    jsonl.write_text(
        '{"timestamp": "2024-01-01T00:00:00+00:00", "nivel_risco": "alto"}\n'
        '{"timestamp": "2024-01-02T00:00:00+00:00", "nivel_risco": "baixo"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(routes, "PREDICTIONS_LOG", jsonl)
    result = routes._ler_metricas_do_log()
    assert result["total"] == 2
    assert result["alto"] == 1
    assert result["baixo_medio"] == 1
    assert result["ultima"] is not None


def test_ler_metricas_do_log_sem_arquivo_retorna_zeros(monkeypatch, tmp_path):
    import app.routes as routes

    monkeypatch.setattr(routes, "PREDICTIONS_LOG", tmp_path / "nao_existe.jsonl")
    result = routes._ler_metricas_do_log()
    assert result["total"] == 0
    assert result["ultima"] is None


# ── Erros e fallbacks ─────────────────────────────────────────────────────────


async def test_predict_erro_interno_retorna_500(client, valid_api_payload):
    mock_erro = MagicMock()
    mock_erro.predict_proba.side_effect = ValueError("modelo quebrado")
    with patch("app.routes.get_model", return_value=(mock_erro, META_MOCK)):
        response = await client.post("/api/v1/predict", json=valid_api_payload)
    assert response.status_code == 500


async def test_predict_redis_pipeline_erro_nao_quebra_resposta(
    client, valid_api_payload, monkeypatch, mock_model_alto_risco
):
    import app.routes as routes

    bad_redis = MagicMock()
    bad_redis.pipeline.return_value.execute.side_effect = Exception("pipeline falhou")
    monkeypatch.setattr(routes, "_redis", bad_redis)
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        response = await client.post("/api/v1/predict", json=valid_api_payload)
    assert response.status_code == 200


async def test_metrics_fallback_jsonl_quando_redis_indisponivel(
    client, monkeypatch, tmp_path, mock_model_alto_risco
):
    import app.routes as routes

    jsonl = tmp_path / "predictions.jsonl"
    jsonl.write_text(
        '{"timestamp": "2024-01-01T00:00:00+00:00", "nivel_risco": "alto"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(routes, "_redis", None)
    monkeypatch.setattr(routes, "PREDICTIONS_LOG", jsonl)
    with patch("app.routes.redis_lib.from_url", side_effect=Exception("sem redis")):
        with patch(
            "app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)
        ):
            response = await client.get("/api/v1/metrics")
    assert response.status_code == 200
    assert response.json()["total_predicoes"] == 1


async def test_metrics_fallback_quando_redis_falha_na_leitura(
    client, monkeypatch, tmp_path, mock_model_alto_risco
):
    import app.routes as routes

    bad_redis = MagicMock()
    bad_redis.get.side_effect = Exception("Redis timeout")
    monkeypatch.setattr(routes, "_redis", bad_redis)
    jsonl = tmp_path / "predictions.jsonl"
    jsonl.write_text("", encoding="utf-8")
    monkeypatch.setattr(routes, "PREDICTIONS_LOG", jsonl)
    with patch("app.routes.get_model", return_value=(mock_model_alto_risco, META_MOCK)):
        response = await client.get("/api/v1/metrics")
    assert response.status_code == 200


# ── Swagger / OpenAPI ─────────────────────────────────────────────────────────


async def test_docs_swagger_disponivel(client):
    response = await client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()


async def test_openapi_json_disponivel(client):
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "/api/v1/predict" in schema["paths"]
    assert "/api/v1/health" in schema["paths"]
    assert "/api/v1/metrics" in schema["paths"]
