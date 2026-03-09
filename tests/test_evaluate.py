"""
Testes unitários para src/evaluate.py.

Cobre: cálculo de métricas, critério de confiabilidade,
geração de relatório e criação de gráficos.
"""

from pathlib import Path

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.evaluate import ModelEvaluator


# ── Fixtures locais ──────────────────────────────────────────────────────────


@pytest.fixture
def evaluator() -> ModelEvaluator:
    return ModelEvaluator()


@pytest.fixture
def metrics_bons() -> dict:
    """Métricas que atendem a todos os critérios de produção."""
    return {
        "f1_macro": 0.85,
        "f1_weighted": 0.87,
        "roc_auc": 0.90,
        "precision_macro": 0.86,
        "recall_macro": 0.84,
        "recall_positiva": 0.80,
        "confusion_matrix": [[70, 5], [8, 17]],
        "classification_report": "report",
    }


@pytest.fixture
def metrics_ruins() -> dict:
    """Métricas abaixo dos critérios mínimos."""
    return {
        "f1_macro": 0.55,
        "f1_weighted": 0.60,
        "roc_auc": 0.65,
        "precision_macro": 0.58,
        "recall_macro": 0.53,
        "recall_positiva": 0.50,
        "confusion_matrix": [[60, 20], [25, 5]],
        "classification_report": "report",
    }


# ── compute_metrics ───────────────────────────────────────────────────────────


def test_compute_metrics_retorna_todas_as_chaves(evaluator, y_arrays):
    y_true, y_pred, y_proba = y_arrays
    metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
    for chave in [
        "f1_macro",
        "f1_weighted",
        "roc_auc",
        "precision_macro",
        "recall_macro",
        "recall_positiva",
        "confusion_matrix",
        "classification_report",
    ]:
        assert chave in metrics, f"Chave '{chave}' ausente em compute_metrics"


def test_compute_metrics_f1_entre_0_e_1(evaluator, y_arrays):
    y_true, y_pred, y_proba = y_arrays
    metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_compute_metrics_roc_auc_entre_0_e_1(evaluator, y_arrays):
    y_true, y_pred, y_proba = y_arrays
    metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_compute_metrics_confusion_matrix_e_lista(evaluator, y_arrays):
    y_true, y_pred, y_proba = y_arrays
    metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
    cm = metrics["confusion_matrix"]
    assert isinstance(cm, list)
    assert len(cm) == 2
    assert len(cm[0]) == 2


def test_compute_metrics_predicao_perfeita_f1_1(evaluator):
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.05, 0.1, 0.15, 0.85, 0.9, 0.95])
    metrics = evaluator.compute_metrics(y, y, proba)
    assert metrics["f1_macro"] == pytest.approx(1.0)


def test_compute_metrics_recall_positiva_entre_0_e_1(evaluator, y_arrays):
    y_true, y_pred, y_proba = y_arrays
    metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
    assert 0.0 <= metrics["recall_positiva"] <= 1.0


# ── is_model_reliable ─────────────────────────────────────────────────────────


def test_is_model_reliable_retorna_true_com_metricas_boas(evaluator, metrics_bons):
    assert evaluator.is_model_reliable(metrics_bons) is True


def test_is_model_reliable_retorna_false_com_metricas_ruins(evaluator, metrics_ruins):
    assert evaluator.is_model_reliable(metrics_ruins) is False


@pytest.mark.parametrize(
    "campo,valor_ruim",
    [
        ("f1_macro", 0.69),
        ("roc_auc", 0.74),
        ("recall_positiva", 0.64),
    ],
)
def test_is_model_reliable_falha_por_criterio(
    evaluator, metrics_bons, campo, valor_ruim
):
    """Qualquer critério abaixo do mínimo deve reprovar o modelo."""
    metrics = metrics_bons.copy()
    metrics[campo] = valor_ruim
    assert evaluator.is_model_reliable(metrics) is False


def test_is_model_reliable_limiar_exato_aprovado(evaluator, metrics_bons):
    """Valores exatamente nos limiares devem ser aprovados."""
    metrics = metrics_bons.copy()
    metrics["f1_macro"] = 0.70
    metrics["roc_auc"] = 0.75
    metrics["recall_positiva"] = 0.65
    assert evaluator.is_model_reliable(metrics) is True


# ── generate_report ───────────────────────────────────────────────────────────


def test_generate_report_cria_arquivo(evaluator, metrics_bons, tmp_path):
    path = str(tmp_path / "relatorio.txt")
    evaluator.generate_report(metrics_bons, path)
    assert Path(path).exists()


def test_generate_report_retorna_caminho(evaluator, metrics_bons, tmp_path):
    path = str(tmp_path / "relatorio.txt")
    retorno = evaluator.generate_report(metrics_bons, path)
    assert retorno == path


def test_generate_report_contem_f1_macro(evaluator, metrics_bons, tmp_path):
    path = str(tmp_path / "relatorio.txt")
    evaluator.generate_report(metrics_bons, path)
    conteudo = Path(path).read_text(encoding="utf-8")
    assert "F1 macro" in conteudo


def test_generate_report_contem_roc_auc(evaluator, metrics_bons, tmp_path):
    path = str(tmp_path / "relatorio.txt")
    evaluator.generate_report(metrics_bons, path)
    conteudo = Path(path).read_text(encoding="utf-8")
    assert "ROC-AUC" in conteudo


def test_generate_report_contem_status_producao(evaluator, metrics_bons, tmp_path):
    path = str(tmp_path / "relatorio.txt")
    evaluator.generate_report(metrics_bons, path)
    conteudo = Path(path).read_text(encoding="utf-8")
    assert "OK" in conteudo or "REPROVADO" in conteudo


# ── plot_confusion_matrix ──────────────────────────────────────────────────────


def test_plot_confusion_matrix_cria_arquivo_png(evaluator, y_arrays, tmp_path):
    y_true, y_pred, _ = y_arrays
    path = str(tmp_path / "cm.png")
    evaluator.plot_confusion_matrix(y_true, y_pred, path)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


# ── plot_roc_curve ────────────────────────────────────────────────────────────


def test_plot_roc_curve_cria_arquivo_png(evaluator, y_arrays, tmp_path):
    y_true, _, y_proba = y_arrays
    path = str(tmp_path / "roc.png")
    evaluator.plot_roc_curve(y_true, y_proba, path)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


# ── plot_feature_importance ───────────────────────────────────────────────────


def test_plot_feature_importance_cria_arquivo_png(evaluator, tmp_path):
    model = MagicMock()
    model.feature_importances_ = np.array([0.3, 0.5, 0.2])
    features = ["INDE", "IAN", "IDA"]
    path = str(tmp_path / "fi.png")
    evaluator.plot_feature_importance(model, features, path)
    assert Path(path).exists()


def test_plot_feature_importance_sem_atributo_nao_levanta_erro(evaluator, tmp_path):
    """Modelo sem feature_importances_ deve ser ignorado silenciosamente."""
    model = MagicMock(spec=[])  # sem feature_importances_
    path = str(tmp_path / "fi_nao_existe.png")
    evaluator.plot_feature_importance(model, ["INDE"], path)
    assert not Path(path).exists()
