"""
Testes unitários para src/train.py.

Cobre: construção do pipeline, treinamento, validação cruzada,
serialização do modelo e filtragem de features.
"""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.train import ModelTrainer


# ── Fixtures locais ──────────────────────────────────────────────────────────


@pytest.fixture
def trainer() -> ModelTrainer:
    return ModelTrainer(config={})


# ── _filter_features ─────────────────────────────────────────────────────────


def test_filter_features_identifica_numericas(trainer, trainer_with_data):
    _, X, _ = trainer_with_data
    trainer._filter_features(X)
    assert len(trainer.numerical_features) > 0


def test_filter_features_identifica_categoricas(trainer, trainer_with_data):
    _, X, _ = trainer_with_data
    trainer._filter_features(X)
    for col in trainer.categorical_features:
        assert col in X.columns


def test_filter_features_feature_names_e_uniao(trainer, trainer_with_data):
    _, X, _ = trainer_with_data
    trainer._filter_features(X)
    esperado = set(trainer.numerical_features) | set(trainer.categorical_features)
    assert set(trainer.feature_names) == esperado


def test_filter_features_ignora_colunas_ausentes(trainer):
    X = pd.DataFrame({"INDE": [5.0], "IAN": [6.0]})
    trainer._filter_features(X)
    assert "INDE" in trainer.numerical_features
    # FASE não está em X, não deve entrar em categorical_features
    assert "FASE" not in trainer.categorical_features


# ── build_pipeline ────────────────────────────────────────────────────────────


def test_build_pipeline_retorna_pipeline(trainer, trainer_with_data):
    _, X, _ = trainer_with_data
    trainer._filter_features(X)
    pipeline = trainer.build_pipeline()
    assert isinstance(pipeline, Pipeline)


def test_build_pipeline_tem_preprocessor(trainer, trainer_with_data):
    _, X, _ = trainer_with_data
    trainer._filter_features(X)
    pipeline = trainer.build_pipeline()
    assert "preprocessor" in pipeline.named_steps


def test_build_pipeline_tem_classifier(trainer, trainer_with_data):
    _, X, _ = trainer_with_data
    trainer._filter_features(X)
    pipeline = trainer.build_pipeline()
    assert "classifier" in pipeline.named_steps


def test_build_pipeline_aceita_classificador_customizado(trainer, trainer_with_data):
    from sklearn.ensemble import RandomForestClassifier

    _, X, _ = trainer_with_data
    trainer._filter_features(X)
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    pipeline = trainer.build_pipeline(classifier=clf)
    assert pipeline.named_steps["classifier"] is clf


# ── train ─────────────────────────────────────────────────────────────────────


def test_train_retorna_pipeline_treinado(trainer, trainer_with_data):
    _, X, y = trainer_with_data
    pipeline = trainer.train(X, y)
    assert isinstance(pipeline, Pipeline)


def test_train_pipeline_prediz_classes_binarias(trainer, trainer_with_data):
    _, X, y = trainer_with_data
    pipeline = trainer.train(X, y)
    preds = pipeline.predict(X[trainer.feature_names])
    assert set(preds).issubset({0, 1})


def test_train_pipeline_prediz_probabilidades_validas(trainer, trainer_with_data):
    _, X, y = trainer_with_data
    pipeline = trainer.train(X, y)
    probas = pipeline.predict_proba(X[trainer.feature_names])
    assert probas.shape[1] == 2
    assert (probas >= 0).all() and (probas <= 1).all()


def test_train_seta_feature_names(trainer, trainer_with_data):
    _, X, y = trainer_with_data
    trainer.train(X, y)
    assert len(trainer.feature_names) > 0


def test_train_numero_predicoes_igual_amostras(trainer, trainer_with_data):
    _, X, y = trainer_with_data
    pipeline = trainer.train(X, y)
    preds = pipeline.predict(X[trainer.feature_names])
    assert len(preds) == len(X)


# ── cross_validate ────────────────────────────────────────────────────────────


def test_cross_validate_retorna_chaves_esperadas(trainer, trainer_with_data):
    _, X, y = trainer_with_data
    trainer._filter_features(X)
    scores = trainer.cross_validate(X, y)
    for chave in ["f1_macro_mean", "roc_auc_mean", "recall_macro_mean"]:
        assert chave in scores, f"Chave '{chave}' ausente no resultado de CV"


def test_cross_validate_f1_entre_0_e_1(trainer, trainer_with_data):
    _, X, y = trainer_with_data
    trainer._filter_features(X)
    scores = trainer.cross_validate(X, y)
    assert 0.0 <= scores["f1_macro_mean"] <= 1.0


def test_cross_validate_roc_auc_entre_0_e_1(trainer, trainer_with_data):
    _, X, y = trainer_with_data
    trainer._filter_features(X)
    scores = trainer.cross_validate(X, y)
    assert 0.0 <= scores["roc_auc_mean"] <= 1.0


# ── save_model ────────────────────────────────────────────────────────────────


def test_save_model_cria_arquivo_joblib(trainer, trained_pipeline, tmp_path):
    path = str(tmp_path / "modelo_teste.joblib")
    trainer.save_model(trained_pipeline, path)
    assert os.path.exists(path)


def test_save_model_com_metadata_cria_dois_arquivos(
    trainer, trained_pipeline, tmp_path
):
    model_path = str(tmp_path / "model.joblib")
    meta_path = str(tmp_path / "meta.joblib")
    trainer.metadata_path = meta_path
    trainer.save_model(trained_pipeline, model_path, metadata={"version": "test"})
    assert os.path.exists(model_path)
    assert os.path.exists(meta_path)


def test_save_model_artefato_carregavel(trainer, trained_pipeline, tmp_path):
    import joblib

    path = str(tmp_path / "model.joblib")
    trainer.save_model(trained_pipeline, path)
    loaded = joblib.load(path)
    assert isinstance(loaded, Pipeline)


def test_save_model_sem_metadata_nao_cria_meta(trainer, trained_pipeline, tmp_path):
    model_path = str(tmp_path / "model.joblib")
    meta_path = str(tmp_path / "nao_deve_existir.joblib")
    trainer.metadata_path = meta_path
    trainer.save_model(trained_pipeline, model_path, metadata=None)
    assert os.path.exists(model_path)
    assert not os.path.exists(meta_path)
