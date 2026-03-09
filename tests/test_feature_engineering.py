"""
Testes unitários para src/feature_engineering.py.

Cobre: features compostas, temporais, de interação,
seleção de features e o pipeline completo (transform).
"""

import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineer


# ── Fixtures locais ──────────────────────────────────────────────────────────


@pytest.fixture
def fe() -> FeatureEngineer:
    return FeatureEngineer()


@pytest.fixture
def df_preprocessed(sample_dataframe) -> pd.DataFrame:
    """sample_dataframe já processado pelo DataPreprocessor."""
    from src.preprocessing import DataPreprocessor

    dp = DataPreprocessor()
    return dp.fit_transform(sample_dataframe.copy())


# ── create_composite_features ────────────────────────────────────────────────


def test_create_composite_creates_indice_bemestar(fe, df_preprocessed):
    result = fe.create_composite_features(df_preprocessed)
    assert "INDICE_BEMESTAR" in result.columns


def test_create_composite_creates_indice_performance(fe, df_preprocessed):
    result = fe.create_composite_features(df_preprocessed)
    assert "INDICE_PERFORMANCE" in result.columns


def test_create_composite_creates_gap_auto_real(fe, df_preprocessed):
    result = fe.create_composite_features(df_preprocessed)
    assert "GAP_AUTO_REAL" in result.columns


def test_create_composite_creates_abaixo_media_geral(fe, df_preprocessed):
    result = fe.create_composite_features(df_preprocessed)
    assert "ABAIXO_MEDIA_GERAL" in result.columns


def test_indice_bemestar_formula_correta(fe, df_preprocessed):
    """INDICE_BEMESTAR deve ser a média de IPS, IPP e IPV."""
    result = fe.create_composite_features(df_preprocessed)
    esperado = (
        df_preprocessed["IPS"] + df_preprocessed["IPP"] + df_preprocessed["IPV"]
    ) / 3
    pd.testing.assert_series_equal(
        result["INDICE_BEMESTAR"].reset_index(drop=True),
        esperado.reset_index(drop=True),
        check_names=False,
    )


def test_indice_performance_formula_correta(fe, df_preprocessed):
    """INDICE_PERFORMANCE deve ser a média de IDA, IEG e IAA."""
    result = fe.create_composite_features(df_preprocessed)
    esperado = (
        df_preprocessed["IDA"] + df_preprocessed["IEG"] + df_preprocessed["IAA"]
    ) / 3
    pd.testing.assert_series_equal(
        result["INDICE_PERFORMANCE"].reset_index(drop=True),
        esperado.reset_index(drop=True),
        check_names=False,
    )


def test_gap_auto_real_formula_correta(fe, df_preprocessed):
    """GAP_AUTO_REAL deve ser IAA - IDA."""
    result = fe.create_composite_features(df_preprocessed)
    esperado = df_preprocessed["IAA"] - df_preprocessed["IDA"]
    pd.testing.assert_series_equal(
        result["GAP_AUTO_REAL"].reset_index(drop=True),
        esperado.reset_index(drop=True),
        check_names=False,
    )


def test_abaixo_media_geral_e_binario(fe, df_preprocessed):
    """ABAIXO_MEDIA_GERAL deve conter apenas 0 e 1."""
    result = fe.create_composite_features(df_preprocessed)
    assert set(result["ABAIXO_MEDIA_GERAL"].unique()).issubset({0, 1})


def test_create_composite_nao_modifica_original(fe, df_preprocessed):
    """create_composite_features não deve alterar o DataFrame original."""
    colunas_antes = set(df_preprocessed.columns)
    fe.create_composite_features(df_preprocessed)
    assert set(df_preprocessed.columns) == colunas_antes


# ── create_temporal_features ─────────────────────────────────────────────────


def test_create_temporal_creates_evolucao_inde(fe, df_preprocessed):
    result = fe.create_temporal_features(df_preprocessed)
    assert "EVOLUCAO_INDE" in result.columns


def test_create_temporal_sem_nome_usa_zero(fe, df_preprocessed):
    """Sem coluna NOME, EVOLUCAO_INDE deve ser zerada."""
    df_sem_nome = df_preprocessed.drop(columns=["NOME"], errors="ignore")
    result = fe.create_temporal_features(df_sem_nome)
    assert "EVOLUCAO_INDE" in result.columns
    assert (result["EVOLUCAO_INDE"] == 0).all()


def test_create_temporal_evolucao_inde_primeiro_ano_e_zero(fe, df_preprocessed):
    """A primeira observação de cada aluno deve ter EVOLUCAO_INDE = 0 (sem diff)."""
    result = fe.create_temporal_features(df_preprocessed)
    assert result["EVOLUCAO_INDE"].isna().sum() == 0


# ── create_interaction_features ──────────────────────────────────────────────


def test_create_interaction_creates_inde_x_fase(fe, df_preprocessed):
    result = fe.create_interaction_features(df_preprocessed)
    assert "INDE_x_FASE" in result.columns


def test_create_interaction_creates_bemestar_x_performance(fe, df_preprocessed):
    # Precisa das compostas antes
    df = fe.create_composite_features(df_preprocessed)
    result = fe.create_interaction_features(df)
    assert "BEMESTAR_x_PERFORMANCE" in result.columns


def test_inde_x_fase_formula_correta(fe, df_preprocessed):
    result = fe.create_interaction_features(df_preprocessed)
    esperado = df_preprocessed["INDE"] * df_preprocessed["FASE"]
    pd.testing.assert_series_equal(
        result["INDE_x_FASE"].reset_index(drop=True),
        esperado.reset_index(drop=True),
        check_names=False,
    )


def test_create_interaction_sem_compostas_pula_bemestar_x_perf(fe, df_preprocessed):
    """Sem INDICE_BEMESTAR no df, BEMESTAR_x_PERFORMANCE não deve ser criado."""
    result = fe.create_interaction_features(df_preprocessed)
    assert "BEMESTAR_x_PERFORMANCE" not in result.columns


# ── select_features ───────────────────────────────────────────────────────────


def test_select_features_exclui_target(fe, df_preprocessed):
    df = fe.transform(df_preprocessed)
    features = fe.select_features(df)
    assert "DEFASAGEM" not in features


def test_select_features_exclui_identificadores(fe, df_preprocessed):
    df = fe.transform(df_preprocessed)
    features = fe.select_features(df)
    for col in ["NOME", "TURMA"]:
        assert col not in features


def test_select_features_retorna_apenas_numericas(fe, df_preprocessed):
    df = fe.transform(df_preprocessed)
    features = fe.select_features(df)
    for col in features:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} não é numérico"


def test_select_features_retorna_lista(fe, df_preprocessed):
    df = fe.transform(df_preprocessed)
    features = fe.select_features(df)
    assert isinstance(features, list)
    assert len(features) > 0


# ── transform (pipeline completo) ─────────────────────────────────────────────


def test_transform_adiciona_features_engineered(fe, df_preprocessed):
    result = fe.transform(df_preprocessed)
    novas = {
        "INDICE_BEMESTAR",
        "INDICE_PERFORMANCE",
        "GAP_AUTO_REAL",
        "ABAIXO_MEDIA_GERAL",
        "EVOLUCAO_INDE",
        "INDE_x_FASE",
    }
    for col in novas:
        assert col in result.columns, f"Feature '{col}' não foi criada"


def test_transform_aumenta_numero_de_colunas(fe, df_preprocessed):
    n_antes = df_preprocessed.shape[1]
    result = fe.transform(df_preprocessed)
    assert result.shape[1] > n_antes


def test_transform_preserva_numero_de_linhas(fe, df_preprocessed):
    result = fe.transform(df_preprocessed)
    assert len(result) == len(df_preprocessed)


def test_transform_nao_modifica_original(fe, df_preprocessed):
    n_cols_antes = df_preprocessed.shape[1]
    fe.transform(df_preprocessed)
    assert df_preprocessed.shape[1] == n_cols_antes


@pytest.mark.parametrize(
    "feature",
    [
        "INDICE_BEMESTAR",
        "INDICE_PERFORMANCE",
        "GAP_AUTO_REAL",
        "EVOLUCAO_INDE",
    ],
)
def test_transform_feature_sem_nulos(fe, df_preprocessed, feature):
    result = fe.transform(df_preprocessed)
    assert result[feature].isna().sum() == 0, f"{feature} contém NaN"
