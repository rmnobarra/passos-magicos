"""
Testes unitários para src/preprocessing.py.

Cobre: carregamento, validação de schema, reshape wide→long,
tratamento de valores ausentes, remoção de duplicatas,
codificação do target e divisão treino/teste.
"""

import io

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import DataPreprocessor, SCHEMA_MINIMO


# ── Fixtures locais ──────────────────────────────────────────────────────────


@pytest.fixture
def dp() -> DataPreprocessor:
    return DataPreprocessor()


@pytest.fixture
def csv_file(tmp_path, sample_dataframe) -> str:
    """Salva o sample_dataframe como CSV temporário e retorna o caminho."""
    path = str(tmp_path / "dados.csv")
    sample_dataframe.to_csv(path, sep=";", index=False)
    return path


# ── load_data ────────────────────────────────────────────────────────────────


def test_load_data_success(dp, csv_file):
    df = dp.load_data(csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_data_file_not_found_raises(dp, tmp_path):
    with pytest.raises(Exception):
        dp.load_data(str(tmp_path / "nao_existe.csv"))


def test_load_data_preserves_row_count(dp, csv_file, sample_dataframe):
    df = dp.load_data(csv_file)
    assert len(df) == len(sample_dataframe)


# ── normalize_column_names ────────────────────────────────────────────────────


def test_normalize_column_names_converts_to_uppercase(dp):
    df = pd.DataFrame({"nome": [1], "inde": [2]})
    result = dp.normalize_column_names(df)
    assert all(c == c.upper() for c in result.columns)


def test_normalize_column_names_replaces_spaces(dp):
    df = pd.DataFrame({"nome aluno": [1], "inde valor": [2]})
    result = dp.normalize_column_names(df)
    assert "NOME_ALUNO" in result.columns
    assert "INDE_VALOR" in result.columns


def test_normalize_column_names_does_not_modify_original(dp, sample_dataframe):
    original_cols = list(sample_dataframe.columns)
    dp.normalize_column_names(sample_dataframe)
    assert list(sample_dataframe.columns) == original_cols


# ── validate_schema ───────────────────────────────────────────────────────────


def test_validate_schema_success(dp, sample_dataframe):
    assert dp.validate_schema(sample_dataframe) is True


def test_validate_schema_missing_column_raises(dp, sample_dataframe):
    df_sem_inde = sample_dataframe.drop(columns=["INDE"])
    with pytest.raises(ValueError, match="Colunas obrigatórias ausentes"):
        dp.validate_schema(df_sem_inde)


@pytest.mark.parametrize("coluna_removida", ["INDE", "IAN", "FASE", "ANO"])
def test_validate_schema_cada_coluna_obrigatoria(dp, sample_dataframe, coluna_removida):
    df = sample_dataframe.drop(columns=[coluna_removida])
    with pytest.raises(ValueError):
        dp.validate_schema(df)


# ── handle_missing_values ────────────────────────────────────────────────────


def test_handle_missing_values_sem_nulos_retorna_mesmo_tamanho(dp, sample_dataframe):
    result = dp.handle_missing_values(sample_dataframe)
    assert len(result) == len(sample_dataframe)


def test_handle_missing_values_imputa_numericos(dp, sample_dataframe_with_missing):
    result = dp.handle_missing_values(sample_dataframe_with_missing)
    for col in ["INDE", "IDA", "IEG"]:
        assert result[col].isna().sum() == 0


def test_handle_missing_values_imputa_categoricos_com_moda(dp, sample_dataframe):
    df = sample_dataframe.copy()
    df.loc[:10, "TURMA"] = np.nan
    result = dp.handle_missing_values(df)
    assert result["TURMA"].isna().sum() == 0


def test_handle_missing_values_remove_linhas_muito_vazias(dp, sample_dataframe):
    df = sample_dataframe.copy()
    # Linha com >50% de NaN
    num_cols = sample_dataframe.select_dtypes(include="number").columns
    df.loc[0, num_cols[: len(num_cols) // 2 + 2]] = np.nan
    result = dp.handle_missing_values(df)
    assert len(result) < len(df) or result.isna().sum().sum() == 0


def test_handle_missing_values_nao_modifica_original(dp, sample_dataframe_with_missing):
    original_nulos = sample_dataframe_with_missing["INDE"].isna().sum()
    dp.handle_missing_values(sample_dataframe_with_missing)
    assert sample_dataframe_with_missing["INDE"].isna().sum() == original_nulos


# ── remove_duplicates ────────────────────────────────────────────────────────


def test_remove_duplicates_elimina_linhas_iguais(dp, sample_dataframe):
    df_com_dup = pd.concat(
        [sample_dataframe, sample_dataframe.iloc[:5]], ignore_index=True
    )
    result = dp.remove_duplicates(df_com_dup)
    assert len(result) == len(sample_dataframe)


def test_remove_duplicates_sem_duplicatas_mantem_tamanho(dp, sample_dataframe):
    result = dp.remove_duplicates(sample_dataframe)
    assert len(result) == len(sample_dataframe)


def test_remove_duplicates_nao_modifica_original(dp, sample_dataframe):
    df_com_dup = pd.concat(
        [sample_dataframe, sample_dataframe.iloc[:3]], ignore_index=True
    )
    n_antes = len(df_com_dup)
    dp.remove_duplicates(df_com_dup)
    assert len(df_com_dup) == n_antes


# ── encode_target ─────────────────────────────────────────────────────────────


def test_encode_target_preserva_binario_existente(dp, sample_dataframe):
    result = dp.encode_target(sample_dataframe)
    valores = set(result["DEFASAGEM"].unique())
    assert valores.issubset({0, 1})


def test_encode_target_deriva_quando_ausente(dp, sample_dataframe):
    df = sample_dataframe.drop(columns=["DEFASAGEM"])
    result = dp.encode_target(df)
    assert "DEFASAGEM" in result.columns
    assert set(result["DEFASAGEM"].unique()).issubset({0, 1})


def test_encode_target_derivacao_segue_regra_negocio(dp, sample_dataframe):
    df = sample_dataframe.drop(columns=["DEFASAGEM"])
    result = dp.encode_target(df)
    em_risco = (df["INDE"] < 5.0) | (df["IAN"] < 5.0)
    assert (result["DEFASAGEM"] == em_risco.astype(int)).all()


def test_encode_target_binariza_valores_nao_binarios(dp, sample_dataframe):
    """DEFASAGEM com valores inteiros negativos (anos de atraso) deve ser binarizada."""
    df = sample_dataframe.copy()
    df["DEFASAGEM"] = np.random.choice([-2, -1, 0, 1, 2], len(df))
    result = dp.encode_target(df)
    assert set(result["DEFASAGEM"].unique()).issubset({0, 1})


def test_encode_target_tipo_inteiro(dp, sample_dataframe):
    result = dp.encode_target(sample_dataframe)
    assert result["DEFASAGEM"].dtype in [int, np.int32, np.int64]


# ── split_data ───────────────────────────────────────────────────────────────


def test_split_data_proporcao_80_20(dp, sample_dataframe):
    X_train, X_test, y_train, y_test = dp.split_data(sample_dataframe, test_size=0.2)
    total = len(X_train) + len(X_test)
    assert abs(len(X_test) / total - 0.2) < 0.05


def test_split_data_sem_vazamento(dp, sample_dataframe):
    X_train, X_test, y_train, y_test = dp.split_data(sample_dataframe)
    idx_train = set(X_train.index)
    idx_test = set(X_test.index)
    assert idx_train.isdisjoint(idx_test)


def test_split_data_sem_defasagem_levanta_erro(dp, sample_dataframe):
    df = sample_dataframe.drop(columns=["DEFASAGEM"])
    with pytest.raises(ValueError, match="DEFASAGEM"):
        dp.split_data(df)


def test_split_data_retorna_quatro_elementos(dp, sample_dataframe):
    resultado = dp.split_data(sample_dataframe)
    assert len(resultado) == 4


# ── wide → long ──────────────────────────────────────────────────────────────


def test_is_wide_format_detecta_sufixo_ano(dp, wide_dataframe):
    assert dp._is_wide_format(wide_dataframe) is True


def test_is_wide_format_rejeita_formato_long(dp, sample_dataframe):
    assert dp._is_wide_format(sample_dataframe) is False


def test_reshape_wide_to_long_multiplica_linhas_por_anos(dp, wide_dataframe):
    n_anos = 2  # 2021 e 2022
    result = dp._reshape_wide_to_long(wide_dataframe)
    assert len(result) == len(wide_dataframe) * n_anos


def test_reshape_wide_to_long_cria_coluna_ano(dp, wide_dataframe):
    result = dp._reshape_wide_to_long(wide_dataframe)
    assert "ANO" in result.columns
    assert set(result["ANO"].unique()) == {2021, 2022}


def test_reshape_wide_to_long_colunas_padronizadas(dp, wide_dataframe):
    result = dp._reshape_wide_to_long(wide_dataframe)
    for col in ["INDE", "IAN", "IDA", "FASE"]:
        assert col in result.columns, f"Coluna {col} ausente após reshape"


def test_reshape_wide_to_long_converte_tipos_numericos(dp, wide_dataframe):
    result = dp._reshape_wide_to_long(wide_dataframe)
    for col in ["INDE", "IAN", "IDA"]:
        assert pd.api.types.is_numeric_dtype(result[col]), f"{col} deveria ser numérico"


# ── fit_transform (pipeline completo) ────────────────────────────────────────


def test_fit_transform_retorna_dataframe(dp, sample_dataframe):
    result = dp.fit_transform(sample_dataframe)
    assert isinstance(result, pd.DataFrame)


def test_fit_transform_contem_coluna_defasagem(dp, sample_dataframe):
    result = dp.fit_transform(sample_dataframe)
    assert "DEFASAGEM" in result.columns


def test_fit_transform_sem_nulos_no_resultado(dp, sample_dataframe_with_missing):
    result = dp.fit_transform(sample_dataframe_with_missing)
    assert result[["INDE", "IDA", "IEG"]].isna().sum().sum() == 0


def test_fit_transform_processa_formato_wide(dp, wide_dataframe):
    """fit_transform deve detectar e converter wide→long automaticamente."""
    result = dp.fit_transform(wide_dataframe)
    assert "ANO" in result.columns
    assert len(result) > len(wide_dataframe)
