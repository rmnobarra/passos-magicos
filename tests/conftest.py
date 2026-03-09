"""
Fixtures compartilhadas entre todos os módulos de teste.

Fornece DataFrames sintéticos que simulam o schema Passos Mágicos
e um modelo treinado com dados sintéticos para testes de integração.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """DataFrame sintético simulando o schema Passos Mágicos em formato long."""
    np.random.seed(42)
    n = 150

    df = pd.DataFrame(
        {
            "NOME": [f"ESTUDANTE_{i:03d}" for i in range(n)],
            "ANO": np.random.choice([2022, 2023, 2024], n),
            "FASE": np.random.randint(0, 9, n),
            "TURMA": np.random.choice(["A", "B", "C", "D"], n),
            "INDE": np.random.uniform(2.0, 10.0, n),
            "IAN": np.random.uniform(2.0, 10.0, n),
            "IDA": np.random.uniform(2.0, 10.0, n),
            "IEG": np.random.uniform(2.0, 10.0, n),
            "IAA": np.random.uniform(2.0, 10.0, n),
            "IPS": np.random.uniform(2.0, 10.0, n),
            "IPP": np.random.uniform(2.0, 10.0, n),
            "IPV": np.random.uniform(2.0, 10.0, n),
            "PONTO_DE_VIRADA": np.random.choice([0, 1], n),
        }
    )

    # Target binário derivado das regras de negócio
    df["DEFASAGEM"] = ((df["INDE"] < 5.0) | (df["IAN"] < 5.0)).astype(int)
    return df


@pytest.fixture
def sample_dataframe_with_missing(sample_dataframe: pd.DataFrame) -> pd.DataFrame:
    """DataFrame com ~10% de NaN em colunas numéricas para testar imputação."""
    np.random.seed(0)
    df = sample_dataframe.copy()
    for col in ["INDE", "IDA", "IEG"]:
        mask = np.random.random(len(df)) < 0.10
        df.loc[mask, col] = np.nan
    return df


@pytest.fixture
def wide_dataframe() -> pd.DataFrame:
    """DataFrame no formato wide (colunas sufixadas por ano) para testar reshape."""
    np.random.seed(7)
    n = 20
    df = pd.DataFrame(
        {
            "NOME": [f"EST_{i:03d}" for i in range(n)],
            # 2021
            "INDE_2021": np.random.uniform(4.0, 9.0, n),
            "IAN_2021": np.random.uniform(4.0, 9.0, n),
            "IDA_2021": np.random.uniform(4.0, 9.0, n),
            "IEG_2021": np.random.uniform(4.0, 9.0, n),
            "IAA_2021": np.random.uniform(4.0, 9.0, n),
            "IPS_2021": np.random.uniform(4.0, 9.0, n),
            "IPP_2021": np.random.uniform(4.0, 9.0, n),
            "IPV_2021": np.random.uniform(4.0, 9.0, n),
            "FASE_2021": np.random.randint(1, 8, n).astype(float),
            "TURMA_2021": np.random.choice(["A", "B"], n),
            "PONTO_VIRADA_2021": np.random.choice(["Sim", "Não"], n),
            "DEFASAGEM_2021": np.random.choice([-2, -1, 0, 1], n),
            # 2022
            "INDE_2022": np.random.uniform(4.0, 9.0, n),
            "IAN_2022": np.random.uniform(4.0, 9.0, n),
            "IDA_2022": np.random.uniform(4.0, 9.0, n),
            "IEG_2022": np.random.uniform(4.0, 9.0, n),
            "IAA_2022": np.random.uniform(4.0, 9.0, n),
            "IPS_2022": np.random.uniform(4.0, 9.0, n),
            "IPP_2022": np.random.uniform(4.0, 9.0, n),
            "IPV_2022": np.random.uniform(4.0, 9.0, n),
            "FASE_2022": np.random.randint(1, 8, n).astype(float),
            "TURMA_2022": np.random.choice(["A", "B"], n),
            "PONTO_VIRADA_2022": np.random.choice(["Sim", "Não"], n),
        }
    )
    return df


@pytest.fixture
def trained_pipeline(sample_dataframe: pd.DataFrame):
    """Pipeline sklearn treinado com dados sintéticos para testes de treino/avaliação."""
    from src.feature_engineering import FeatureEngineer
    from src.preprocessing import DataPreprocessor
    from src.train import ModelTrainer

    preprocessor = DataPreprocessor()
    fe = FeatureEngineer()
    trainer = ModelTrainer(config={})

    df = preprocessor.fit_transform(sample_dataframe.copy())
    df = fe.transform(df)

    feature_cols = [c for c in df.columns if c not in ["DEFASAGEM", "NOME", "TURMA"]]
    X = df[feature_cols]
    y = df["DEFASAGEM"]

    return trainer.train(X, y)


@pytest.fixture
def trainer_with_data(sample_dataframe: pd.DataFrame):
    """ModelTrainer inicializado + X/y prontos para uso nos testes."""
    from src.feature_engineering import FeatureEngineer
    from src.preprocessing import DataPreprocessor
    from src.train import ModelTrainer

    preprocessor = DataPreprocessor()
    fe = FeatureEngineer()
    trainer = ModelTrainer(config={})

    df = preprocessor.fit_transform(sample_dataframe.copy())
    df = fe.transform(df)

    feature_cols = [c for c in df.columns if c not in ["DEFASAGEM", "NOME", "TURMA"]]
    X = df[feature_cols]
    y = df["DEFASAGEM"]

    return trainer, X, y


@pytest.fixture
def y_arrays():
    """Arrays sintéticos de y_true / y_pred / y_proba para testes de avaliação."""
    np.random.seed(42)
    n = 80
    y_true = np.random.randint(0, 2, n)
    y_pred = y_true.copy()
    # Introduzir ~10% de erros propositais para métricas não triviais
    flip_idx = np.random.choice(n, size=int(n * 0.10), replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]
    y_proba = np.where(
        y_pred == 1, np.random.uniform(0.55, 0.99, n), np.random.uniform(0.01, 0.45, n)
    )
    return y_true, y_pred, y_proba


@pytest.fixture
def valid_api_payload() -> dict:
    """Payload válido para o endpoint POST /predict."""
    return {
        "student_id": "TEST-001",
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


@pytest.fixture
def high_risk_payload() -> dict:
    """Payload com indicadores baixos, esperado classificar como alto risco."""
    return {
        "inde": 2.5,
        "ian": 2.8,
        "ida": 3.0,
        "ieg": 3.5,
        "iaa": 4.0,
        "ips": 3.2,
        "ipp": 3.8,
        "ipv": 2.9,
        "fase": 1,
        "ano": 2024,
        "ponto_de_virada": False,
    }
