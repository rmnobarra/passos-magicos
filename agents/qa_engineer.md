# Agente: QA Engineer — Passos Mágicos

## Papel
Você é um engenheiro de qualidade especialista em testes. Sua responsabilidade é garantir ≥80% de cobertura de testes unitários e implementar o sistema de monitoramento contínuo.

## Arquivos sob sua responsabilidade
- `tests/conftest.py`
- `tests/test_preprocessing.py`
- `tests/test_feature_engineering.py`
- `tests/test_train.py`
- `tests/test_evaluate.py`
- `tests/test_api.py`
- `monitoring/drift_detector.py`
- `monitoring/dashboard.py`

---

## Tarefa 1: `tests/conftest.py`

```python
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


@pytest.fixture
def sample_dataframe():
    """DataFrame sintético simulando o schema Passos Mágicos."""
    np.random.seed(42)
    n = 150

    df = pd.DataFrame({
        'NOME': [f'ESTUDANTE_{i:03d}' for i in range(n)],
        'ANO': np.random.choice([2022, 2023, 2024], n),
        'FASE': np.random.randint(0, 9, n),
        'TURMA': np.random.choice(['A', 'B', 'C', 'D'], n),
        'INDE': np.random.uniform(2.0, 10.0, n),
        'IAN': np.random.uniform(2.0, 10.0, n),
        'IDA': np.random.uniform(2.0, 10.0, n),
        'IEG': np.random.uniform(2.0, 10.0, n),
        'IAA': np.random.uniform(2.0, 10.0, n),
        'IPS': np.random.uniform(2.0, 10.0, n),
        'IPP': np.random.uniform(2.0, 10.0, n),
        'IPV': np.random.uniform(2.0, 10.0, n),
        'PONTO_DE_VIRADA': np.random.choice([0, 1], n),
    })

    # Derivar target DEFASAGEM
    df['DEFASAGEM'] = ((df['INDE'] < 5.0) | (df['IAN'] < 5.0)).astype(int)
    return df


@pytest.fixture
def sample_dataframe_with_missing(sample_dataframe):
    """DataFrame com valores ausentes para testar imputação."""
    df = sample_dataframe.copy()
    # Introduzir ~10% de NaN em colunas numéricas
    for col in ['INDE', 'IDA', 'IEG']:
        mask = np.random.random(len(df)) < 0.1
        df.loc[mask, col] = np.nan
    return df


@pytest.fixture
def trained_model(sample_dataframe):
    """Modelo treinado com dados sintéticos para testes de API."""
    from src.preprocessing import DataPreprocessor
    from src.feature_engineering import FeatureEngineer
    from src.train import ModelTrainer
    from sklearn.pipeline import Pipeline

    preprocessor = DataPreprocessor()
    fe = FeatureEngineer()
    trainer = ModelTrainer(config={})

    df = preprocessor.fit_transform(sample_dataframe.copy())
    df = fe.transform(df)

    feature_cols = [c for c in df.columns if c not in ['DEFASAGEM', 'NOME', 'TURMA']]
    X = df[feature_cols]
    y = df['DEFASAGEM']

    return trainer.train(X, y)


@pytest.fixture
def valid_api_payload():
    return {
        "student_id": "TEST-001",
        "inde": 6.5, "ian": 7.2, "ida": 5.8,
        "ieg": 6.0, "iaa": 7.5, "ips": 6.8,
        "ipp": 7.1, "ipv": 6.3, "fase": 3,
        "ano": 2024, "ponto_de_virada": False
    }


@pytest.fixture
def high_risk_payload():
    """Payload esperado para predição de alto risco."""
    return {
        "inde": 2.5, "ian": 2.8, "ida": 3.0,
        "ieg": 3.5, "iaa": 4.0, "ips": 3.2,
        "ipp": 3.8, "ipv": 2.9, "fase": 1,
        "ano": 2024, "ponto_de_virada": False
    }
```

---

## Tarefa 2: Estratégia de Cobertura

### Mapeamento de cobertura por módulo:
| Módulo | Alvo | Testes obrigatórios |
|--------|------|---------------------|
| `src/preprocessing.py` | 85% | load, validate, missing, duplicates, target, split |
| `src/feature_engineering.py` | 85% | composite features, temporal, transform |
| `src/train.py` | 80% | build_pipeline, train, cross_validate, save |
| `src/evaluate.py` | 80% | metrics, reliability check, report |
| `app/routes.py` | 80% | predict, health, metrics, validation errors |
| `src/utils.py` | 75% | logger, save/load artifact, ensure_dir |

### Comando de cobertura:
```bash
pytest tests/ \
  --cov=src \
  --cov=app \
  --cov-report=term-missing \
  --cov-report=html:htmlcov \
  --cov-fail-under=80 \
  -v
```

---

## Tarefa 3: `monitoring/drift_detector.py`

```python
"""
Módulo de detecção de data drift usando Evidently.
Compara distribuição dos dados de produção com os dados de referência (treino).
"""
import pandas as pd
import joblib
import logging
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import DatasetDriftMetric

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, reference_data_path: str, threshold: float = 0.15):
        """
        Args:
            reference_data_path: Caminho para os dados de referência (treino)
            threshold: Limite de drift para alertar (padrão: 15%)
        """
        self.threshold = threshold
        self.reference_data = pd.read_csv(reference_data_path)
        logger.info(f"DriftDetector inicializado com {len(self.reference_data)} amostras de referência")

    def detect_drift(self, current_data: pd.DataFrame) -> dict:
        """
        Detecta drift entre dados atuais e de referência.
        
        Returns:
            dict com 'drift_detected' (bool), 'drift_score' (float),
            'drifted_features' (list) e 'report_path' (str)
        """
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )

        result = report.as_dict()
        drift_metrics = result['metrics'][0]['result']

        drifted_features = [
            feat for feat, info in drift_metrics['drift_by_columns'].items()
            if info['drift_detected']
        ]

        drift_score = drift_metrics.get('share_of_drifted_columns', 0)
        drift_detected = drift_score > self.threshold

        if drift_detected:
            logger.warning(
                f"DRIFT DETECTADO! Score={drift_score:.2%} "
                f"| Features com drift: {drifted_features}"
            )

        # Salvar relatório HTML
        report_path = f"reports/drift_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.html"
        Path("reports").mkdir(exist_ok=True)
        report.save_html(report_path)

        return {
            "drift_detected": drift_detected,
            "drift_score": round(drift_score, 4),
            "drifted_features": drifted_features,
            "report_path": report_path
        }
```

---

## Tarefa 4: `monitoring/dashboard.py`

```python
"""
Dashboard Streamlit para monitoramento do modelo em produção.
Exibe métricas de uso, logs de predições e alertas de drift.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Passos Mágicos — Monitor ML",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Passos Mágicos — Monitoramento do Modelo de Defasagem")
st.caption(f"Atualizado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# --- Carregar logs de predições ---
LOG_PATH = Path("logs/predictions.jsonl")

def load_predictions():
    if not LOG_PATH.exists():
        return pd.DataFrame()
    records = []
    with open(LOG_PATH) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass
    return pd.DataFrame(records) if records else pd.DataFrame()

df_preds = load_predictions()

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)

if not df_preds.empty:
    total = len(df_preds)
    alto_risco = df_preds['risco_defasagem'].sum() if 'risco_defasagem' in df_preds else 0
    col1.metric("Total de Predições", total)
    col2.metric("Em Risco de Defasagem", int(alto_risco))
    col3.metric("Taxa de Risco", f"{alto_risco/total:.1%}" if total > 0 else "—")
    col4.metric("Última Predição", df_preds['timestamp'].max() if 'timestamp' in df_preds else "—")
else:
    col1.metric("Total de Predições", 0)
    col2.metric("Em Risco", 0)
    col3.metric("Taxa de Risco", "—")
    col4.metric("Última Predição", "—")
    st.info("Nenhuma predição registrada ainda. Faça chamadas à API para ver os dados aqui.")

st.divider()

# --- Seção de Drift ---
st.subheader("📊 Status de Drift do Modelo")
drift_reports = list(Path("reports").glob("drift_report_*.html")) if Path("reports").exists() else []

if drift_reports:
    latest = max(drift_reports, key=os.path.getctime)
    st.success(f"✅ Último relatório de drift: `{latest.name}`")
    with open(latest) as f:
        st.download_button("📥 Baixar Relatório de Drift", f.read(), file_name=latest.name)
else:
    st.warning("⚠️ Nenhum relatório de drift gerado ainda. Execute `python monitoring/drift_detector.py`")

# --- Tabela de predições recentes ---
if not df_preds.empty:
    st.subheader("📋 Predições Recentes")
    st.dataframe(df_preds.tail(50), use_container_width=True)
```

---

## Padrões de qualidade para testes
- Usar `pytest.mark.parametrize` para testar múltiplos cenários com um único teste
- Nunca depender de arquivos externos nos testes unitários (usar fixtures/mocks)
- Testes de API: mockar o modelo com `unittest.mock.patch` quando o .joblib não existir
- Nomear testes descritivamente: `test_<função>_<cenário>_<resultado_esperado>`
