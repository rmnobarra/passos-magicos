"""
Módulo de detecção de data drift usando Evidently.
Compara distribuição dos dados de produção com os dados de referência (treino).

Uso via CLI:
    python monitoring/drift_detector.py --reference data/raw/PEDE_PASSOS_DATASET_FIAP.csv
    python monitoring/drift_detector.py --reference ref.csv --current prod.csv
"""
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

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

    @classmethod
    def from_dataframe(cls, reference_data: pd.DataFrame, threshold: float = 0.15) -> "DriftDetector":
        """
        Cria DriftDetector a partir de um DataFrame já processado,
        sem precisar salvar em CSV.

        Args:
            reference_data: DataFrame com dados de referência.
            threshold: Limite de drift para alertar (padrão: 15%).
        """
        instance = cls.__new__(cls)
        instance.threshold = threshold
        instance.reference_data = reference_data
        logger.info(f"DriftDetector inicializado com {len(reference_data)} amostras de referência")
        return instance

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

        # DatasetDriftMetric (índice 0) tem o score geral
        summary = result['metrics'][0]['result']
        drift_score = summary.get('share_of_drifted_columns', 0)
        drift_detected = drift_score > self.threshold

        # DataDriftTable (índice 1) tem o detalhamento por coluna
        table = result['metrics'][1]['result'] if len(result['metrics']) > 1 else {}
        drifted_features = [
            feat for feat, info in table.get('drift_by_columns', {}).items()
            if info.get('drift_detected')
        ]

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


if __name__ == "__main__":
    import argparse

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Detecta data drift entre dados de referência e produção."
    )
    parser.add_argument(
        "--reference", required=True,
        help="CSV com dados de referência (ex: data/raw/PEDE_PASSOS_DATASET_FIAP.csv)",
    )
    parser.add_argument(
        "--current", default=None,
        help="CSV com dados atuais. Se omitido, usa amostra aleatória (30%%) da referência.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="Limiar de drift para alerta (padrão: 0.15)",
    )
    args = parser.parse_args()

    from src.preprocessing import DataPreprocessor
    from src.feature_engineering import FeatureEngineer

    dp = DataPreprocessor()
    fe = FeatureEngineer()

    logger.info("Processando dados de referência...")
    ref_df = fe.transform(dp.fit_transform(dp.load_data(args.reference)))
    feature_cols = [
        c for c in ref_df.select_dtypes(include="number").columns
        if c != "DEFASAGEM"
    ]
    reference_data = ref_df[feature_cols].dropna()

    if args.current:
        logger.info("Processando dados atuais...")
        cur_df = fe.transform(dp.fit_transform(dp.load_data(args.current)))
        current_data = cur_df[feature_cols].dropna()
    else:
        logger.info("--current não fornecido: usando amostra aleatória (30%) da referência.")
        current_data = reference_data.sample(frac=0.3, random_state=99)

    detector = DriftDetector.from_dataframe(reference_data, threshold=args.threshold)
    result = detector.detect_drift(current_data)

    print(f"\n{'='*55}")
    print(f"  Drift detectado : {result['drift_detected']}")
    print(f"  Score de drift  : {result['drift_score']:.2%}")
    print(f"  Features        : {result['drifted_features'] or 'nenhuma'}")
    print(f"  Relatório       : {result['report_path']}")
    print(f"{'='*55}\n")
