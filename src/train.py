"""
Módulo de treinamento do modelo preditivo de defasagem escolar — Passos Mágicos.

Implementa a classe ModelTrainer com comparação de algoritmos, validação cruzada,
busca de hiperparâmetros e serialização do melhor modelo encontrado.

Uso via linha de comando:
    python src/train.py --data data/raw/PEDE_PASSOS_DATASET_FIAP.csv
"""

import argparse
import os
import sys

# Garante que a raiz do projeto esteja no sys.path ao executar diretamente
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.evaluate import ModelEvaluator
from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataPreprocessor
from src.utils import ensure_dir, get_logger, load_config, save_artifact

logger = get_logger(__name__)

# Features numéricas candidatas para o modelo
NUMERICAL_FEATURES_CANDIDATAS: list[str] = [
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
]

# Features categóricas candidatas para o modelo
CATEGORICAL_FEATURES_CANDIDATAS: list[str] = [
    "FASE",
    "PONTO_DE_VIRADA",
]


class ModelTrainer:
    """
    Responsável por construir, comparar, treinar e serializar o modelo preditivo.

    Compara RandomForestClassifier, XGBClassifier e LGBMClassifier via
    validação cruzada estratificada, seleciona o melhor e o persiste em disco.
    """

    def __init__(self, config: dict) -> None:
        """
        Inicializa o ModelTrainer com configurações do projeto.

        Args:
            config: Dicionário com entradas como 'model_path' e 'metadata_path'.
        """
        self.config = config
        self.model_path: str = config.get("model_path", "app/model/model.joblib")
        self.metadata_path: str = config.get(
            "metadata_path", "app/model/metadata.joblib"
        )
        self.numerical_features: list[str] = []
        self.categorical_features: list[str] = []
        self.feature_names: list[str] = []

    def _filter_features(self, X: pd.DataFrame) -> None:
        """
        Identifica quais features candidatas estão presentes em X.

        Atualiza os atributos internos numerical_features, categorical_features
        e feature_names com as colunas efetivamente disponíveis no DataFrame.

        Args:
            X: DataFrame com as features disponíveis.
        """
        self.numerical_features = [
            f for f in NUMERICAL_FEATURES_CANDIDATAS if f in X.columns
        ]
        self.categorical_features = [
            f for f in CATEGORICAL_FEATURES_CANDIDATAS if f in X.columns
        ]
        self.feature_names = self.numerical_features + self.categorical_features

        logger.info(
            f"Features numéricas ({len(self.numerical_features)}): {self.numerical_features}"
        )
        logger.info(
            f"Features categóricas ({len(self.categorical_features)}): {self.categorical_features}"
        )

    def build_pipeline(self, classifier=None) -> Pipeline:
        """
        Constrói o pipeline sklearn com ColumnTransformer e classificador.

        Aplica StandardScaler nas features numéricas e OneHotEncoder nas
        categóricas. Usa os atributos internos definidos por _filter_features.

        Args:
            classifier: Classificador sklearn-compatible. Se None, usa
                        RandomForestClassifier com os hiperparâmetros padrão.

        Returns:
            Pipeline sklearn configurado, pronto para fit.

        Raises:
            ValueError: Se _filter_features ainda não foi chamado.
        """
        if classifier is None:
            classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

        transformers = []
        if self.numerical_features:
            transformers.append(("num", StandardScaler(), self.numerical_features))
        if self.categorical_features:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_features,
                )
            )

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Treina o pipeline RandomForest padrão nos dados de treino.

        Chama _filter_features internamente para definir quais colunas usar.

        Args:
            X_train: DataFrame com as features de treino.
            y_train: Série com o target de treino.

        Returns:
            Pipeline treinado.
        """
        logger.info("Treinando pipeline RandomForestClassifier")
        self._filter_features(X_train)
        pipeline = self.build_pipeline()
        pipeline.fit(X_train[self.feature_names], y_train)
        logger.info("Treinamento concluído")
        return pipeline

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pipeline: Optional[Pipeline] = None,
    ) -> dict:
        """
        Executa validação cruzada estratificada com 5 folds.

        Métricas calculadas: F1 macro, ROC-AUC, Precision macro, Recall macro.

        Args:
            X: DataFrame com as features completas.
            y: Série com o target completo.
            pipeline: Pipeline a validar. Se None, constrói um novo.

        Returns:
            Dicionário com médias e desvios padrão das métricas de CV.
        """
        logger.info("Executando validação cruzada — StratifiedKFold(n_splits=5)")

        if pipeline is None:
            self._filter_features(X)
            pipeline = self.build_pipeline()

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(
            pipeline,
            X[self.feature_names],
            y,
            cv=cv,
            scoring=["f1_macro", "roc_auc", "precision_macro", "recall_macro"],
            return_train_score=True,
            n_jobs=-1,
        )

        resumo = {
            "f1_macro_mean": float(scores["test_f1_macro"].mean()),
            "f1_macro_std": float(scores["test_f1_macro"].std()),
            "roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "roc_auc_std": float(scores["test_roc_auc"].std()),
            "precision_macro_mean": float(scores["test_precision_macro"].mean()),
            "precision_macro_std": float(scores["test_precision_macro"].std()),
            "recall_macro_mean": float(scores["test_recall_macro"].mean()),
            "recall_macro_std": float(scores["test_recall_macro"].std()),
            "train_f1_macro_mean": float(scores["train_f1_macro"].mean()),
        }

        logger.info(
            f"CV — F1 macro: {resumo['f1_macro_mean']:.4f} ± {resumo['f1_macro_std']:.4f} | "
            f"ROC-AUC: {resumo['roc_auc_mean']:.4f} ± {resumo['roc_auc_std']:.4f} | "
            f"Recall macro: {resumo['recall_macro_mean']:.4f}"
        )
        return resumo

    def hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Pipeline:
        """
        Executa busca aleatória de hiperparâmetros sobre o RandomForestClassifier.

        Usa RandomizedSearchCV com 10 iterações e validação cruzada 3-fold,
        otimizando F1 macro.

        Args:
            X_train: DataFrame com as features de treino.
            y_train: Série com o target de treino.

        Returns:
            Pipeline com os melhores hiperparâmetros encontrados, já treinado.
        """
        logger.info("Iniciando busca de hiperparâmetros — RandomizedSearchCV")
        self._filter_features(X_train)
        pipeline = self.build_pipeline()

        param_dist = {
            "classifier__n_estimators": [100, 200, 300, 500],
            "classifier__max_depth": [5, 8, 10, 15, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__min_samples_leaf": [1, 2, 4],
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=10,
            scoring="f1_macro",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train[self.feature_names], y_train)

        logger.info(f"Melhores hiperparâmetros: {search.best_params_}")
        logger.info(f"Melhor F1 macro (CV 3-fold): {search.best_score_:.4f}")
        return search.best_estimator_

    def _compare_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> tuple[Pipeline, str, dict]:
        """
        Compara RandomForest, XGBoost e LightGBM via validação cruzada.

        Seleciona o classificador com maior F1 macro médio no CV, retreina
        o melhor nos dados completos de treino e retorna.

        Args:
            X_train: DataFrame com as features de treino.
            y_train: Série com o target de treino.

        Returns:
            Tupla (pipeline_treinado, nome_algoritmo, métricas_cv).
        """
        logger.info("=" * 55)
        logger.info("Comparação de modelos — RandomForest | XGBoost | LightGBM")
        logger.info("=" * 55)

        # Peso para classes desbalanceadas no XGBoost
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos = n_neg / max(n_pos, 1)

        candidatos: dict = {
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "XGBClassifier": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0,
            ),
            "LGBMClassifier": LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        melhor_score = -1.0
        melhor_clf = None
        melhor_nome = ""
        melhor_metricas: dict = {}

        for nome, clf in candidatos.items():
            try:
                pipeline = self.build_pipeline(classifier=clf)
                scores = cross_validate(
                    pipeline,
                    X_train[self.feature_names],
                    y_train,
                    cv=cv,
                    scoring=["f1_macro", "roc_auc"],
                    n_jobs=-1,
                )
                f1_mean = float(scores["test_f1_macro"].mean())
                roc_mean = float(scores["test_roc_auc"].mean())

                logger.info(
                    f"  {nome:<35s} | F1 macro: {f1_mean:.4f} | ROC-AUC: {roc_mean:.4f}"
                )

                if f1_mean > melhor_score:
                    melhor_score = f1_mean
                    melhor_clf = clf
                    melhor_nome = nome
                    melhor_metricas = {
                        "f1_macro_mean": f1_mean,
                        "roc_auc_mean": roc_mean,
                    }
            except Exception as exc:
                logger.warning(f"  {nome}: falhou na comparação — {exc}")

        logger.info(f"Melhor modelo: {melhor_nome} (F1 macro CV: {melhor_score:.4f})")
        logger.info("=" * 55)

        # Retreinar o melhor nos dados de treino completos
        melhor_pipeline = self.build_pipeline(classifier=melhor_clf)
        melhor_pipeline.fit(X_train[self.feature_names], y_train)

        return melhor_pipeline, melhor_nome, melhor_metricas

    def save_model(
        self,
        pipeline: Pipeline,
        path: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Serializa o pipeline treinado em disco usando joblib.

        Cria o diretório de destino automaticamente caso não exista.
        Se metadados forem fornecidos, os salva no caminho definido
        em self.metadata_path.

        Args:
            pipeline: Pipeline sklearn treinado.
            path: Caminho de saída para o modelo (.joblib).
            metadata: Dicionário de metadados opcionais a serializar.
        """
        ensure_dir("app/model")
        save_artifact(pipeline, path)

        if metadata is not None:
            save_artifact(metadata, self.metadata_path)
            logger.info(f"Metadados salvos em: {self.metadata_path}")

    def run(self, data_path: str) -> None:
        """
        Orquestra o fluxo completo de treinamento do modelo.

        Etapas executadas:
        1. Carregamento e pré-processamento dos dados brutos.
        2. Feature engineering (features compostas, temporais e de interação).
        3. Divisão estratificada treino/teste (80/20).
        4. Comparação de modelos (RF, XGBoost, LightGBM) via CV.
        5. Validação cruzada completa do melhor modelo.
        6. Avaliação no conjunto de teste e geração de relatório.
        7. Serialização do modelo e metadados em app/model/.

        Args:
            data_path: Caminho para o arquivo CSV com os dados brutos.
        """
        logger.info("=" * 60)
        logger.info("PIPELINE DE TREINAMENTO — Passos Mágicos")
        logger.info("=" * 60)

        # ── 1. Pré-processamento ────────────────────────────────────────
        preprocessador = DataPreprocessor()
        df_raw = preprocessador.load_data(data_path)
        df_clean = preprocessador.fit_transform(df_raw)

        # ── 2. Feature engineering ──────────────────────────────────────
        engenheiro = FeatureEngineer()
        df_features = engenheiro.transform(df_clean)

        # ── 3. Divisão treino/teste ─────────────────────────────────────
        X_train, X_test, y_train, y_test = preprocessador.split_data(df_features)
        self._filter_features(X_train)

        logger.info(
            f"Dados prontos — treino: {len(X_train)} | teste: {len(X_test)} | "
            f"features: {len(self.feature_names)}"
        )

        # ── 4. Comparar modelos ─────────────────────────────────────────
        best_pipeline, best_algo, _ = self._compare_models(X_train, y_train)

        # ── 5. Validação cruzada completa do melhor classificador ───────
        best_clf = best_pipeline.named_steps["classifier"]
        cv_scores = self.cross_validate(
            X_train,
            y_train,
            pipeline=self.build_pipeline(classifier=best_clf),
        )

        # ── 6. Avaliação no conjunto de teste ───────────────────────────
        avaliador = ModelEvaluator()
        ensure_dir("reports/figures")

        y_pred = best_pipeline.predict(X_test[self.feature_names])
        y_proba = best_pipeline.predict_proba(X_test[self.feature_names])[:, 1]

        metrics = avaliador.compute_metrics(y_test, y_pred, y_proba)
        avaliador.plot_confusion_matrix(
            y_test, y_pred, "reports/figures/confusion_matrix.png"
        )
        avaliador.plot_roc_curve(y_test, y_proba, "reports/figures/roc_curve.png")
        if hasattr(best_clf, "feature_importances_"):
            # Obter nomes das features pós-encoding (ColumnTransformer expande categóricas)
            try:
                encoded_feature_names = list(
                    best_pipeline.named_steps["preprocessor"].get_feature_names_out()
                )
            except Exception:
                encoded_feature_names = self.feature_names
            avaliador.plot_feature_importance(
                best_clf,
                encoded_feature_names,
                "reports/figures/feature_importance.png",
            )

        ensure_dir("reports")
        report_path = avaliador.generate_report(
            metrics, "reports/evaluation_report.txt"
        )

        confiavel = avaliador.is_model_reliable(metrics)
        status = (
            "APROVADO para produção"
            if confiavel
            else "REPROVADO — métricas abaixo do mínimo"
        )
        logger.info(f"Confiabilidade do modelo: {status}")

        # ── 7. Serializar modelo e metadados ────────────────────────────
        metadata = {
            "version": "1.0.0",
            "trained_at": datetime.now().isoformat(),
            "algorithm": best_algo,
            "features": self.feature_names,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "cv_scores": cv_scores,
            "test_metrics": metrics,
            "model_reliable": confiavel,
        }
        self.save_model(best_pipeline, self.model_path, metadata=metadata)

        logger.info("=" * 60)
        logger.info(f"Modelo salvo em    : {self.model_path}")
        logger.info(f"Metadados salvos em: {self.metadata_path}")
        logger.info(f"Relatório salvo em : {report_path}")
        logger.info(f"Status             : {status}")
        logger.info("=" * 60)


# ── Entrypoint CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treina o modelo preditivo de defasagem escolar — Passos Mágicos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        metavar="CAMINHO_CSV",
        help="Caminho para o arquivo CSV com os dados brutos",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="app/model/model.joblib",
        metavar="CAMINHO",
        help="Caminho de saída do modelo serializado",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="app/model/metadata.joblib",
        metavar="CAMINHO",
        help="Caminho de saída dos metadados do modelo",
    )
    args = parser.parse_args()

    config = load_config()
    config["model_path"] = args.model_path
    config["metadata_path"] = args.metadata_path

    trainer = ModelTrainer(config=config)
    trainer.run(data_path=args.data)
