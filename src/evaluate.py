"""
Módulo de avaliação do modelo preditivo de defasagem escolar — Passos Mágicos.

Implementa a classe ModelEvaluator com cálculo de métricas, geração de
gráficos (matriz de confusão, curva ROC, importância de features) e
critério de confiabilidade para promoção à produção.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # backend não-interativo; deve ser definido antes de pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Responsável pela avaliação completa do modelo preditivo.

    Calcula métricas de classificação, gera visualizações diagnósticas
    e determina se o modelo atende aos critérios mínimos para produção.
    """

    def compute_metrics(
        self,
        y_true: Any,
        y_pred: Any,
        y_proba: Any,
    ) -> dict:
        """
        Calcula as métricas de avaliação do modelo no conjunto de teste.

        Métricas calculadas:
        - F1 macro e F1 weighted
        - ROC-AUC
        - Precision macro e Recall macro
        - Recall da classe positiva (defasagem=1)
        - Matriz de confusão (como lista aninhada)
        - Relatório completo de classificação

        Args:
            y_true: Valores reais do target (array-like).
            y_pred: Predições binárias do modelo (array-like).
            y_proba: Probabilidades previstas para a classe positiva (array-like).

        Returns:
            Dicionário com todas as métricas calculadas.
        """
        logger.info("Calculando métricas de avaliação")

        metrics = {
            "f1_macro": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "precision_macro": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall_positiva": float(
                recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, zero_division=0
            ),
        }

        logger.info(
            f"Métricas — F1 macro: {metrics['f1_macro']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"Recall positiva: {metrics['recall_positiva']:.4f}"
        )
        return metrics

    def plot_confusion_matrix(
        self,
        y_true: Any,
        y_pred: Any,
        save_path: str,
    ) -> None:
        """
        Gera e salva a matriz de confusão como imagem PNG.

        Args:
            y_true: Valores reais do target.
            y_pred: Predições binárias do modelo.
            save_path: Caminho completo para salvar a imagem (ex: reports/figures/cm.png).
        """
        logger.info(f"Gerando matriz de confusão em: {save_path}")
        ensure_dir(str(Path(save_path).parent))

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            display_labels=["Sem risco (0)", "Com risco (1)"],
            cmap="Blues",
            ax=ax,
        )
        ax.set_title("Matriz de Confusão — Defasagem Escolar", fontsize=13)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        logger.info("Matriz de confusão salva")

    def plot_roc_curve(
        self,
        y_true: Any,
        y_proba: Any,
        save_path: str,
    ) -> None:
        """
        Gera e salva a curva ROC como imagem PNG.

        Args:
            y_true: Valores reais do target.
            y_proba: Probabilidades previstas para a classe positiva.
            save_path: Caminho completo para salvar a imagem (ex: reports/figures/roc.png).
        """
        logger.info(f"Gerando curva ROC em: {save_path}")
        ensure_dir(str(Path(save_path).parent))

        auc = roc_auc_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(7, 5))
        RocCurveDisplay.from_predictions(
            y_true,
            y_proba,
            name=f"Modelo (AUC = {auc:.4f})",
            ax=ax,
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Aleatório (AUC = 0.50)")
        ax.set_title("Curva ROC — Defasagem Escolar", fontsize=13)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        logger.info("Curva ROC salva")

    def plot_feature_importance(
        self,
        model: Any,
        features: list[str],
        save_path: str,
        top_n: int = 20,
    ) -> None:
        """
        Gera e salva o gráfico de importância das features como imagem PNG.

        Compatível com modelos que expõem o atributo feature_importances_
        (RandomForest, XGBoost, LightGBM).

        Args:
            model: Classificador treinado com atributo feature_importances_.
            features: Lista de nomes das features na mesma ordem usada no treino.
            save_path: Caminho completo para salvar a imagem.
            top_n: Número máximo de features a exibir (padrão: 20).
        """
        logger.info(f"Gerando importância de features em: {save_path}")
        ensure_dir(str(Path(save_path).parent))

        if not hasattr(model, "feature_importances_"):
            logger.warning("Modelo não possui feature_importances_. Gráfico ignorado.")
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [features[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
        bars = ax.barh(
            range(len(top_features)), top_importances[::-1], color="steelblue"
        )
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1], fontsize=9)
        ax.set_xlabel("Importância (Gini)", fontsize=10)
        ax.set_title(f"Top {top_n} Features mais Relevantes", fontsize=13)
        ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        logger.info("Gráfico de importância de features salvo")

    def generate_report(self, metrics: dict, save_path: str) -> str:
        """
        Gera e salva um relatório textual com todas as métricas de avaliação.

        O relatório inclui as métricas numéricas, a matriz de confusão e
        o relatório detalhado por classe do sklearn.

        Args:
            metrics: Dicionário retornado por compute_metrics.
            save_path: Caminho completo para salvar o relatório (.txt).

        Returns:
            Caminho do arquivo de relatório salvo.
        """
        logger.info(f"Gerando relatório de avaliação em: {save_path}")
        ensure_dir(str(Path(save_path).parent))

        separador = "=" * 55
        linhas = [
            separador,
            "RELATÓRIO DE AVALIAÇÃO — Passos Mágicos",
            separador,
            "",
            "── Métricas Principais ──────────────────────────────",
            f"  F1 macro              : {metrics.get('f1_macro', 0):.4f}",
            f"  F1 weighted           : {metrics.get('f1_weighted', 0):.4f}",
            f"  ROC-AUC               : {metrics.get('roc_auc', 0):.4f}",
            f"  Precision macro       : {metrics.get('precision_macro', 0):.4f}",
            f"  Recall macro          : {metrics.get('recall_macro', 0):.4f}",
            f"  Recall classe positiva: {metrics.get('recall_positiva', 0):.4f}",
            "",
            "── Critérios de Produção ────────────────────────────",
            f"  F1 macro >= 0.70      : {'OK' if metrics.get('f1_macro', 0) >= 0.70 else 'REPROVADO'}",
            f"  ROC-AUC >= 0.75       : {'OK' if metrics.get('roc_auc', 0) >= 0.75 else 'REPROVADO'}",
            f"  Recall positiva >= 0.65: {'OK' if metrics.get('recall_positiva', 0) >= 0.65 else 'REPROVADO'}",
            "",
            "── Matriz de Confusão ───────────────────────────────",
        ]

        cm = metrics.get("confusion_matrix", [])
        if cm:
            linhas.append(f"  {'':20s}  Pred 0   Pred 1")
            for i, row in enumerate(cm):
                linhas.append(f"  Real {i}              {row[0]:>6}   {row[1]:>6}")

        linhas += [
            "",
            "── Relatório por Classe (sklearn) ───────────────────",
        ]
        cls_report = metrics.get("classification_report", "")
        if cls_report:
            linhas += [f"  {linha}" for linha in cls_report.splitlines()]

        linhas += ["", separador]

        conteudo = "\n".join(linhas)
        Path(save_path).write_text(conteudo, encoding="utf-8")

        # Imprimir também no log
        for linha in linhas:
            logger.info(linha)

        return save_path

    def is_model_reliable(self, metrics: dict) -> bool:
        """
        Verifica se o modelo atende aos critérios mínimos para produção.

        Critérios:
        - F1-Score macro >= 0.70 (balanceia precision e recall)
        - ROC-AUC >= 0.75 (capacidade discriminatória)
        - Recall da classe positiva >= 0.65 (custo alto de falso negativo:
          deixar aluno em risco sem intervenção)

        Args:
            metrics: Dicionário retornado por compute_metrics.

        Returns:
            True se todos os critérios forem atendidos, False caso contrário.
        """
        f1 = metrics.get("f1_macro", 0.0)
        auc = metrics.get("roc_auc", 0.0)
        recall_pos = metrics.get("recall_positiva", 0.0)

        aprovado = f1 >= 0.70 and auc >= 0.75 and recall_pos >= 0.65

        logger.info(
            f"Critérios de confiabilidade — "
            f"F1 macro: {f1:.4f} ({'OK' if f1 >= 0.70 else 'FALHOU'}) | "
            f"ROC-AUC: {auc:.4f} ({'OK' if auc >= 0.75 else 'FALHOU'}) | "
            f"Recall positiva: {recall_pos:.4f} ({'OK' if recall_pos >= 0.65 else 'FALHOU'})"
        )
        return aprovado
