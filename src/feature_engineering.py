"""
Módulo de feature engineering do projeto Passos Mágicos.

Implementa a classe FeatureEngineer responsável por criar features
compostas, temporais e de interação a partir dos dados pré-processados,
além de selecionar as features mais relevantes para o modelo.
"""

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

# Colunas de indicadores psicossociais
INDICADORES_PSICOSSOCIAIS: list[str] = ["IPS", "IPP", "IPV"]

# Colunas de indicadores de performance
INDICADORES_PERFORMANCE: list[str] = ["IDA", "IEG", "IAA"]

# Todas as colunas de indicadores educacionais
TODOS_INDICADORES: list[str] = ["IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV"]

# Features numéricas base para o modelo
FEATURES_BASE: list[str] = [
    "INDE", "IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "FASE", "ANO",
]


class FeatureEngineer:
    """
    Responsável pela criação e seleção de features para o modelo preditivo.

    Cria features compostas a partir dos indicadores educacionais, features
    temporais baseadas no histórico do aluno e features de interação entre
    variáveis relevantes.
    """

    def create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features compostas a partir dos indicadores educacionais.

        Features criadas:
        - INDICE_BEMESTAR: média dos indicadores psicossociais (IPS, IPP, IPV).
        - INDICE_PERFORMANCE: média dos indicadores de desempenho (IDA, IEG, IAA).
        - GAP_AUTO_REAL: diferença entre autoavaliação (IAA) e desempenho real (IDA).
        - ABAIXO_MEDIA_GERAL: flag indicando aluno abaixo da mediana em todos os indicadores.

        Args:
            df: DataFrame pré-processado com as colunas de indicadores.

        Returns:
            Novo DataFrame com as features compostas adicionadas.
        """
        logger.info("Criando features compostas")
        resultado = df.copy()

        # Índice de bem-estar geral
        cols_bemestar = [c for c in INDICADORES_PSICOSSOCIAIS if c in resultado.columns]
        if cols_bemestar:
            resultado["INDICE_BEMESTAR"] = resultado[cols_bemestar].mean(axis=1)
            logger.info(f"Feature INDICE_BEMESTAR criada com colunas: {cols_bemestar}")

        # Índice de performance acadêmica
        cols_performance = [c for c in INDICADORES_PERFORMANCE if c in resultado.columns]
        if cols_performance:
            resultado["INDICE_PERFORMANCE"] = resultado[cols_performance].mean(axis=1)
            logger.info(f"Feature INDICE_PERFORMANCE criada com colunas: {cols_performance}")

        # Gap entre autoavaliação e desempenho real
        if "IAA" in resultado.columns and "IDA" in resultado.columns:
            resultado["GAP_AUTO_REAL"] = resultado["IAA"] - resultado["IDA"]
            logger.info("Feature GAP_AUTO_REAL criada")

        # Flag: estudante abaixo da mediana em todos os indicadores
        cols_indicadores = [c for c in TODOS_INDICADORES if c in resultado.columns]
        if cols_indicadores:
            medianas = resultado[cols_indicadores].median()
            resultado["ABAIXO_MEDIA_GERAL"] = (
                resultado[cols_indicadores] < medianas
            ).all(axis=1).astype(int)
            logger.info("Feature ABAIXO_MEDIA_GERAL criada")

        logger.info("Features compostas concluídas")
        return resultado

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features temporais baseadas na evolução do aluno ao longo dos anos.

        Requer que o dataset contenha múltiplos registros anuais por aluno,
        identificados pela coluna NOME ou similar.

        Features criadas:
        - EVOLUCAO_INDE: variação do INDE em relação ao ano anterior do mesmo aluno.

        Args:
            df: DataFrame com coluna ANO e identificador de aluno.

        Returns:
            Novo DataFrame com as features temporais adicionadas.
        """
        logger.info("Criando features temporais")
        resultado = df.copy()

        coluna_aluno = next(
            (c for c in ["NOME", "ALUNO", "ID_ALUNO", "ESTUDANTE"] if c in resultado.columns),
            None,
        )

        if coluna_aluno and "INDE" in resultado.columns and "ANO" in resultado.columns:
            resultado = resultado.sort_values([coluna_aluno, "ANO"])
            resultado["EVOLUCAO_INDE"] = (
                resultado.groupby(coluna_aluno)["INDE"]
                .diff()
                .fillna(0)
            )
            logger.info(
                f"Feature EVOLUCAO_INDE criada usando agrupamento por '{coluna_aluno}'"
            )
        else:
            resultado["EVOLUCAO_INDE"] = 0.0
            logger.info(
                "Coluna de identificação do aluno não encontrada — "
                "EVOLUCAO_INDE definida como 0"
            )

        logger.info("Features temporais concluídas")
        return resultado

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de interação entre variáveis relevantes para o modelo.

        Features criadas:
        - INDE_x_FASE: produto entre INDE e FASE (captura adequação por nível).
        - BEMESTAR_x_PERFORMANCE: produto entre bem-estar e performance (se disponíveis).

        Args:
            df: DataFrame com as features base e compostas já criadas.

        Returns:
            Novo DataFrame com as features de interação adicionadas.
        """
        logger.info("Criando features de interação")
        resultado = df.copy()

        # Interação entre INDE e FASE
        if "INDE" in resultado.columns and "FASE" in resultado.columns:
            resultado["INDE_x_FASE"] = resultado["INDE"] * resultado["FASE"]
            logger.info("Feature INDE_x_FASE criada")

        # Interação entre índice de bem-estar e performance
        if "INDICE_BEMESTAR" in resultado.columns and "INDICE_PERFORMANCE" in resultado.columns:
            resultado["BEMESTAR_x_PERFORMANCE"] = (
                resultado["INDICE_BEMESTAR"] * resultado["INDICE_PERFORMANCE"]
            )
            logger.info("Feature BEMESTAR_x_PERFORMANCE criada")

        logger.info("Features de interação concluídas")
        return resultado

    def select_features(self, df: pd.DataFrame) -> list[str]:
        """
        Seleciona as features relevantes presentes no DataFrame para o modelo.

        Combina as features base com as features criadas pelo engenheiro,
        excluindo colunas de identificação, target e colunas de texto.

        Args:
            df: DataFrame com todas as features (base + engineered).

        Returns:
            Lista de nomes de colunas selecionadas para treinamento.
        """
        logger.info("Selecionando features para o modelo")

        colunas_excluir = {
            "DEFASAGEM", "NOME", "ALUNO", "ID_ALUNO", "ESTUDANTE",
            "TURMA", "PONTO_DE_VIRADA",
        }

        # Features base numéricas
        features_candidatas = [c for c in FEATURES_BASE if c in df.columns]

        # Features engineered
        features_engineered = [
            "INDICE_BEMESTAR", "INDICE_PERFORMANCE", "GAP_AUTO_REAL",
            "ABAIXO_MEDIA_GERAL", "EVOLUCAO_INDE",
            "INDE_x_FASE", "BEMESTAR_x_PERFORMANCE",
        ]
        features_candidatas += [c for c in features_engineered if c in df.columns]

        # Remover colunas excluídas e garantir que são numéricas
        features_selecionadas = [
            c for c in features_candidatas
            if c not in colunas_excluir
            and c in df.columns
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        logger.info(f"Features selecionadas ({len(features_selecionadas)}): {features_selecionadas}")
        return features_selecionadas

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orquestra todo o pipeline de feature engineering.

        Executa em ordem:
        1. Features compostas (índices derivados dos indicadores)
        2. Features temporais (evolução entre anos)
        3. Features de interação (produtos entre variáveis)

        Args:
            df: DataFrame pré-processado e pronto para feature engineering.

        Returns:
            DataFrame enriquecido com todas as features criadas.
        """
        logger.info("Iniciando pipeline de feature engineering")

        resultado = self.create_composite_features(df)
        resultado = self.create_temporal_features(resultado)
        resultado = self.create_interaction_features(resultado)

        n_features_novas = resultado.shape[1] - df.shape[1]
        logger.info(
            f"Feature engineering concluído — "
            f"{n_features_novas} novas features criadas, "
            f"total: {resultado.shape[1]} colunas"
        )
        return resultado
