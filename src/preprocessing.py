"""
Módulo de pré-processamento de dados do projeto Passos Mágicos.

Implementa a classe DataPreprocessor responsável por carregar, validar,
limpar e preparar os dados para o pipeline de treinamento.

O dataset original está em formato *wide* (uma linha por aluno, colunas
sufixadas por ano, ex: INDE_2022, FASE_2021). Este módulo converte
automaticamente para o formato *long* (uma linha por aluno/ano).
"""

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_logger

logger = get_logger(__name__)

# Colunas numéricas com indicadores educacionais (formato long)
COLUNAS_NUMERICAS: list[str] = [
    "INDE", "IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV",
]

# Colunas categóricas esperadas (formato long)
COLUNAS_CATEGORICAS: list[str] = ["FASE", "TURMA", "ANO", "PONTO_DE_VIRADA"]

# Schema mínimo esperado após reshape para formato long
SCHEMA_MINIMO: set[str] = set(COLUNAS_NUMERICAS) | {"FASE", "ANO"}


class DataPreprocessor:
    """
    Responsável pelo pré-processamento completo dos dados Passos Mágicos.

    Detecta automaticamente se o dataset está em formato wide (colunas por
    ano) e converte para long antes de validar, limpar e preparar os dados.
    """

    def __init__(self) -> None:
        """Inicializa o DataPreprocessor."""
        self._medianas_por_fase: Optional[pd.DataFrame] = None
        self._modas: Optional[pd.Series] = None

    # ── Carregamento ────────────────────────────────────────────────────────

    def load_data(self, path: str) -> pd.DataFrame:
        """
        Carrega o dataset a partir de um arquivo CSV.

        Tenta UTF-8 primeiro; caso falhe, utiliza latin-1.

        Args:
            path: Caminho para o arquivo CSV com os dados brutos.

        Returns:
            DataFrame com os dados carregados.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            ValueError: Se o arquivo estiver vazio.
        """
        logger.info(f"Carregando dados de: {path}")

        try:
            df = pd.read_csv(path, sep=";", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep=";", encoding="latin-1")

        if df.empty:
            raise ValueError(f"Arquivo vazio: {path}")

        logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
        return df

    # ── Detecção e reshape wide → long ──────────────────────────────────────

    def _is_wide_format(self, df: pd.DataFrame) -> bool:
        """
        Verifica se o DataFrame está no formato wide (colunas com sufixo de ano).

        Args:
            df: DataFrame a inspecionar.

        Returns:
            True se houver pelo menos uma coluna no padrão NOME_AAAA.
        """
        return any(re.search(r"_\d{4}$", col) for col in df.columns)

    def _reshape_wide_to_long(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte o dataset do formato wide para o formato long.

        Para cada ano detectado nas colunas (ex: 2021, 2022), cria um
        sub-DataFrame com as colunas padronizadas (INDE, IAN, FASE, etc.)
        e concatena todos os anos em um único DataFrame long.

        Mapeamentos especiais:
        - FASE_TURMA_{ano} → FASE (extrai apenas o número inicial)
        - PONTO_VIRADA_{ano} → PONTO_DE_VIRADA
        - DEFASAGEM_{ano} → DEFASAGEM (apenas quando presente)

        Args:
            df: DataFrame wide com colunas sufixadas por ano.

        Returns:
            DataFrame long com uma linha por aluno/ano.
        """
        # Detectar todos os anos presentes nas colunas
        anos = sorted({
            int(m.group(1))
            for col in df.columns
            for m in [re.search(r"_(\d{4})$", col)]
            if m
        })

        logger.info(f"Formato wide detectado — anos encontrados: {anos}")
        frames: list[pd.DataFrame] = []

        for ano in anos:
            s = f"_{ano}"

            # --- indicadores numéricos principais ---
            mapa: dict[str, str] = {}
            for indicador in COLUNAS_NUMERICAS:
                col = f"{indicador}{s}"
                if col in df.columns:
                    mapa[indicador] = col

            # Pular ano sem INDE (dados insuficientes)
            if "INDE" not in mapa:
                logger.info(f"Ano {ano}: coluna INDE ausente, ignorado")
                continue

            # --- FASE ---
            if f"FASE{s}" in df.columns:
                mapa["FASE"] = f"FASE{s}"
            elif f"FASE_TURMA{s}" in df.columns:
                # FASE_TURMA_2020 → extrai apenas o número leading (ex: "7G" → 7)
                mapa["FASE"] = f"FASE_TURMA{s}"

            # --- TURMA ---
            if f"TURMA{s}" in df.columns:
                mapa["TURMA"] = f"TURMA{s}"

            # --- PONTO_DE_VIRADA ---
            for candidato in [f"PONTO_VIRADA{s}", f"PONTO_DE_VIRADA{s}"]:
                if candidato in df.columns:
                    mapa["PONTO_DE_VIRADA"] = candidato
                    break

            # --- DEFASAGEM (apenas quando existir para este ano) ---
            if f"DEFASAGEM{s}" in df.columns:
                mapa["DEFASAGEM"] = f"DEFASAGEM{s}"

            # Construir sub-DataFrame para este ano
            subset: dict[str, object] = {}

            if "NOME" in df.columns:
                subset["NOME"] = df["NOME"].values

            subset["ANO"] = ano

            for nome_padrao, nome_col in mapa.items():
                coluna = df[nome_col]
                if nome_padrao == "FASE" and nome_col.startswith("FASE_TURMA"):
                    # Extrair parte numérica: "7G" → 7, "8" → 8
                    subset["FASE"] = pd.to_numeric(
                        coluna.astype(str).str.extract(r"^(\d+)")[0],
                        errors="coerce",
                    ).values
                else:
                    subset[nome_padrao] = coluna.values

            frames.append(pd.DataFrame(subset))
            logger.info(
                f"Ano {ano}: {len(df)} registros extraídos "
                f"({list(mapa.keys())})"
            )

        if not frames:
            logger.warning("Nenhum ano processado — retornando DataFrame original")
            return df

        resultado = pd.concat(frames, ignore_index=True)

        # Garantir que indicadores numéricos estejam como float
        # (CSV pode ter lido valores como strings por separadores mistos)
        for col in COLUNAS_NUMERICAS:
            if col in resultado.columns:
                resultado[col] = pd.to_numeric(resultado[col], errors="coerce")

        # FASE também deve ser numérica
        if "FASE" in resultado.columns:
            resultado["FASE"] = pd.to_numeric(resultado["FASE"], errors="coerce")

        logger.info(
            f"Reshape concluído: {len(df)} linhas × {len(df.columns)} colunas "
            f"→ {len(resultado)} linhas × {len(resultado.columns)} colunas"
        )
        return resultado

    # ── Validação ───────────────────────────────────────────────────────────

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Valida se o DataFrame contém as colunas mínimas esperadas.

        Args:
            df: DataFrame a ser validado (deve estar em formato long).

        Returns:
            True se o schema for válido.

        Raises:
            ValueError: Se colunas obrigatórias estiverem ausentes.
        """
        logger.info("Validando schema do dataset")

        colunas_presentes = set(df.columns.str.upper())
        colunas_faltantes = SCHEMA_MINIMO - colunas_presentes

        if colunas_faltantes:
            raise ValueError(
                f"Colunas obrigatórias ausentes: {colunas_faltantes}"
            )

        logger.info("Schema validado com sucesso")
        return True

    # ── Limpeza ─────────────────────────────────────────────────────────────

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trata valores ausentes seguindo as regras de negócio do projeto.

        Regras:
        - Remove linhas com mais de 50% de valores ausentes.
        - Indicadores numéricos (INDE, IAN, IDA, etc.): imputar com mediana por FASE.
        - Colunas categóricas: imputar com moda.

        Args:
            df: DataFrame com possíveis valores ausentes.

        Returns:
            Novo DataFrame com valores ausentes tratados.
        """
        logger.info("Iniciando tratamento de valores ausentes")
        resultado = df.copy()

        # Remover linhas com mais de 50% de NaN
        limite = len(resultado.columns) * 0.5
        n_antes = len(resultado)
        resultado = resultado.dropna(thresh=int(limite) + 1)
        n_removidas = n_antes - len(resultado)
        if n_removidas > 0:
            logger.info(f"Removidas {n_removidas} linhas com >50% de valores ausentes")

        # Imputar indicadores numéricos com mediana por FASE
        colunas_num_presentes = [c for c in COLUNAS_NUMERICAS if c in resultado.columns]

        if "FASE" in resultado.columns and colunas_num_presentes:
            self._medianas_por_fase = resultado.groupby("FASE")[colunas_num_presentes].median()

            for col in colunas_num_presentes:
                nulos = resultado[col].isna().sum()
                if nulos > 0:
                    resultado[col] = resultado.groupby("FASE")[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                    # Fallback: mediana global para fases sem dados suficientes
                    resultado[col] = resultado[col].fillna(resultado[col].median())
                    logger.info(f"Coluna '{col}': {nulos} valores imputados pela mediana por FASE")
        else:
            for col in colunas_num_presentes:
                nulos = resultado[col].isna().sum()
                if nulos > 0:
                    resultado[col] = resultado[col].fillna(resultado[col].median())
                    logger.info(f"Coluna '{col}': {nulos} valores imputados pela mediana global")

        # Imputar colunas categóricas com moda
        colunas_cat_presentes = [c for c in COLUNAS_CATEGORICAS if c in resultado.columns]
        self._modas = pd.Series(dtype=object)

        for col in colunas_cat_presentes:
            nulos = resultado[col].isna().sum()
            if nulos > 0:
                moda = resultado[col].mode()
                if not moda.empty:
                    self._modas[col] = moda.iloc[0]
                    resultado[col] = resultado[col].fillna(moda.iloc[0])
                    logger.info(f"Coluna '{col}': {nulos} valores imputados pela moda")

        logger.info("Tratamento de valores ausentes concluído")
        return resultado

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove linhas duplicadas do DataFrame.

        Args:
            df: DataFrame original.

        Returns:
            DataFrame sem linhas duplicadas.
        """
        logger.info("Removendo duplicatas")
        resultado = df.copy()

        n_antes = len(resultado)
        resultado = resultado.drop_duplicates()
        n_removidas = n_antes - len(resultado)

        logger.info(f"Duplicatas removidas: {n_removidas}")
        return resultado

    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza os nomes das colunas para maiúsculas sem espaços extras.

        Args:
            df: DataFrame com nomes de colunas originais.

        Returns:
            DataFrame com nomes de colunas em maiúsculas e sem espaços.
        """
        logger.info("Normalizando nomes das colunas")
        resultado = df.copy()

        resultado.columns = (
            resultado.columns
            .str.strip()
            .str.upper()
            .str.replace(r"\s+", "_", regex=True)
        )

        return resultado

    # ── Target ──────────────────────────────────────────────────────────────

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Garante que a coluna target DEFASAGEM existe e é binária (0/1).

        Quatro cenários tratados:
        1. Coluna ausente → deriva de INDE < 5.0 OU IAN < 5.0.
        2. Coluna em texto (Sim/Não) → converte para 0/1.
        3. Coluna já binária (apenas 0 e 1) com possíveis NaN → usa valores
           reais onde disponíveis e deriva de INDE/IAN onde NaN.
        4. Coluna com valores inteiros não-binários (ex: -1, -2, 3 indicando
           anos de defasagem) → considera qualquer valor ≠ 0 como risco (1),
           exceto que se o dado vier do dataset real usa-se o sinal:
           negativo = atraso = em risco (1), zero/positivo = sem risco (0).
           Para linhas sem dado real, deriva de INDE/IAN.

        Args:
            df: DataFrame com os dados.

        Returns:
            DataFrame com a coluna DEFASAGEM garantida como inteiro 0/1.
        """
        logger.info("Codificando coluna target DEFASAGEM")
        resultado = df.copy()

        if "DEFASAGEM" not in resultado.columns:
            logger.info("Coluna DEFASAGEM ausente — derivando de INDE e IAN")
            resultado["DEFASAGEM"] = (
                (resultado["INDE"] < 5.0) | (resultado["IAN"] < 5.0)
            ).astype(int)

        else:
            col = resultado["DEFASAGEM"]

            # Normalizar texto (Sim/Não) para numérico
            if col.dtype == object:
                mapa_texto = {"sim": 1, "não": 0, "nao": 0, "yes": 1, "no": 0}
                resultado["DEFASAGEM"] = col.str.lower().map(mapa_texto)
            else:
                resultado["DEFASAGEM"] = pd.to_numeric(col, errors="coerce")

            # Verificar se os valores presentes já são binários (somente 0 e 1)
            valores_presentes = set(resultado["DEFASAGEM"].dropna().unique())
            ja_binario = valores_presentes.issubset({0, 1, 0.0, 1.0})

            if ja_binario:
                # Cenário 3: binário com possíveis NaN — derivar onde NaN
                mascara_nulo = resultado["DEFASAGEM"].isna()
                if mascara_nulo.any():
                    logger.info(
                        f"{mascara_nulo.sum()} linhas sem DEFASAGEM real — "
                        "derivando de INDE e IAN"
                    )
                    resultado.loc[mascara_nulo, "DEFASAGEM"] = (
                        (resultado.loc[mascara_nulo, "INDE"] < 5.0)
                        | (resultado.loc[mascara_nulo, "IAN"] < 5.0)
                    ).astype(int)
            else:
                # Cenário 4: valores inteiros representando anos de defasagem
                # (ex: -1 = 1 ano atrás, 0 = no nível, +1 = 1 ano à frente)
                # Convenção adotada: negativo = em risco (1), >= 0 = sem risco (0)
                logger.info(
                    f"Coluna DEFASAGEM não-binária detectada "
                    f"(valores únicos: {sorted(valores_presentes)}) — "
                    "binarizando: negativo → em risco (1), ≥ 0 → sem risco (0)"
                )
                # Usar dados reais onde disponíveis
                mascara_real = resultado["DEFASAGEM"].notna()
                resultado.loc[mascara_real, "DEFASAGEM"] = (
                    resultado.loc[mascara_real, "DEFASAGEM"] < 0
                ).astype(int)

                # Para linhas sem dado real, derivar de INDE e IAN
                mascara_nulo = resultado["DEFASAGEM"].isna()
                if mascara_nulo.any():
                    resultado.loc[mascara_nulo, "DEFASAGEM"] = (
                        (resultado.loc[mascara_nulo, "INDE"] < 5.0)
                        | (resultado.loc[mascara_nulo, "IAN"] < 5.0)
                    ).astype(int)

            resultado["DEFASAGEM"] = resultado["DEFASAGEM"].astype(int)

        distribuicao = resultado["DEFASAGEM"].value_counts().to_dict()
        logger.info(f"Distribuição do target: {distribuicao}")
        return resultado

    # ── Split ───────────────────────────────────────────────────────────────

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide o DataFrame em conjuntos de treino e teste com estratificação.

        Args:
            df: DataFrame completo com features e target.
            test_size: Proporção do conjunto de teste (padrão: 0.2).
            random_state: Semente para reprodutibilidade (padrão: 42).

        Returns:
            Tupla (X_train, X_test, y_train, y_test).

        Raises:
            ValueError: Se a coluna DEFASAGEM não existir no DataFrame.
        """
        logger.info(f"Dividindo dados — test_size={test_size}")

        if "DEFASAGEM" not in df.columns:
            raise ValueError(
                "Coluna DEFASAGEM não encontrada. Execute encode_target primeiro."
            )

        X = df.drop(columns=["DEFASAGEM"])
        y = df["DEFASAGEM"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        logger.info(
            f"Split concluído — treino: {len(X_train)} amostras, "
            f"teste: {len(X_test)} amostras"
        )
        return X_train, X_test, y_train, y_test

    # ── Pipeline principal ──────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orquestra todo o pipeline de pré-processamento.

        Executa em ordem:
        1. Normalização dos nomes das colunas
        2. Detecção e conversão wide → long (se necessário)
        3. Validação do schema
        4. Remoção de duplicatas
        5. Tratamento de valores ausentes
        6. Codificação do target

        Args:
            df: DataFrame bruto carregado do CSV.

        Returns:
            DataFrame limpo e pronto para o feature engineering.
        """
        logger.info("Iniciando pipeline de pré-processamento")

        resultado = self.normalize_column_names(df)

        # Reshape wide → long quando o dataset usa colunas sufixadas por ano
        if self._is_wide_format(resultado):
            resultado = self._reshape_wide_to_long(resultado)

        self.validate_schema(resultado)
        resultado = self.remove_duplicates(resultado)
        resultado = self.handle_missing_values(resultado)
        resultado = self.encode_target(resultado)

        logger.info(
            f"Pré-processamento concluído — {resultado.shape[0]} linhas, "
            f"{resultado.shape[1]} colunas"
        )
        return resultado
