"""
Módulo de utilitários gerais do projeto Passos Mágicos.

Fornece funções de logging configurado, carregamento de configurações,
persistência de artefatos e manipulação de diretórios.
"""

import logging
import os
from pathlib import Path
from typing import Any

import joblib
from dotenv import load_dotenv


def get_logger(name: str) -> logging.Logger:
    """
    Cria e retorna um logger configurado com formato padronizado.

    O logger utiliza o formato:
        %(asctime)s | %(levelname)s | %(name)s | %(message)s

    O nível de log é definido pela variável de ambiente LOG_LEVEL
    (padrão: INFO).

    Args:
        name: Nome do logger, normalmente __name__ do módulo chamador.

    Returns:
        Logger configurado e pronto para uso.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, log_level, logging.INFO))
    return logger


def load_config() -> dict:
    """
    Carrega variáveis de ambiente do arquivo .env e retorna como dicionário.

    Caso o arquivo .env não exista, utiliza as variáveis de ambiente já
    definidas no sistema.

    Returns:
        Dicionário com as variáveis de configuração relevantes ao projeto.
    """
    load_dotenv()

    config = {
        "app_env": os.getenv("APP_ENV", "development"),
        "log_level": os.getenv("LOG_LEVEL", "DEBUG"),
        "api_host": os.getenv("API_HOST", "0.0.0.0"),
        "api_port": int(os.getenv("API_PORT", "8000")),
        "model_path": os.getenv("MODEL_PATH", "app/model/model.joblib"),
        "metadata_path": os.getenv("METADATA_PATH", "app/model/metadata.joblib"),
        "data_path": os.getenv("DATA_PATH", "data/raw/"),
        "monitor_port": int(os.getenv("MONITOR_PORT", "8501")),
        "drift_threshold": float(os.getenv("DRIFT_THRESHOLD", "0.15")),
    }

    return config


def save_artifact(obj: Any, path: str) -> None:
    """
    Serializa e salva um artefato (modelo, preprocessador, etc.) usando joblib.

    Cria o diretório de destino automaticamente caso não exista.

    Args:
        obj: Objeto Python a ser serializado (modelo, pipeline, etc.).
        path: Caminho completo do arquivo de destino (ex: app/model/model.joblib).

    Raises:
        OSError: Se não for possível criar o diretório ou salvar o arquivo.
    """
    logger = get_logger(__name__)

    dest = Path(path)
    ensure_dir(str(dest.parent))

    logger.info(f"Salvando artefato em: {path}")
    joblib.dump(obj, path)
    logger.info(f"Artefato salvo com sucesso: {path}")


def load_artifact(path: str) -> Any:
    """
    Carrega um artefato serializado com joblib a partir do caminho informado.

    Args:
        path: Caminho completo do arquivo a ser carregado.

    Returns:
        Objeto deserializado (modelo, pipeline, etc.).

    Raises:
        FileNotFoundError: Se o arquivo não existir no caminho especificado.
        Exception: Se ocorrer erro durante a desserialização.
    """
    logger = get_logger(__name__)

    if not Path(path).exists():
        raise FileNotFoundError(f"Artefato não encontrado: {path}")

    logger.info(f"Carregando artefato de: {path}")
    obj = joblib.load(path)
    logger.info(f"Artefato carregado com sucesso: {path}")

    return obj


def ensure_dir(path: str) -> None:
    """
    Cria o diretório especificado (e os intermediários) caso não exista.

    Equivalente a `mkdir -p` no shell. Não gera erro se o diretório
    já existir.

    Args:
        path: Caminho do diretório a ser criado.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
