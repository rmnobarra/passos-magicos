"""
Middleware de logging da API Passos Mágicos.

Registra informações de cada requisição HTTP recebida: método, URL,
código de status, IP do cliente e tempo de processamento.
"""

import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware que registra log estruturado para cada requisição HTTP.

    Captura: método, caminho, IP do cliente, status de resposta e
    duração total em milissegundos.
    """

    def __init__(self, app: ASGIApp) -> None:
        """
        Inicializa o middleware com a aplicação ASGI.

        Args:
            app: Aplicação ASGI a ser envolvida pelo middleware.
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Intercepta a requisição, registra log e encaminha ao próximo handler.

        Args:
            request: Objeto de requisição HTTP.
            call_next: Função que invoca o próximo middleware ou handler.

        Returns:
            Resposta HTTP gerada pelo handler.
        """
        inicio = time.perf_counter()

        client_ip = request.client.host if request.client else "desconhecido"
        metodo = request.method
        caminho = request.url.path

        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as exc:
            logger.error(
                f"Erro não tratado | {metodo} {caminho} | IP={client_ip} | erro={exc}"
            )
            raise

        duracao_ms = (time.perf_counter() - inicio) * 1000

        nivel = logging.INFO if status < 400 else logging.WARNING
        logger.log(
            nivel,
            f"{metodo} {caminho} | status={status} | "
            f"ip={client_ip} | duração={duracao_ms:.1f}ms",
        )

        return response
