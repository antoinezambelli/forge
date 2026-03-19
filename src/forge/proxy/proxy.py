"""ProxyServer — programmatic API for the forge proxy.

Two modes:
- Managed: forge starts and manages the backend via ServerManager.
- External: user manages the backend, proxy connects to it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

from forge.clients.base import LLMClient
from forge.clients.llamafile import LlamafileClient
from forge.clients.ollama import OllamaClient
from forge.context.manager import ContextManager
from forge.context.strategies import TieredCompact
from forge.proxy.server import HTTPServer
from forge.server import BudgetMode, ServerManager

logger = logging.getLogger("forge.proxy")


class ProxyServer:
    """OpenAI-compatible proxy that applies forge guardrails transparently.

    Managed mode — forge starts the backend::

        proxy = ProxyServer(backend="llamaserver", gguf="model.gguf")
        proxy.start()   # starts llama-server on :8080 + proxy on :8081
        proxy.stop()    # stops both

    External mode — user manages the backend::

        proxy = ProxyServer(backend_url="http://localhost:8080")
        proxy.start()   # starts proxy on :8081 only
        proxy.stop()

    """

    def __init__(
        self,
        # External mode
        backend_url: str | None = None,
        # Managed mode
        backend: str | None = None,
        model: str | None = None,
        gguf: str | Path | None = None,
        backend_port: int = 8080,
        budget_mode: BudgetMode = BudgetMode.BACKEND,
        budget_tokens: int | None = None,
        extra_flags: list[str] | None = None,
        # Proxy settings
        host: str = "127.0.0.1",
        port: int = 8081,
        serialize: bool | None = None,
        max_retries: int = 3,
        rescue_enabled: bool = True,
    ) -> None:
        """
        Args:
            backend_url: URL of an externally managed backend (external mode).
            backend: Backend type — "llamaserver" or "ollama" (managed mode).
            model: Model name (managed mode, required for ollama).
            gguf: Path to GGUF file (managed mode, llamaserver/llamafile).
            backend_port: Port for the managed backend (default 8080).
            budget_mode: How to determine context budget.
            budget_tokens: Explicit token budget (for MANUAL mode).
            extra_flags: Additional CLI flags for the managed backend.
            host: Proxy listen host.
            port: Proxy listen port.
            serialize: Serialize requests via lock. None = auto (True for
                managed, False for external).
            max_retries: Max consecutive retries for bad LLM responses.
            rescue_enabled: Attempt rescue parsing of text responses.
        """
        if backend_url is None and backend is None:
            raise ValueError("Provide either backend_url (external) or backend (managed)")

        self._backend_url = backend_url
        self._backend = backend
        self._model = model
        self._gguf = gguf
        self._backend_port = backend_port
        self._budget_mode = budget_mode
        self._budget_tokens = budget_tokens
        self._extra_flags = extra_flags
        self._host = host
        self._port = port
        self._max_retries = max_retries
        self._rescue_enabled = rescue_enabled

        # Auto-detect serialization: managed = single GPU = serialize
        if serialize is None:
            self._serialize = backend is not None
        else:
            self._serialize = serialize

        self._server_manager: ServerManager | None = None
        self._http_server: HTTPServer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = False

    @property
    def url(self) -> str:
        """The proxy's base URL."""
        return f"http://{self._host}:{self._port}"

    def start(self) -> None:
        """Start the proxy (and managed backend if applicable).

        Blocks until the proxy is ready to accept connections.
        """
        if self._started:
            return

        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop, args=(ready,), daemon=True,
        )
        self._thread.start()
        ready.wait(timeout=120)

        if not self._started:
            raise RuntimeError("Proxy failed to start")

        logger.info("Proxy ready at %s", self.url)

    def stop(self) -> None:
        """Stop the proxy (and managed backend if applicable)."""
        if not self._started or self._loop is None:
            return

        asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop).result(timeout=30)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._started = False
        logger.info("Proxy stopped")

    def _run_loop(self, ready: threading.Event) -> None:
        """Event loop thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_start(ready))
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def _async_start(self, ready: threading.Event) -> None:
        """Async startup: backend + HTTP server."""
        client: LLMClient
        context_manager: ContextManager

        if self._backend_url is not None:
            # External mode — connect to existing backend
            # LlamafileClient expects base_url with /v1 suffix
            base = self._backend_url.rstrip("/")
            if not base.endswith("/v1"):
                base = base + "/v1"
            client = LlamafileClient(
                model="default",
                base_url=base,
                mode="native",
            )
            if self._budget_tokens is not None:
                budget = self._budget_tokens
            else:
                # Try to auto-detect from backend /props
                ctx_len = await client.get_context_length()
                budget = ctx_len if ctx_len is not None else 8192
            context_manager = ContextManager(
                strategy=TieredCompact(),
                budget_tokens=budget,
            )
        else:
            # Managed mode
            assert self._backend is not None
            if self._backend == "ollama":
                assert self._model is not None
                client = OllamaClient(model=self._model)
            else:
                client = LlamafileClient(
                    model=self._model or "default",
                    base_url=f"http://localhost:{self._backend_port}/v1",
                    mode="native",
                )

            server_mgr = ServerManager(
                backend=self._backend,
                port=self._backend_port,
            )
            self._server_manager = server_mgr

            budget = await server_mgr.start_with_budget(
                model=self._model or "",
                gguf_path=self._gguf or "",
                mode="native",
                budget_mode=self._budget_mode,
                manual_tokens=self._budget_tokens,
                extra_flags=self._extra_flags,
            )

            if self._backend == "ollama" and hasattr(client, "set_num_ctx"):
                client.set_num_ctx(budget)

            context_manager = ContextManager(
                strategy=TieredCompact(),
                budget_tokens=budget,
            )

        self._http_server = HTTPServer(
            client=client,
            context_manager=context_manager,
            host=self._host,
            port=self._port,
            serialize_requests=self._serialize,
            max_retries=self._max_retries,
            rescue_enabled=self._rescue_enabled,
        )
        await self._http_server.start()
        self._started = True
        ready.set()

    async def _async_stop(self) -> None:
        """Async shutdown."""
        if self._http_server is not None:
            await self._http_server.stop()
        if self._server_manager is not None:
            await self._server_manager.stop()
