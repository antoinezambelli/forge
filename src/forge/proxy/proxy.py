"""ProxyServer — programmatic API for the forge proxy.

Two modes (orthogonal to ``backend``):

- Managed: forge starts and manages the backend via ServerManager.
- External: user manages the backend, proxy connects to it via URL.
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
from forge.clients.vllm import VLLMClient
from forge.context.manager import ContextManager
from forge.context.strategies import TieredCompact
from forge.proxy.server import HTTPServer
from forge.server import BudgetMode, ServerManager, setup_backend

logger = logging.getLogger("forge.proxy")

_VALID_BACKENDS = ("llamaserver", "llamafile", "ollama", "vllm")
_EXTERNAL_BACKENDS = ("llamaserver", "llamafile", "vllm")


class ProxyServer:
    """OpenAI-compatible proxy that applies forge guardrails transparently.

    ``backend`` is required; mode is determined by which other params are set.

    Managed mode — forge starts the backend::

        ProxyServer(backend="llamaserver", gguf="model.gguf")
        ProxyServer(backend="vllm",        model_path="/path/to/awq-dir")
        ProxyServer(backend="ollama",      model="ministral-3:14b")

    External mode — user manages the backend::

        ProxyServer(backend="llamaserver", url="http://localhost:8080")
        ProxyServer(backend="vllm",        url="http://localhost:8000")

    Ollama is rejected in external mode (use OllamaClient directly).
    """

    def __init__(
        self,
        *,
        backend: str,
        # External mode
        url: str | None = None,
        # Managed mode — identity (one of, depending on backend)
        model: str | None = None,
        gguf: str | Path | None = None,
        model_path: str | Path | None = None,
        # Managed mode — server config
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
            backend: Backend type — one of ``"llamaserver"``, ``"llamafile"``,
                ``"ollama"``, ``"vllm"``. Required.
            url: URL of an externally managed backend (external mode).
                Ollama is rejected in external mode.
            model: Ollama model name (managed mode, ollama only).
            gguf: Path to GGUF file (managed mode, llamaserver/llamafile only).
            model_path: Path to model directory or HF repo id (managed mode,
                vllm only).
            backend_port: Port for the managed backend (default 8080).
            budget_mode: How to determine context budget (managed mode).
            budget_tokens: Explicit token budget. Required in external mode if
                the backend doesn't report its context length.
            extra_flags: Additional CLI flags for the managed backend.
            host: Proxy listen host.
            port: Proxy listen port.
            serialize: Serialize requests via lock. None = auto (True for
                managed, False for external).
            max_retries: Max consecutive retries for bad LLM responses.
            rescue_enabled: Attempt rescue parsing of text responses.

        Raises:
            ValueError: backend is invalid, mode is ambiguous (both url and
                an identity field set, or neither set), or the identity field
                doesn't match the backend (e.g. ``backend="vllm"`` with
                ``gguf=...``).
        """
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {_VALID_BACKENDS}, got {backend!r}",
            )

        # Mode = external iff url is set. Identity fields must be absent.
        identity_set = sum(x is not None for x in (model, gguf, model_path))
        if url is not None:
            if backend not in _EXTERNAL_BACKENDS:
                raise ValueError(
                    f"backend={backend!r} is not supported in external mode "
                    f"(supported: {_EXTERNAL_BACKENDS})",
                )
            if identity_set > 0:
                raise ValueError(
                    "external mode (url=...) does not accept identity fields "
                    "(model, gguf, model_path)",
                )
        else:
            if identity_set != 1:
                raise ValueError(
                    "managed mode requires exactly one identity field "
                    "(model, gguf, or model_path)",
                )
            if backend == "ollama" and model is None:
                raise ValueError("backend='ollama' requires model")
            if backend in ("llamaserver", "llamafile") and gguf is None:
                raise ValueError(
                    f"backend={backend!r} requires gguf",
                )
            if backend == "vllm" and model_path is None:
                raise ValueError("backend='vllm' requires model_path")

        self._backend = backend
        self._url = url
        self._model = model
        self._gguf = gguf
        self._model_path = model_path
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
            self._serialize = url is None
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
        if self._url is not None:
            client, context_manager = await self._setup_external()
        else:
            client, context_manager = await self._setup_managed()

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

    async def _setup_external(self) -> tuple[LLMClient, ContextManager]:
        """External mode: connect to caller-managed backend."""
        assert self._url is not None
        base = self._url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"

        client: LLMClient
        if self._backend in ("llamaserver", "llamafile"):
            # Caller manages the backend, so we don't have a GGUF path.
            # "default" is a placeholder identity for the wire model field
            # (llama-server ignores it) and JSONL model field.
            client = LlamafileClient(
                gguf_path="default",
                base_url=base,
                mode="native",
            )
        elif self._backend == "vllm":
            client = VLLMClient(
                model_path="default",
                base_url=base,
            )
            # Unlike llama.cpp, vLLM validates the wire `model` field against
            # its --served-model-name aliases (404 on mismatch). External mode
            # has no model path to send, so discover the served identity from
            # /v1/models instead of shipping the "default" placeholder.
            served = await client.get_served_model_name()
            if served:
                logger.info("Discovered vLLM served model name: %s", served)
                client.model_path = served
                client.model = served
            else:
                logger.warning(
                    "Could not discover a served model name from %s/models; "
                    "sending placeholder 'default' (vLLM will 404 if it "
                    "validates the model field)",
                    base,
                )
        else:
            # Should be unreachable per __init__ validation
            raise ValueError(f"backend={self._backend!r} not valid in external mode")

        if self._budget_tokens is not None:
            budget = self._budget_tokens
        else:
            ctx_len = await client.get_context_length()
            if ctx_len is None:
                raise RuntimeError(
                    f"backend at {self._url} did not report a context length; "
                    "pass budget_tokens explicitly",
                )
            budget = ctx_len

        context_manager = ContextManager(
            strategy=TieredCompact(),
            budget_tokens=budget,
        )
        return client, context_manager

    async def _setup_managed(self) -> tuple[LLMClient, ContextManager]:
        """Managed mode: forge starts the backend via setup_backend."""
        client = self._build_managed_client()

        server, context_manager = await setup_backend(
            backend=self._backend,
            model=self._model,
            gguf_path=self._gguf,
            model_path=self._model_path,
            mode="native",
            budget_mode=self._budget_mode,
            manual_tokens=self._budget_tokens,
            client=client,
            port=self._backend_port,
            extra_flags=self._extra_flags,
        )
        self._server_manager = server
        return client, context_manager

    def _build_managed_client(self) -> LLMClient:
        """Construct the right client for the managed backend."""
        base_url = f"http://localhost:{self._backend_port}/v1"
        if self._backend == "ollama":
            assert self._model is not None
            return OllamaClient(model=self._model)
        if self._backend in ("llamaserver", "llamafile"):
            assert self._gguf is not None
            return LlamafileClient(
                gguf_path=self._gguf,
                base_url=base_url,
                mode="native",
            )
        if self._backend == "vllm":
            assert self._model_path is not None
            return VLLMClient(
                model_path=self._model_path,
                base_url=base_url,
            )
        # Unreachable per __init__ validation
        raise ValueError(f"unsupported backend: {self._backend!r}")

    async def _async_stop(self) -> None:
        """Async shutdown."""
        if self._http_server is not None:
            await self._http_server.stop()
        if self._server_manager is not None:
            await self._server_manager.stop()
