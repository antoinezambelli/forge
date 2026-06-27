"""ProxyServer — programmatic API for the forge proxy.

Two modes:
- Managed: forge starts and manages the backend via ServerManager.
- External: user manages the backend, proxy connects to it.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from pathlib import Path
from typing import Literal

from forge.clients.base import LLMClient
from forge.clients.llamafile import LlamafileClient
from forge.clients.ollama import OllamaClient
from forge.clients.vllm import VLLMClient
from forge.context.manager import ContextManager
from forge.context.strategies import TieredCompact
from forge.core.reasoning import DEFAULT_REASONING_REPLAY, ReasoningReplay, validate_reasoning_replay
from forge.proxy.handler import LazyDiscovery
from forge.proxy.server import HTTPServer
from forge.server import BudgetMode, ServerManager, setup_backend

# Placeholder budget for a context manager whose real budget is discovered
# lazily on the first request (external passthrough). It is never read before
# discovery overwrites it — the discovery hook runs before any compaction —
# but it's set large so an accidental early read degrades safe (no compaction)
# rather than over-compacting on a tiny budget.
_DEFERRED_BUDGET_PLACEHOLDER = 1_000_000

logger = logging.getLogger("forge.proxy")


class ProxyServer:
    """OpenAI- and Anthropic-compatible proxy that applies forge guardrails transparently.

    Managed mode — forge starts the backend::

        ProxyServer(backend="llamaserver", gguf="model.gguf")
        ProxyServer(backend="vllm", model_path="/path/to/awq-dir")
        ProxyServer(backend="ollama", model="ministral-3:14b")
        proxy.start()   # starts the backend on :8080 + proxy on :8081
        proxy.stop()    # stops both

    External mode — user manages the backend::

        ProxyServer(backend_url="http://localhost:8080")                  # llama.cpp (default)
        ProxyServer(backend_url="http://localhost:8000", backend="vllm")  # vLLM
        ProxyServer(backend_url="https://api.anthropic.com",
                    backend_protocol="anthropic")                         # Anthropic-shape
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
        model_path: str | Path | None = None,
        backend_port: int = 8080,
        budget_mode: BudgetMode = BudgetMode.BACKEND,
        budget_tokens: int | None = None,
        extra_flags: list[str] | None = None,
        # Proxy settings
        host: str = "127.0.0.1",
        port: int = 8081,
        serialize: bool | None = None,
        max_retries: int = 3,
        max_tool_errors: int = 2,
        rescue_enabled: bool = True,
        backend_capability: Literal["native", "prompt"] = "native",
        inject_respond_tool: bool = False,
        backend_protocol: Literal["openai", "anthropic"] = "openai",
        backend_timeout: float = 300.0,
        reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
        backend_api_key: str | None = None,
    ) -> None:
        """
        Args:
            backend_url: URL of an externally managed backend (external mode).
            backend: Backend type — "llamaserver", "llamafile", "ollama", or
                "vllm". Required for managed mode; in external mode it selects
                the client adapter ("vllm" for a vLLM server, otherwise the
                OpenAI-compatible llama.cpp adapter).
            model: Model name (managed mode, required for ollama).
            gguf: Path to GGUF file (managed mode, llamaserver/llamafile).
            model_path: Path to a model directory or HF repo id (managed mode,
                vllm only).
            backend_port: Port for the managed backend (default 8080).
            budget_mode: How to determine context budget.
            budget_tokens: Explicit token budget. In external mode this is
                required if the backend doesn't report its context length.
            extra_flags: Additional CLI flags for the managed backend.
            host: Proxy listen host.
            port: Proxy listen port.
            serialize: Serialize requests via lock. None = auto (True for
                managed, False for external).
            max_retries: Max consecutive retries for bad LLM responses.
            max_tool_errors: Max consecutive tool-call errors (malformed args)
                before exhaustion. Default 2.
            rescue_enabled: Attempt rescue parsing of text responses.
            backend_capability: Tool-calling protocol for the backend.
                ``native`` (default) forwards the client's OpenAI tools/messages
                verbatim to a function-calling-capable backend (transparent
                passthrough). ``prompt`` opts into prompt-injection for a non-FC
                llama.cpp/llamafile backend — tools are stripped into the prompt
                and the JSON tool call is parsed back out (the same path the
                WorkflowRunner uses). Only valid for llama.cpp/llamafile
                backends; rejected for vllm/ollama and the anthropic protocol.
                Selected once at construction and frozen — never probed or
                switched mid-stream.
            inject_respond_tool: When True, inject forge's synthetic respond()
                tool into requests that already carry tools (keeps the model in
                tool-calling mode). Default False. Orthogonal to
                backend_capability — works in both native and prompt modes.
            backend_protocol: Wire format of the external backend.
                ``openai`` (default) for llama.cpp, vLLM, Ollama. ``anthropic``
                for Anthropic-shape downstreams (the official Anthropic API,
                LiteLLM's /v1/messages, a self-hosted Anthropic proxy).
                Only meaningful in external mode; ignored in managed mode.
            backend_timeout: Timeout in seconds for requests from the proxy to
                the downstream backend.
            reasoning_replay: How much captured reasoning to replay to the
                backend on later turns: ``full``, ``keep-last``, or ``none``.
            backend_api_key: Static credential forge sends to the backend in
                its native auth header (LM Studio / hosted providers / service
                accounts). Baked into the backend client at construction. When
                set, an inbound auth header is a second credential and the
                request is refused (one credential per request). Leave None for
                pure inbound-credential passthrough.
        """
        if backend_url is None and backend is None:
            raise ValueError("Provide either backend_url (external) or backend (managed)")
        if backend_protocol == "anthropic" and backend_url is None:
            raise ValueError(
                "backend_protocol='anthropic' requires external mode (backend_url=...). "
                "Managed mode launches local llama.cpp / Ollama, which only speak OpenAI."
            )
        if backend == "vllm" and backend_protocol == "anthropic":
            raise ValueError(
                "backend='vllm' speaks the OpenAI protocol; backend_protocol='anthropic' "
                "is not applicable."
            )
        # Prompt-injection is a llama.cpp/llamafile capability only. vLLM and
        # Ollama clients are native-only (they accept-ignore raw tools and have
        # no prompt path); the anthropic protocol does its own tool conversion.
        # backend=None (external) defaults to the llama.cpp adapter, which
        # supports prompt — so only vllm/ollama and anthropic are rejected.
        if backend_capability == "prompt":
            if backend_protocol == "anthropic":
                raise ValueError(
                    "backend_capability='prompt' is not supported with the "
                    "anthropic protocol (native tool calling only)."
                )
            if backend in ("vllm", "ollama"):
                raise ValueError(
                    f"backend_capability='prompt' is only supported for "
                    f"llama.cpp/llamafile backends, not backend={backend!r}."
                )
        if not math.isfinite(backend_timeout) or backend_timeout <= 0:
            raise ValueError("backend_timeout must be a finite value greater than 0")
        # Managed mode: each backend requires its own identity field. Fail
        # fast at construction with a clear message (mirrors setup_backend).
        if backend_url is None:
            if backend == "ollama" and model is None:
                raise ValueError("backend='ollama' requires model")
            if backend in ("llamaserver", "llamafile") and gguf is None:
                raise ValueError(f"backend={backend!r} requires gguf")
            if backend == "vllm" and model_path is None:
                raise ValueError("backend='vllm' requires model_path")

        self._backend_url = backend_url
        self._backend = backend
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
        self._max_tool_errors = max_tool_errors
        self._rescue_enabled = rescue_enabled
        self._backend_capability = backend_capability
        self._inject_respond_tool = inject_respond_tool
        self._backend_protocol = backend_protocol
        self._backend_timeout = backend_timeout
        self._reasoning_replay = validate_reasoning_replay(reasoning_replay)
        # A blank/whitespace --backend-api-key is not a credential: normalize it
        # to None so it neither rides the wire as garbage nor disables lazy
        # discovery (which keys off "is a static key present").
        self._backend_api_key = (
            backend_api_key if (backend_api_key and backend_api_key.strip()) else None
        )

        # Auto-detect serialization: managed (no external url) = single local
        # GPU = serialize. External callers manage their own concurrency.
        if serialize is None:
            self._serialize = backend_url is None
        else:
            self._serialize = serialize

        self._server_manager: ServerManager | None = None
        self._http_server: HTTPServer | None = None
        self._client: LLMClient | None = None
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

        logger.info(
            "Proxy ready at %s (backend_timeout=%.1fs)",
            self.url,
            self._backend_timeout,
        )

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
        if self._backend_url is not None:
            client, context_manager, lazy_discovery = await self._setup_external()
        else:
            client, context_manager, lazy_discovery = await self._setup_managed()

        self._client = client
        self._http_server = HTTPServer(
            client=client,
            context_manager=context_manager,
            host=self._host,
            port=self._port,
            serialize_requests=self._serialize,
            max_retries=self._max_retries,
            max_tool_errors=self._max_tool_errors,
            rescue_enabled=self._rescue_enabled,
            native_passthrough=self._backend_capability == "native",
            inject_respond_tool=self._inject_respond_tool,
            reasoning_replay=self._reasoning_replay,
            backend_protocol=self._backend_protocol,
            backend_api_key_present=bool(self._backend_api_key),
            lazy_discovery=lazy_discovery,
        )
        await self._http_server.start()
        self._started = True
        ready.set()

    async def _setup_external(
        self,
    ) -> tuple[LLMClient, ContextManager, LazyDiscovery | None]:
        """External mode: connect to a caller-managed backend.

        Returns the client, its context manager, and an optional LazyDiscovery
        latch — non-None only when backend discovery is deferred to the first
        request (external passthrough; see Path 2 below).
        """
        assert self._backend_url is not None

        if self._backend_protocol == "anthropic":
            # Path 1 — downstream speaks the Anthropic Messages API
            # (LiteLLM /v1/messages, real Anthropic, self-hosted proxy).
            # AnthropicClient handles base_url and SDK retries; forge
            # guardrails wrap its inference loop like any other client.
            # Lazy import: the anthropic SDK is an optional dependency
            # (forge-guardrails[anthropic]). Only Path 1 needs it, so
            # Path 2 / local-backend users must not be forced to install
            # it just to start the proxy.
            try:
                from forge.clients.anthropic import AnthropicClient
            except ImportError as exc:
                raise RuntimeError(
                    "backend_protocol='anthropic' requires the anthropic SDK. "
                    "Install it with: pip install 'forge-guardrails[anthropic]'"
                ) from exc
            client: LLMClient = AnthropicClient(
                model=self._model or "claude",
                base_url=self._backend_url.rstrip("/"),
                timeout=self._backend_timeout,
                # Explicit key (or "" for pure passthrough) so an ambient
                # ANTHROPIC_* env var can't become a silent second credential.
                api_key=self._backend_api_key or "",
            )
            # Anthropic models report a known context length; keep the legacy
            # 8192 fallback rather than failing the well-behaved Path-1 case.
            budget = self._budget_tokens or await client.get_context_length() or 8192
            context_manager = ContextManager(
                strategy=TieredCompact(),
                budget_tokens=budget,
            )
            # Anthropic reports a static context length with no network call, so
            # there's nothing to defer.
            return client, context_manager, None

        # Path 2 / default — OpenAI-shape downstream (llama.cpp or vLLM).
        base = self._backend_url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"

        if self._backend == "vllm":
            client = VLLMClient(
                model_path="default",
                base_url=base,
                timeout=self._backend_timeout,
                api_key=self._backend_api_key or "",
            )
        else:
            # llamaserver / llamafile / unspecified — OpenAI-compatible adapter.
            # Caller manages the backend, so we don't have a GGUF path. "default"
            # is a placeholder identity for the wire model field (llama-server
            # ignores it) and the JSONL model field.
            client = LlamafileClient(
                gguf_path=self._model or "default",
                base_url=base,
                mode=self._backend_capability,
                timeout=self._backend_timeout,
                api_key=self._backend_api_key or "",
            )

        # Decide eager vs deferred discovery. A static --backend-api-key
        # authenticates the startup probe, so discover now and keep boot-time
        # fail-fast. Pure passthrough (no static key) can't probe at startup —
        # the probe would be unauthenticated against a gated backend (finding
        # #2) — so defer it to the first request, which carries the credential.
        # vLLM always needs deferral when passthrough (its served-identity probe
        # is unauthenticated too); llama.cpp only needs it to discover a budget.
        static_key = bool(self._backend_api_key)
        defer = (not static_key) and (
            self._budget_tokens is None or self._backend == "vllm"
        )

        if defer:
            apply_budget = self._budget_tokens is None
            budget = (
                _DEFERRED_BUDGET_PLACEHOLDER
                if apply_budget
                else self._budget_tokens
            )
            lazy_discovery: LazyDiscovery | None = LazyDiscovery(
                deferred=True, apply_budget=apply_budget,
            )
        else:
            lazy_discovery = None
            if self._backend == "vllm":
                # Unlike llama.cpp, vLLM validates the wire `model` field against
                # its --served-model-name aliases (404 on mismatch). External
                # mode has no model path to send, so discover the served identity
                # from /v1/models instead of shipping the "default" placeholder.
                served = await client.get_served_model_name()
                if served:
                    logger.info("Discovered vLLM served model name: %s", served)
                    client._set_model_identity(served)
                else:
                    logger.warning(
                        "Could not discover a served model name from %s/models; "
                        "sending placeholder 'default' (vLLM will 404 if it "
                        "validates the model field)",
                        base,
                    )
            if self._budget_tokens is not None:
                budget = self._budget_tokens
            else:
                ctx_len = await client.get_context_length()
                if ctx_len is None:
                    raise RuntimeError(
                        f"backend at {self._backend_url} did not report a context "
                        "length; pass budget_tokens explicitly"
                    )
                budget = ctx_len

        context_manager = ContextManager(
            strategy=TieredCompact(),
            budget_tokens=budget,
        )
        return client, context_manager, lazy_discovery

    async def _setup_managed(
        self,
    ) -> tuple[LLMClient, ContextManager, LazyDiscovery | None]:
        """Managed mode: forge starts the backend via setup_backend."""
        assert self._backend is not None
        client = self._build_managed_client()

        # The backend process is always launched in native mode (--jinja enables
        # the native tools API). This is independent of backend_capability: in
        # prompt capability the proxy simply doesn't send native tools, so a
        # native-launched backend (jinja template present but unused) serves the
        # prompt-injected request fine. Keeping launch native avoids changing
        # backend startup flags for the opt-in path. Pass each backend only its
        # own identity field — setup_backend enforces mutual exclusivity.
        server, context_manager = await setup_backend(
            backend=self._backend,
            model=self._model if self._backend == "ollama" else None,
            gguf_path=self._gguf if self._backend in ("llamaserver", "llamafile") else None,
            model_path=self._model_path if self._backend == "vllm" else None,
            mode="native",
            budget_mode=self._budget_mode,
            manual_tokens=self._budget_tokens,
            client=client,
            port=self._backend_port,
            extra_flags=self._extra_flags,
        )
        self._server_manager = server
        # Managed mode probes its own ungated local backend at startup — never
        # deferred.
        return client, context_manager, None

    def _build_managed_client(self) -> LLMClient:
        """Construct the right client for the managed backend."""
        base_url = f"http://localhost:{self._backend_port}/v1"
        if self._backend == "ollama":
            assert self._model is not None
            return OllamaClient(
                model=self._model,
                timeout=self._backend_timeout,
                api_key=self._backend_api_key or "",
            )
        if self._backend in ("llamaserver", "llamafile"):
            return LlamafileClient(
                gguf_path=self._gguf or "default",
                base_url=base_url,
                mode=self._backend_capability,
                timeout=self._backend_timeout,
                api_key=self._backend_api_key or "",
            )
        if self._backend == "vllm":
            assert self._model_path is not None
            return VLLMClient(
                model_path=self._model_path,
                base_url=base_url,
                timeout=self._backend_timeout,
                api_key=self._backend_api_key or "",
            )
        raise ValueError(f"unsupported backend: {self._backend!r}")

    async def _async_stop(self) -> None:
        """Async shutdown."""
        if self._http_server is not None:
            await self._http_server.stop()
        if self._server_manager is not None:
            await self._server_manager.stop()
        if self._client is not None:
            await self._client.aclose()
