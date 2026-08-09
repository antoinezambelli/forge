"""ProxyServer — programmatic API for the forge proxy.

Two modes:
- Managed: forge spawns or attaches to the backend according to its profile.
- External: user manages the backend, proxy connects to it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Literal

from forge._backend_profiles import (
    ArtifactIdentity,
    ClientAdapter,
    ManagedBackendProfile,
    MetadataFormat,
    UnmanagedBackendProfile,
    parse_vllm_model_catalog,
)
from forge._endpoint_layouts import BackendOperation
from forge.clients.base import LLMClient
from forge.clients.llamafile import LlamafileClient
from forge.clients.ollama import OllamaClient
from forge.clients.vllm import VLLMClient
from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.reasoning import DEFAULT_REASONING_REPLAY, ReasoningReplay
from forge.proxy._config import _RawProxyConfig, _normalize_proxy_config
from forge.proxy.handler import LazyDiscovery
from forge.proxy.server import HTTPServer
from forge.server import BudgetMode, ServerManager, _setup_managed_backend

logger = logging.getLogger("forge.proxy")


class ProxyServer:
    """OpenAI- and Anthropic-compatible proxy that applies forge guardrails transparently.

    Managed mode — forge spawns or attaches according to the backend profile::

        ProxyServer(backend="llamaserver", gguf="model.gguf")
        ProxyServer(backend="vllm", model_path="/path/to/awq-dir")
        ProxyServer(backend="ollama", model="ministral-3:14b")
        proxy.start()   # starts or attaches per profile; proxy listens on :8081
        proxy.stop()    # stops the proxy and any backend process forge started

    External mode — user manages the backend::

        ProxyServer(backend_url="http://localhost:8080")                  # llama.cpp (default)
        ProxyServer(backend_url="http://localhost:8000", backend="vllm")  # vLLM
        ProxyServer(backend_url="https://api.anthropic.com",
                    backend="anthropic")                                  # Anthropic-shape
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
        backend_port: int | None = None,
        budget_mode: BudgetMode | None = None,
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
        backend_timeout: float = 300.0,
        reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
        backend_api_key: str | None = None,
    ) -> None:
        """
        Args:
            backend_url: URL of an externally managed backend (external mode).
            backend: Backend selector. Managed mode supports "llamaserver",
                "llamafile", "ollama", and "vllm". In external mode, omission
                or "openai" selects the generic OpenAI-compatible profile;
                "llamaserver", "llamafile", "ollama", "vllm", and
                "anthropic" select their corresponding unmanaged profiles.
            model: Model name (managed mode, required for ollama).
            gguf: Path to GGUF file (managed mode, llamaserver/llamafile).
            model_path: Path to a model directory or HF repo id (managed mode,
                vllm only).
            backend_port: Backend target port. Omission selects the managed
                profile default or preserves an unmanaged URL authority.
            budget_mode: Managed context mode. Omission selects ``backend``.
            budget_tokens: Positive manual allocation, valid in managed mode
                only with ``budget_mode=manual``. In external mode it is only
                an operator-supplied reporting denominator; omission leaves
                the reporting window unavailable when it cannot be discovered
                and inference remains available.
            extra_flags: Opaque argv tail for a Forge-spawned llama-server,
                llamafile, or vLLM process. Rejected for managed Ollama and
                external mode.
            host: Proxy listen host.
            port: Proxy listen port.
            serialize: Serialize requests via lock. None = auto (True for
                managed, False for external).
            max_retries: Max consecutive retries for bad LLM responses.
            max_tool_errors: Max consecutive tool-call errors (malformed args)
                before exhaustion. Default 2.
            rescue_enabled: Attempt rescue parsing of text responses.
            backend_capability: Tool-calling protocol for the backend.
                ``native`` (default) uses the selected adapter's structured
                tool path. Compatible generic OpenAI/llama and vLLM clean
                paths preserve raw caller fields; Ollama and Anthropic convert
                or rebuild their downstream shapes, and retries rebuild the
                request. The llama/OpenAI adapter may still merge consecutive
                visible same-role messages for template compatibility; it does
                not compact or delete history. ``prompt`` opts into
                prompt-injection for a non-FC llama.cpp/llamafile backend —
                tools are stripped into the prompt and the JSON tool call is
                parsed back out (the same path the WorkflowRunner uses). Only
                valid for llama.cpp/llamafile backends; rejected for vllm/ollama
                and the anthropic protocol. Selected once at construction and
                frozen — never probed or switched mid-stream.
            inject_respond_tool: When True, inject forge's synthetic respond()
                tool into requests that already carry tools (keeps the model in
                tool-calling mode). Default False. Orthogonal to
                backend_capability — works in both native and prompt modes.
            backend_timeout: Timeout in seconds for requests from the proxy to
                the downstream backend.
            reasoning_replay: How much captured reasoning to replay to the
                backend on later turns: ``full``, ``keep-last``, or ``none``.
            backend_api_key: Static credential forge sends to the backend in
                its native auth header (LM Studio / hosted providers / service
                accounts). Baked into the backend client at construction. When
                set, an inbound auth header is a second credential and the
                request is refused (at most one credential per request). Leave
                None for pure inbound-credential passthrough.
        """
        config = _normalize_proxy_config(_RawProxyConfig(
            backend_url=backend_url,
            backend=backend,
            model=model,
            gguf=gguf,
            model_path=model_path,
            backend_port=backend_port,
            budget_mode=budget_mode,
            budget_tokens=budget_tokens,
            extra_flags=extra_flags,
            host=host,
            port=port,
            serialize=serialize,
            max_retries=max_retries,
            max_tool_errors=max_tool_errors,
            rescue_enabled=rescue_enabled,
            backend_capability=backend_capability,
            inject_respond_tool=inject_respond_tool,
            backend_timeout=backend_timeout,
            reasoning_replay=reasoning_replay,
            backend_api_key=backend_api_key,
        ))
        # Prompt-injection is a llama.cpp/llamafile capability only. vLLM and
        # Ollama clients are native-only (vLLM preserves raw tools;
        # Ollama rebuilds them, and neither has a prompt path); the anthropic
        # protocol does its own tool conversion.
        # backend=None (external) defaults to the llama.cpp adapter, which
        # supports prompt — so only vllm/ollama and anthropic are rejected.
        self._backend_url = config.backend_url
        self._backend = config.backend
        self._profile = config.profile
        self._model = config.model
        self._gguf = config.gguf
        self._model_path = config.model_path
        self._backend_port = config.backend_port
        self._budget_mode = config.budget_mode
        self._budget_tokens = config.budget_tokens
        self._extra_flags = list(config.extra_flags) if config.extra_flags else None
        self._host = config.host
        self._port = config.port
        self._max_retries = config.max_retries
        self._max_tool_errors = config.max_tool_errors
        self._rescue_enabled = config.rescue_enabled
        self._backend_capability = config.backend_capability
        self._inject_respond_tool = config.inject_respond_tool
        self._backend_protocol = config.protocol
        self._backend_timeout = config.backend_timeout
        self._reasoning_replay = config.reasoning_replay
        self._backend_api_key = config.backend_api_key
        self._resolved_backend = config.resolved_backend

        self._serialize = config.serialize

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
        """Start the proxy, spawning or attaching to its managed backend.

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
        """Stop the proxy, stopping a spawned process or unloading Ollama."""
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
        assert self._resolved_backend is not None
        if isinstance(self._profile, UnmanagedBackendProfile):
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
            client_adapter=self._profile.family_profile.client_adapter,
            backend_api_key_present=bool(self._backend_api_key),
            lazy_discovery=lazy_discovery,
        )
        metadata_format = self._profile.family_profile.metadata_format
        uses_official_anthropic = getattr(
            client, "_uses_official_metadata_root", None,
        )
        if (
            self._profile.family_profile.client_adapter == ClientAdapter.ANTHROPIC
            and callable(uses_official_anthropic)
            and uses_official_anthropic()
        ):
            metadata_format = MetadataFormat.ANTHROPIC_MODELS
        metadata_url: str | None = None
        if metadata_format == MetadataFormat.VLLM_MODELS:
            metadata_url = self._resolved_backend.address(
                BackendOperation.MODEL_CATALOG,
            )
        elif metadata_format == MetadataFormat.LLAMA_PROPERTIES:
            metadata_url = self._resolved_backend.address(
                BackendOperation.PROPERTIES,
            )
        self._http_server._configure_metadata_courier(
            mount_root=self._resolved_backend.connection.mount_root,
            backend_api_key=self._backend_api_key,
            timeout=self._backend_timeout,
            private_catalog_url=(
                metadata_url
                if metadata_format == MetadataFormat.VLLM_MODELS else None
            ),
            catalog_parser=(
                parse_vllm_model_catalog
                if metadata_format == MetadataFormat.VLLM_MODELS else None
            ),
        )
        self._http_server._configure_context_reporting(
            managed=isinstance(self._profile, ManagedBackendProfile),
            context_window_tokens=context_manager.budget_tokens,
            metadata_format=metadata_format,
            metadata_url=metadata_url,
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
        assert self._resolved_backend is not None
        profile = self._profile
        assert isinstance(profile, UnmanagedBackendProfile)
        adapter = profile.family_profile.client_adapter

        if adapter == ClientAdapter.ANTHROPIC:
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
                    "backend='anthropic' requires the anthropic SDK. "
                    "Install it with: pip install 'forge-guardrails[anthropic]'"
                ) from exc
            client: LLMClient = AnthropicClient(
                model=self._model,
                base_url=self._resolved_backend.adapter_base_url,
                timeout=self._backend_timeout,
                # Explicit key (or "" for pure passthrough) so an ambient
                # ANTHROPIC_* env var can't become a silent second credential.
                api_key=self._backend_api_key or "",
            )
            context_manager = ContextManager(
                strategy=NoCompact(),
                budget_tokens=self._budget_tokens,
            )
            # Protocol alone supplies no trustworthy reporting denominator.
            return client, context_manager, None

        # Path 2 / default — OpenAI-shape downstream (llama.cpp or vLLM).
        base = self._resolved_backend.adapter_base_url

        if adapter == ClientAdapter.VLLM:
            # An explicit --model pins the wire identity (issue #122): it seeds
            # (model, sampling_key) at construction and suppresses served-name
            # adoption in both discovery paths below. Without it, "default" is
            # a placeholder that discovery replaces.
            client = VLLMClient(
                model_path=self._model or "default",
                base_url=base,
                timeout=self._backend_timeout,
                api_key=self._backend_api_key or "",
                adopt_served_identity=self._model is None,
            )
        else:
            # llamaserver / llamafile / unspecified — OpenAI-compatible adapter.
            # Caller manages the backend, so we don't have a GGUF path. gguf_path
            # intentionally receives a bare model name here (proxy mode); the
            # client strips only a trailing .gguf/.llamafile if present.
            client = LlamafileClient(
                gguf_path=self._model or "default",
                base_url=base,
                mode=self._backend_capability,
                timeout=self._backend_timeout,
                api_key=self._backend_api_key or "",
            )

        # Profiles without identity discovery do not probe merely for context.
        # Context metadata is reporting-only, so no budget-only probe may block
        # startup or a first inference request.
        if not profile.identity_discovery:
            context_manager = ContextManager(
                strategy=NoCompact(),
                budget_tokens=self._budget_tokens,
            )
            return client, context_manager, None

        lazy_discovery = LazyDiscovery() if self._model is None else None

        context_manager = ContextManager(
            strategy=NoCompact(),
            budget_tokens=self._budget_tokens,
        )
        return client, context_manager, lazy_discovery

    async def _setup_managed(
        self,
    ) -> tuple[LLMClient, ContextManager, LazyDiscovery | None]:
        """Managed mode: spawn or attach, then compose Proxy context policy."""
        assert self._backend is not None
        profile = self._profile
        assert isinstance(profile, ManagedBackendProfile)
        client = self._build_managed_client(profile)

        # The backend process is always launched in native mode (--jinja enables
        # the native tools API). This is independent of backend_capability: in
        # prompt capability the proxy simply doesn't send native tools, so a
        # native-launched backend (jinja template present but unused) serves the
        # prompt-injected request fine. Keeping launch native avoids changing
        # backend startup flags for the opt-in path. Pass each backend only its
        # own identity field — the shared setup seam enforces exclusivity.
        assert self._backend_port is not None
        assert self._budget_mode is not None
        managed_setup = await _setup_managed_backend(
            backend=self._backend,
            model=(
                self._model
                if profile.required_identity == ArtifactIdentity.MODEL_TAG else None
            ),
            gguf_path=(
                self._gguf
                if profile.required_identity == ArtifactIdentity.GGUF_PATH else None
            ),
            model_path=(
                self._model_path
                if profile.required_identity == ArtifactIdentity.MODEL_PATH else None
            ),
            mode="native",
            budget_mode=self._budget_mode,
            manual_tokens=self._budget_tokens,
            client=client,
            port=self._backend_port,
            extra_flags=self._extra_flags,
            allow_missing_backend_window=True,
        )
        server = managed_setup.server
        context_manager = ContextManager(
            strategy=NoCompact(),
            budget_tokens=managed_setup.context_window_tokens,
        )
        if (
            profile.family_profile.client_adapter == ClientAdapter.OLLAMA
            and isinstance(server, ServerManager)
        ):
            server._set_resolved_daemon_target(self._resolved_backend)
        self._server_manager = server
        # Managed mode probes its own ungated local backend at startup — never
        # deferred.
        return client, context_manager, None

    def _build_managed_client(self, profile: ManagedBackendProfile) -> LLMClient:
        """Construct the right client for the managed backend."""
        assert self._resolved_backend is not None
        adapter = profile.family_profile.client_adapter
        base_url = self._resolved_backend.adapter_base_url
        if adapter == ClientAdapter.OLLAMA:
            assert self._model is not None
            return OllamaClient(
                model=self._model,
                base_url=base_url,
                timeout=self._backend_timeout,
                api_key=self._backend_api_key or "",
            )
        if adapter == ClientAdapter.LLAMAFILE:
            # gguf_path may be a real GGUF file path or a bare model name
            # (proxy external mode); the client handles both via the same
            # extension-stripping logic.
            return LlamafileClient(
                gguf_path=self._gguf or "default",
                base_url=base_url,
                mode=self._backend_capability,
                timeout=self._backend_timeout,
                api_key=self._backend_api_key or "",
            )
        if adapter == ClientAdapter.VLLM:
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
