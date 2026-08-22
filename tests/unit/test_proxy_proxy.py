"""Tests for ProxyServer construction and wiring.

HTTPServer protocol-level tests live in test_proxy_server.py; Anthropic
Path-1 wiring in test_proxy_path1.py. This file covers the ProxyServer
wrapper: construction validation, client selection, and the external/
managed setup paths (including vLLM).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge._backend_profiles import (
    ClientAdapter,
    MetadataFormat,
    UnmanagedBackendProfile,
)
from forge.clients.llamafile import LlamafileClient
from forge.clients.ollama import OllamaClient
from forge.clients.vllm import VLLMClient
from forge.context.strategies import NoCompact
from forge.proxy.proxy import ProxyServer
from forge.server import BudgetMode, ServerManager, _ManagedBackendSetup


pytestmark = pytest.mark.usefixtures("mock_httpx_client_constructor")


class TestConstructorValidation:
    """__init__ validation: selector and managed identity rules."""

    def test_neither_url_nor_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="Provide either backend_url"):
            ProxyServer()

    @pytest.mark.parametrize("backend", ["openai", "anthropic"])
    def test_wire_family_selectors_require_unmanaged_mode(self, backend: str) -> None:
        with pytest.raises(ValueError, match="requires backend_url"):
            ProxyServer(backend=backend)

    def test_unknown_selector_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported backend"):
            ProxyServer(backend_url="http://x:8000", backend="unknown")

    # Managed identity rules
    def test_managed_ollama_requires_model(self) -> None:
        with pytest.raises(ValueError, match="backend='ollama' requires model"):
            ProxyServer(backend="ollama")

    def test_managed_llamaserver_requires_gguf(self) -> None:
        with pytest.raises(ValueError, match="requires gguf"):
            ProxyServer(backend="llamaserver")

    def test_managed_llamafile_requires_gguf(self) -> None:
        with pytest.raises(ValueError, match="requires gguf"):
            ProxyServer(backend="llamafile")

    def test_managed_vllm_requires_model_path(self) -> None:
        with pytest.raises(ValueError, match="requires model_path"):
            ProxyServer(backend="vllm")

    @pytest.mark.parametrize("backend_timeout", [0, -1, float("nan"), float("inf")])
    def test_backend_timeout_must_be_finite_and_positive(
        self, backend_timeout: float,
    ) -> None:
        with pytest.raises(
            ValueError, match="backend_timeout must be a finite value greater than 0",
        ):
            ProxyServer(backend_url="http://x:8000", backend_timeout=backend_timeout)

class TestSetupExternal:
    """External setup is metadata-free; only unpinned vLLM gets a latch."""

    @pytest.mark.asyncio
    async def test_llamaserver_uses_llamafile_client(self) -> None:
        proxy = ProxyServer(
            backend_url="http://localhost:8080",
            budget_tokens=8192,
            backend_timeout=1800.0,
        )
        client, ctx, lazy = await proxy._setup_external()
        assert isinstance(client, LlamafileClient)
        assert client.base_url == "http://localhost:8080/v1"
        assert client._http.timeout.read == 1800.0
        assert ctx.budget_tokens == 8192
        assert lazy is None

    @pytest.mark.asyncio
    async def test_external_vllm_retains_mount_port_and_prefix(self) -> None:
        proxy = ProxyServer(
            backend_url="https://gateway.example:9443/team/v1",
            backend="vllm",
            model="served",
            budget_tokens=8192,
        )
        client, _, _ = await proxy._setup_external()
        assert isinstance(client, VLLMClient)
        assert client._chat_url == (
            "https://gateway.example:9443/team/v1/chat/completions"
        )
        assert client._models_url == "https://gateway.example:9443/team/v1/models"

    @pytest.mark.asyncio
    async def test_anthropic_has_no_protocol_context_fallback(self) -> None:
        proxy = ProxyServer(
            backend_url="https://anthropic-gateway.example/service/",
            backend="anthropic",
        )
        fake_client = MagicMock()
        fake_client.get_context_length = AsyncMock(return_value=200_000)
        with patch(
            "forge.clients.anthropic.AnthropicClient", return_value=fake_client,
        ):
            _, ctx, lazy = await proxy._setup_external()
        fake_client.get_context_length.assert_not_awaited()
        assert ctx.budget_tokens is None
        assert lazy is None

    @pytest.mark.asyncio
    async def test_unpinned_vllm_all_auth_budget_cells_are_lazy(self) -> None:
        for static_key in (None, "K"):
            for budget in (None, 4096):
                case = (static_key, budget)
                proxy = ProxyServer(
                    backend_url="http://localhost:8000/deploy",
                    backend="vllm",
                    backend_api_key=static_key,
                    budget_tokens=budget,
                )
                with patch.object(
                    VLLMClient, "get_served_model_name", new_callable=AsyncMock,
                ) as served, patch.object(
                    VLLMClient, "get_context_length", new_callable=AsyncMock,
                ) as context:
                    client, ctx, lazy = await proxy._setup_external()
                assert served.await_count == 0, case
                assert context.await_count == 0, case
                assert client.model == "default", case
                assert lazy is not None and lazy.done is False, case
                assert ctx.budget_tokens == budget, case

    @pytest.mark.asyncio
    async def test_profile_identity_discovery_flag_owns_latch_creation(self) -> None:
        proxy = ProxyServer(
            backend_url="http://localhost:8000",
            backend="vllm",
        )
        assert isinstance(proxy._profile, UnmanagedBackendProfile)
        proxy._profile = replace(proxy._profile, identity_discovery=False)
        assert proxy._resolved_backend is not None
        proxy._resolved_backend = replace(
            proxy._resolved_backend,
            profile=proxy._profile,
        )

        client, _, lazy = await proxy._setup_external()

        assert isinstance(client, VLLMClient)
        assert lazy is None

    @pytest.mark.asyncio
    async def test_pinned_vllm_never_creates_identity_work(self) -> None:
        for static_key in (None, "K"):
            for budget in (None, 4096):
                case = (static_key, budget)
                proxy = ProxyServer(
                    backend_url="http://localhost:8000",
                    backend="vllm",
                    backend_api_key=static_key,
                    budget_tokens=budget,
                    model="pinned",
                )
                client, ctx, lazy = await proxy._setup_external()
                assert client.model == "pinned", case
                assert lazy is None, case
                assert ctx.budget_tokens == budget, case

    @pytest.mark.asyncio
    async def test_generic_missing_budget_does_not_probe(self) -> None:
        proxy = ProxyServer(backend_url="http://localhost:8080")
        with patch.object(
            LlamafileClient, "get_context_length", new_callable=AsyncMock,
        ) as probe:
            _, ctx, lazy = await proxy._setup_external()
        probe.assert_not_awaited()
        assert ctx.budget_tokens is None
        assert lazy is None

    @pytest.mark.asyncio
    async def test_explicit_llamafile_backend_uses_llamafile_client(self) -> None:
        proxy = ProxyServer(
            backend_url="http://localhost:8080",
            backend="llamafile",
            budget_tokens=8192,
        )
        client, _, _ = await proxy._setup_external()
        assert isinstance(client, LlamafileClient)

    @pytest.mark.asyncio
    async def test_nested_terminal_v1_mount_preserves_prefix(self) -> None:
        proxy = ProxyServer(
            backend_url="https://gateway.example/team/deploy/v1/",
            budget_tokens=8192,
        )
        client, _, _ = await proxy._setup_external()
        assert client._chat_url == (
            "https://gateway.example/team/deploy/v1/chat/completions"
        )
        assert client._props_url == "https://gateway.example/team/deploy/props"

    @pytest.mark.asyncio
    async def test_external_ollama_retains_openai_compatibility(self) -> None:
        proxy = ProxyServer(
            backend_url="http://ollama-gateway:11434/root",
            backend="ollama",
            budget_tokens=8192,
        )
        client, _, lazy = await proxy._setup_external()
        assert isinstance(client, LlamafileClient)
        assert client._chat_url == (
            "http://ollama-gateway:11434/root/v1/chat/completions"
        )
        assert lazy is None

    @pytest.mark.asyncio
    async def test_pinned_repo_id_keeps_wire_path_and_registry_key(self) -> None:
        proxy = ProxyServer(
            backend_url="http://localhost:8000",
            backend="vllm",
            model="google/gemma-4-26B-A4B-it",
        )
        client, _, lazy = await proxy._setup_external()
        assert client.model == "google/gemma-4-26B-A4B-it"
        assert client.sampling_key == "gemma-4-26B-A4B-it"
        assert lazy is None

    @pytest.mark.asyncio
    async def test_whitespace_vllm_model_remains_an_explicit_pin(self) -> None:
        proxy = ProxyServer(
            backend_url="http://localhost:8000",
            backend="vllm",
            model="  ",
        )
        client, _, lazy = await proxy._setup_external()
        assert client.model == "  "
        assert client._adopt_served_identity is False
        assert lazy is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "backend_url, expected",
        [
            ("http://localhost:8080/v1", "http://localhost:8080/v1"),
            ("http://localhost:8080/", "http://localhost:8080/v1"),
        ],
    )
    async def test_openai_base_suffix_normalization(
        self, backend_url: str, expected: str,
    ) -> None:
        proxy = ProxyServer(backend_url=backend_url, budget_tokens=8192)
        client, _, _ = await proxy._setup_external()
        assert client.base_url == expected


class TestSetupManaged:
    """Managed mode delegates lifecycle setup then applies NoCompact."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("proxy_kwargs", "client_type", "base_url", "identity_field", "identity"),
        [
            (
                {"backend": "llamaserver", "gguf": "/models/x.gguf"},
                LlamafileClient,
                "http://localhost:8080/v1",
                "gguf_path",
                "/models/x.gguf",
            ),
            (
                {"backend": "llamafile", "gguf": "/models/x.gguf"},
                LlamafileClient,
                "http://localhost:8080/v1",
                "gguf_path",
                "/models/x.gguf",
            ),
            (
                {"backend": "vllm", "model_path": "/models/awq"},
                VLLMClient,
                "http://localhost:8080/v1",
                "model_path",
                "/models/awq",
            ),
            (
                {"backend": "ollama", "model": "ministral-3:14b"},
                OllamaClient,
                "http://localhost:11434",
                "model",
                "ministral-3:14b",
            ),
        ],
        ids=["llamaserver", "llamafile", "vllm", "ollama"],
    )
    async def test_managed_backend_client_and_context_wiring(
        self,
        proxy_kwargs: dict[str, str],
        client_type: type,
        base_url: str,
        identity_field: str,
        identity: str,
    ) -> None:
        proxy = ProxyServer(**proxy_kwargs)
        backend_manager = MagicMock()
        with patch(
            "forge.proxy.proxy._setup_managed_backend",
            new_callable=AsyncMock,
            return_value=_ManagedBackendSetup(backend_manager, 8192),
        ) as setup:
            client, context, lazy_discovery = await proxy._setup_managed()

        assert isinstance(client, client_type)
        assert client.base_url == base_url
        assert context.budget_tokens == 8192
        assert isinstance(context.strategy, NoCompact)
        assert lazy_discovery is None
        assert proxy._server_manager is backend_manager
        kwargs = setup.await_args.kwargs
        assert kwargs["backend"] == proxy_kwargs["backend"]
        assert kwargs["client"] is client
        assert kwargs["mode"] == "native"
        assert kwargs[identity_field] == identity
        assert {
            key: kwargs[key] for key in ("model", "gguf_path", "model_path")
            if key != identity_field
        } == {
            key: None for key in ("model", "gguf_path", "model_path")
            if key != identity_field
        }
        if isinstance(client, LlamafileClient):
            assert client.mode == "native"

    @pytest.mark.asyncio
    async def test_nondefault_managed_options_are_forwarded(self) -> None:
        proxy = ProxyServer(
            backend="llamaserver",
            gguf="/models/x.gguf",
            backend_port=8080,
            budget_mode=BudgetMode.FORGE_FAST,
            extra_flags=["-ngl", "99"],
            backend_timeout=1800.0,
        )
        mock_server = MagicMock()

        with patch(
            "forge.proxy.proxy._setup_managed_backend",
            new_callable=AsyncMock,
            return_value=_ManagedBackendSetup(mock_server, 16384),
        ) as mock_setup:
            client, ctx, _ = await proxy._setup_managed()

        assert isinstance(client, LlamafileClient)
        assert client.base_url == "http://localhost:8080/v1"
        assert client._http.timeout.read == 1800.0
        kwargs = mock_setup.await_args.kwargs
        assert kwargs["backend"] == "llamaserver"
        assert kwargs["gguf_path"] == "/models/x.gguf"
        assert kwargs["model"] is None
        assert kwargs["model_path"] is None
        assert kwargs["mode"] == "native"
        assert kwargs["port"] == 8080
        assert kwargs["budget_mode"] == BudgetMode.FORGE_FAST
        assert kwargs["extra_flags"] == ["-ngl", "99"]
        assert kwargs["client"] is client
        assert kwargs["allow_missing_backend_window"] is True
        assert proxy._server_manager is mock_server
        assert isinstance(ctx.strategy, NoCompact)
        assert ctx.budget_tokens == 16384

    @pytest.mark.asyncio
    async def test_ollama_custom_port_wires_client_and_daemon_target(self) -> None:
        proxy = ProxyServer(
            backend="ollama", model="tag", backend_port=22445,
        )
        server = ServerManager("ollama")
        with patch(
            "forge.proxy.proxy._setup_managed_backend",
            new_callable=AsyncMock,
            return_value=_ManagedBackendSetup(server, 4096),
        ):
            client, _, _ = await proxy._setup_managed()
        assert isinstance(client, OllamaClient)
        assert client.base_url == "http://localhost:22445"
        assert server._resolved_backend is not None
        assert server._resolved_backend.connection.mount_root == (
            "http://localhost:22445"
        )
        assert server._daemon_target_overridden is True

class TestBackendCapability:
    """backend_capability selects the tool-calling protocol, declared once at
    construction and frozen. native (default) = verbatim passthrough; prompt =
    opt-in prompt-injection for non-FC llama.cpp/llamafile backends."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (
                {"backend_url": "http://x:8000", "backend": "vllm"},
                "only supported for",
            ),
            ({"backend": "ollama", "model": "m"}, "only supported for"),
            (
                {"backend_url": "http://x:8080", "backend": "anthropic"},
                "only supported for llama-shaped",
            ),
        ],
        ids=["vllm", "ollama", "anthropic"],
    )
    def test_prompt_rejects_non_llama_capabilities(
        self, kwargs: dict[str, str], match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            ProxyServer(**kwargs, backend_capability="prompt")

    def test_prompt_allowed_for_external_llamacpp(self) -> None:
        # backend=None (external) defaults to the llama.cpp adapter → prompt ok.
        ProxyServer(backend_url="http://x:8080", backend_capability="prompt")
        ProxyServer(backend="llamafile", gguf="m.gguf", backend_capability="prompt")

    @pytest.mark.asyncio
    async def test_external_default_builds_native_client(self) -> None:
        proxy = ProxyServer(backend_url="http://localhost:8080", budget_tokens=8192)
        client, _, _ = await proxy._setup_external()
        assert isinstance(client, LlamafileClient)
        assert client.mode == "native"

    @pytest.mark.asyncio
    async def test_external_prompt_builds_prompt_client(self) -> None:
        proxy = ProxyServer(
            backend_url="http://localhost:8080",
            backend_capability="prompt",
            budget_tokens=8192,
        )
        client, _, _ = await proxy._setup_external()
        assert isinstance(client, LlamafileClient)
        assert client.mode == "prompt"

    @pytest.mark.asyncio
    async def test_managed_prompt_client_is_prompt_but_launch_native(self) -> None:
        # The managed LlamafileClient runs in prompt mode, but the backend
        # process is still launched native (--jinja present, just unused).
        proxy = ProxyServer(
            backend="llamafile", gguf="/m/x.gguf", backend_capability="prompt",
        )
        with patch(
            "forge.proxy.proxy._setup_managed_backend",
            new_callable=AsyncMock,
            return_value=_ManagedBackendSetup(MagicMock(), 8192),
        ) as mock_setup:
            client, _, _ = await proxy._setup_managed()
        assert isinstance(client, LlamafileClient)
        assert client.mode == "prompt"
        assert mock_setup.await_args.kwargs["mode"] == "native"

class TestLifecycle:
    """start()/stop() thread + state management."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("proxy_kwargs", "expected"),
        [
            ({}, (False, 3, 2)),
            (
                {"serialize": True, "max_retries": 0, "max_tool_errors": 5},
                (True, 0, 5),
            ),
        ],
        ids=["external-defaults", "external-overrides"],
    )
    async def test_http_server_receives_serialization_and_retry_controls(
        self,
        proxy_kwargs: dict[str, object],
        expected: tuple[bool, int, int],
    ) -> None:
        proxy = ProxyServer(backend_url="http://backend", **proxy_kwargs)
        http_server = MagicMock()
        http_server.start = AsyncMock()
        with patch(
            "forge.proxy.proxy.HTTPServer", return_value=http_server
        ) as server_cls:
            await proxy._async_start(MagicMock())

        kwargs = server_cls.call_args.kwargs
        assert (
            kwargs["serialize_requests"],
            kwargs["max_retries"],
            kwargs["max_tool_errors"],
        ) == expected
        assert proxy._client is not None
        await proxy._client.aclose()

    def test_url_property(self) -> None:
        proxy = ProxyServer(backend_url="http://localhost:8000", host="0.0.0.0", port=9000)
        assert proxy.url == "http://0.0.0.0:9000"

    def test_stop_before_start_noop(self) -> None:
        ProxyServer(backend_url="http://localhost:8000").stop()  # should not raise

    def test_start_twice_idempotent(self) -> None:
        proxy = ProxyServer(backend_url="http://localhost:8000")
        proxy._started = True
        proxy.start()  # returns immediately without spawning a thread
        assert proxy._thread is None

    def test_start_and_stop_own_event_loop_thread(self) -> None:
        proxy = ProxyServer(backend_url="http://localhost:8000")

        async def signal_ready(ready) -> None:
            proxy._started = True
            ready.set()

        with patch.object(proxy, "_async_start", side_effect=signal_ready) as start:
            proxy.start()
            thread = proxy._thread
            loop = proxy._loop

            try:
                assert thread is not None and thread.is_alive()
                assert loop is not None and loop.is_running()
            finally:
                proxy.stop()

        start.assert_awaited_once()
        assert not thread.is_alive()
        assert loop.is_closed()
        assert proxy._started is False

    @pytest.mark.asyncio
    async def test_managed_async_start_keeps_serialization_with_private_setup(
        self,
    ) -> None:
        proxy = ProxyServer(backend="llamaserver", gguf="/models/x.gguf")
        backend_manager = MagicMock()
        http_server = MagicMock()
        http_server.start = AsyncMock()
        ready = MagicMock()
        with patch(
            "forge.proxy.proxy._setup_managed_backend",
            new_callable=AsyncMock,
            return_value=_ManagedBackendSetup(backend_manager, 8192),
        ), patch("forge.proxy.proxy.HTTPServer", return_value=http_server) as server_cls:
            await proxy._async_start(ready)

        kwargs = server_cls.call_args.kwargs
        assert kwargs["serialize_requests"] is True
        assert kwargs["max_retries"] == 3
        assert kwargs["max_tool_errors"] == 2
        assert kwargs["backend_protocol"] == "openai"
        assert kwargs["client_adapter"] == ClientAdapter.LLAMAFILE
        http_server._configure_metadata_courier.assert_called_once_with(
            mount_root=proxy._resolved_backend.connection.mount_root,
            backend_api_key=None,
            timeout=300.0,
            private_catalog_url=None,
            catalog_parser=None,
        )
        assert isinstance(kwargs["context_manager"].strategy, NoCompact)
        assert proxy._server_manager is backend_manager
        ready.set.assert_called_once_with()
        assert proxy._client is not None
        await proxy._client.aclose()

    @pytest.mark.asyncio
    async def test_official_anthropic_wires_models_metadata_reporting(self) -> None:
        proxy = ProxyServer(
            backend_url="https://api.anthropic.com",
            backend="anthropic",
            model="claude-exact",
            backend_api_key="key",
        )
        http_server = MagicMock()
        http_server.start = AsyncMock()
        ready = MagicMock()
        with patch("forge.proxy.proxy.HTTPServer", return_value=http_server):
            await proxy._async_start(ready)

        http_server._configure_context_reporting.assert_called_once_with(
            managed=False,
            context_window_tokens=None,
            metadata_format=MetadataFormat.ANTHROPIC_MODELS,
            metadata_url=None,
        )
        ready.set.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_managed_ollama_wires_resolved_window_reporting(self) -> None:
        proxy = ProxyServer(backend="ollama", model="tag")
        backend_manager = MagicMock()
        http_server = MagicMock()
        http_server.start = AsyncMock()
        ready = MagicMock()
        with patch(
            "forge.proxy.proxy._setup_managed_backend",
            new_callable=AsyncMock,
            return_value=_ManagedBackendSetup(backend_manager, 32768),
        ), patch("forge.proxy.proxy.HTTPServer", return_value=http_server):
            await proxy._async_start(ready)

        http_server._configure_context_reporting.assert_called_once_with(
            managed=True,
            context_window_tokens=32768,
            metadata_format=MetadataFormat.NONE,
            metadata_url=None,
        )
        ready.set.assert_called_once_with()
        await proxy._client.aclose()

    @pytest.mark.asyncio
    async def test_async_stop_preserves_http_backend_client_order(self) -> None:
        proxy = ProxyServer(backend="llamaserver", gguf="/models/x.gguf")
        order: list[str] = []

        async def stop_http() -> None:
            order.append("http")

        async def stop_backend() -> None:
            order.append("backend")

        async def close_client() -> None:
            order.append("client")

        proxy._http_server = MagicMock()
        proxy._http_server.stop = AsyncMock(side_effect=stop_http)
        proxy._server_manager = MagicMock()
        proxy._server_manager.stop = AsyncMock(side_effect=stop_backend)
        proxy._client = MagicMock()
        proxy._client.aclose = AsyncMock(side_effect=close_client)

        await proxy._async_stop()

        assert order == ["http", "backend", "client"]

    def test_docker_uses_module_entrypoint_and_forge_liveness(self) -> None:
        dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
        assert (
            'ENTRYPOINT ["python", "-m", "forge.proxy", "--host", "0.0.0.0", '
            '"--port", "8081"]'
        ) in dockerfile
        assert 'ENTRYPOINT ["forge-proxy"' not in dockerfile
        assert "http://127.0.0.1:8081/forge/health" in dockerfile
        assert 'http://127.0.0.1:8081/health"]' not in dockerfile
