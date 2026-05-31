"""Tests for ProxyServer construction and wiring.

HTTPServer protocol-level tests live in test_proxy_server.py; Anthropic
Path-1 wiring in test_proxy_path1.py. This file covers the ProxyServer
wrapper: construction validation, client selection, and the external/
managed setup paths (including vLLM).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.clients.llamafile import LlamafileClient
from forge.clients.ollama import OllamaClient
from forge.clients.vllm import VLLMClient
from forge.context.manager import ContextManager
from forge.proxy.proxy import ProxyServer
from forge.server import BudgetMode


class TestConstructorValidation:
    """__init__ validation: protocol guards and managed identity rules."""

    def test_neither_url_nor_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="Provide either backend_url"):
            ProxyServer()

    def test_anthropic_requires_external(self) -> None:
        with pytest.raises(ValueError, match="requires external mode"):
            ProxyServer(backend="llamaserver", gguf="m.gguf", backend_protocol="anthropic")

    def test_vllm_rejects_anthropic_protocol(self) -> None:
        with pytest.raises(ValueError, match="speaks the OpenAI protocol"):
            ProxyServer(backend_url="http://x:8000", backend="vllm", backend_protocol="anthropic")

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

    def test_managed_ok(self) -> None:
        ProxyServer(backend="llamaserver", gguf="m.gguf")
        ProxyServer(backend="llamafile", gguf="m.gguf")
        ProxyServer(backend="vllm", model_path="/m")
        ProxyServer(backend="ollama", model="llama3")

    def test_external_ok(self) -> None:
        proxy = ProxyServer(backend_url="http://x:8080")
        assert proxy._backend_url == "http://x:8080"
        assert proxy._backend is None
        proxy2 = ProxyServer(backend_url="http://x:8000", backend="vllm")
        assert proxy2._backend == "vllm"

    def test_backend_timeout_default_and_override(self) -> None:
        assert ProxyServer(backend_url="http://x:8000")._backend_timeout == 300.0
        proxy = ProxyServer(backend_url="http://x:8000", backend_timeout=1800.0)
        assert proxy._backend_timeout == 1800.0

    # Serialize auto-detection: managed (no url) serializes, external does not.
    def test_serialize_auto_managed_true(self) -> None:
        assert ProxyServer(backend="vllm", model_path="/m")._serialize is True

    def test_serialize_auto_external_false(self) -> None:
        # Even with backend set (external vLLM), external mode does not serialize.
        assert ProxyServer(backend_url="http://x:8000", backend="vllm")._serialize is False

    def test_serialize_override(self) -> None:
        assert ProxyServer(backend_url="http://x:8000", serialize=True)._serialize is True


class TestSetupExternal:
    """External mode constructs the right client and resolves budget."""

    @pytest.mark.asyncio
    async def test_llamaserver_uses_llamafile_client(self) -> None:
        proxy = ProxyServer(
            backend_url="http://localhost:8080",
            budget_tokens=8192,
            backend_timeout=1800.0,
        )
        client, ctx = await proxy._setup_external()
        assert isinstance(client, LlamafileClient)
        assert client.base_url == "http://localhost:8080/v1"
        assert client._http.timeout.read == 1800.0
        assert ctx.budget_tokens == 8192

    @pytest.mark.asyncio
    async def test_explicit_llamafile_backend_uses_llamafile_client(self) -> None:
        proxy = ProxyServer(
            backend_url="http://localhost:8080", backend="llamafile", budget_tokens=8192,
        )
        client, _ = await proxy._setup_external()
        assert isinstance(client, LlamafileClient)

    @pytest.mark.asyncio
    async def test_vllm_uses_vllm_client(self) -> None:
        proxy = ProxyServer(
            backend_url="http://localhost:8000",
            backend="vllm",
            budget_tokens=8192,
            backend_timeout=1800.0,
        )
        with patch.object(
            VLLMClient, "get_served_model_name", new_callable=AsyncMock, return_value=None,
        ):
            client, ctx = await proxy._setup_external()
        assert isinstance(client, VLLMClient)
        assert client.base_url == "http://localhost:8000/v1"
        assert client._http.timeout.read == 1800.0
        assert ctx.budget_tokens == 8192

    @pytest.mark.asyncio
    async def test_vllm_adopts_served_model_name(self) -> None:
        proxy = ProxyServer(
            backend_url="http://localhost:8000", backend="vllm", budget_tokens=8192,
        )
        with patch.object(
            VLLMClient, "get_served_model_name",
            new_callable=AsyncMock, return_value="my-awq-model",
        ):
            client, _ = await proxy._setup_external()
        assert client.model_path == "my-awq-model"
        assert client.model == "my-awq-model"

    @pytest.mark.asyncio
    async def test_vllm_served_repo_id_keeps_wire_path_derives_registry_key(self) -> None:
        # An HF-repo-id served name must reach the wire verbatim (vLLM validates
        # it), while the registry key is the derived stem — the (model_path,
        # model) invariant, applied to served-name adoption.
        proxy = ProxyServer(
            backend_url="http://localhost:8000", backend="vllm", budget_tokens=8192,
        )
        with patch.object(
            VLLMClient, "get_served_model_name",
            new_callable=AsyncMock, return_value="google/gemma-4-26B-A4B-it",
        ):
            client, _ = await proxy._setup_external()
        assert client.model_path == "google/gemma-4-26B-A4B-it"
        assert client.model == "gemma-4-26B-A4B-it"

    @pytest.mark.asyncio
    async def test_vllm_keeps_placeholder_when_discovery_fails(self) -> None:
        proxy = ProxyServer(
            backend_url="http://localhost:8000", backend="vllm", budget_tokens=8192,
        )
        with patch.object(
            VLLMClient, "get_served_model_name", new_callable=AsyncMock, return_value=None,
        ):
            client, _ = await proxy._setup_external()
        assert client.model_path == "default"

    @pytest.mark.asyncio
    async def test_url_v1_suffix_preserved(self) -> None:
        proxy = ProxyServer(backend_url="http://localhost:8080/v1", budget_tokens=8192)
        client, _ = await proxy._setup_external()
        assert client.base_url == "http://localhost:8080/v1"

    @pytest.mark.asyncio
    async def test_url_trailing_slash_stripped(self) -> None:
        proxy = ProxyServer(backend_url="http://localhost:8080/", budget_tokens=8192)
        client, _ = await proxy._setup_external()
        assert client.base_url == "http://localhost:8080/v1"

    @pytest.mark.asyncio
    async def test_budget_from_backend_when_unspecified(self) -> None:
        proxy = ProxyServer(backend_url="http://localhost:8080")
        with patch.object(
            LlamafileClient, "get_context_length",
            new_callable=AsyncMock, return_value=32768,
        ):
            _, ctx = await proxy._setup_external()
        assert ctx.budget_tokens == 32768

    @pytest.mark.asyncio
    async def test_budget_unresolvable_raises(self) -> None:
        proxy = ProxyServer(backend_url="http://localhost:8080")
        with patch.object(
            LlamafileClient, "get_context_length",
            new_callable=AsyncMock, return_value=None,
        ), pytest.raises(RuntimeError, match="did not report a context length"):
            await proxy._setup_external()


class TestSetupManaged:
    """Managed mode delegates to setup_backend with the right identity field."""

    @pytest.mark.asyncio
    async def test_llamaserver_wiring(self) -> None:
        proxy = ProxyServer(
            backend="llamaserver",
            gguf="/models/x.gguf",
            backend_port=8080,
            budget_mode=BudgetMode.FORGE_FAST,
            extra_flags=["-ngl", "99"],
            backend_timeout=1800.0,
        )
        mock_ctx = ContextManager.__new__(ContextManager)
        mock_ctx.budget_tokens = 16384
        mock_server = MagicMock()

        with patch(
            "forge.proxy.proxy.setup_backend",
            new_callable=AsyncMock, return_value=(mock_server, mock_ctx),
        ) as mock_setup:
            client, ctx = await proxy._setup_managed()

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
        assert proxy._server_manager is mock_server
        assert ctx is mock_ctx

    @pytest.mark.asyncio
    async def test_vllm_wiring(self) -> None:
        proxy = ProxyServer(
            backend="vllm", model_path="/models/awq", backend_port=8000,
            budget_tokens=113000, budget_mode=BudgetMode.MANUAL,
            backend_timeout=1800.0,
        )
        mock_ctx = ContextManager.__new__(ContextManager)
        mock_ctx.budget_tokens = 113000
        with patch(
            "forge.proxy.proxy.setup_backend",
            new_callable=AsyncMock, return_value=(MagicMock(), mock_ctx),
        ) as mock_setup:
            client, _ = await proxy._setup_managed()

        assert isinstance(client, VLLMClient)
        assert client.base_url == "http://localhost:8000/v1"
        assert client._http.timeout.read == 1800.0
        kwargs = mock_setup.await_args.kwargs
        assert kwargs["backend"] == "vllm"
        assert kwargs["model_path"] == "/models/awq"
        assert kwargs["gguf_path"] is None
        assert kwargs["model"] is None
        assert kwargs["manual_tokens"] == 113000
        assert kwargs["budget_mode"] == BudgetMode.MANUAL

    @pytest.mark.asyncio
    async def test_ollama_wiring(self) -> None:
        proxy = ProxyServer(
            backend="ollama",
            model="ministral-3:14b",
            backend_timeout=1800.0,
        )
        mock_ctx = ContextManager.__new__(ContextManager)
        mock_ctx.budget_tokens = 4096
        with patch(
            "forge.proxy.proxy.setup_backend",
            new_callable=AsyncMock, return_value=(MagicMock(), mock_ctx),
        ) as mock_setup:
            client, _ = await proxy._setup_managed()
        assert isinstance(client, OllamaClient)
        assert client._http.timeout.read == 1800.0
        kwargs = mock_setup.await_args.kwargs
        assert kwargs["backend"] == "ollama"
        assert kwargs["model"] == "ministral-3:14b"
        assert kwargs["gguf_path"] is None
        assert kwargs["model_path"] is None
        # Client is passed through so setup_backend can wire num_ctx.
        assert kwargs["client"] is client

    @pytest.mark.asyncio
    async def test_managed_llamafile_client_is_native(self) -> None:
        # The proxy is native-only: the managed LlamafileClient is built in
        # native mode and the backend process is launched native too.
        proxy = ProxyServer(backend="llamafile", gguf="/m/x.gguf")
        mock_ctx = ContextManager.__new__(ContextManager)
        mock_ctx.budget_tokens = 8192
        with patch(
            "forge.proxy.proxy.setup_backend",
            new_callable=AsyncMock, return_value=(MagicMock(), mock_ctx),
        ) as mock_setup:
            client, _ = await proxy._setup_managed()
        assert isinstance(client, LlamafileClient)
        assert client.mode == "native"
        assert mock_setup.await_args.kwargs["mode"] == "native"


class TestBackendCapability:
    """backend_capability selects the tool-calling protocol, declared once at
    construction and frozen. native (default) = verbatim passthrough; prompt =
    opt-in prompt-injection for non-FC llama.cpp/llamafile backends."""

    def test_default_is_native(self) -> None:
        assert ProxyServer(backend_url="http://x:8080")._backend_capability == "native"

    def test_prompt_stored(self) -> None:
        proxy = ProxyServer(backend_url="http://x:8080", backend_capability="prompt")
        assert proxy._backend_capability == "prompt"

    # Guards: prompt is a llama.cpp/llamafile capability only.
    def test_prompt_rejects_vllm(self) -> None:
        with pytest.raises(ValueError, match="only supported for"):
            ProxyServer(backend_url="http://x:8000", backend="vllm", backend_capability="prompt")

    def test_prompt_rejects_ollama(self) -> None:
        with pytest.raises(ValueError, match="only supported for"):
            ProxyServer(backend="ollama", model="m", backend_capability="prompt")

    def test_prompt_rejects_anthropic_protocol(self) -> None:
        with pytest.raises(ValueError, match="not supported with the anthropic"):
            ProxyServer(
                backend_url="http://x:8080",
                backend_protocol="anthropic",
                backend_capability="prompt",
            )

    def test_prompt_allowed_for_external_llamacpp(self) -> None:
        # backend=None (external) defaults to the llama.cpp adapter → prompt ok.
        ProxyServer(backend_url="http://x:8080", backend_capability="prompt")
        ProxyServer(backend="llamafile", gguf="m.gguf", backend_capability="prompt")

    @pytest.mark.asyncio
    async def test_external_default_builds_native_client(self) -> None:
        proxy = ProxyServer(backend_url="http://localhost:8080", budget_tokens=8192)
        client, _ = await proxy._setup_external()
        assert isinstance(client, LlamafileClient)
        assert client.mode == "native"

    @pytest.mark.asyncio
    async def test_external_prompt_builds_prompt_client(self) -> None:
        proxy = ProxyServer(
            backend_url="http://localhost:8080",
            backend_capability="prompt",
            budget_tokens=8192,
        )
        client, _ = await proxy._setup_external()
        assert isinstance(client, LlamafileClient)
        assert client.mode == "prompt"

    @pytest.mark.asyncio
    async def test_managed_prompt_client_is_prompt_but_launch_native(self) -> None:
        # The managed LlamafileClient runs in prompt mode, but the backend
        # process is still launched native (--jinja present, just unused).
        proxy = ProxyServer(
            backend="llamafile", gguf="/m/x.gguf", backend_capability="prompt",
        )
        mock_ctx = ContextManager.__new__(ContextManager)
        mock_ctx.budget_tokens = 8192
        with patch(
            "forge.proxy.proxy.setup_backend",
            new_callable=AsyncMock, return_value=(MagicMock(), mock_ctx),
        ) as mock_setup:
            client, _ = await proxy._setup_managed()
        assert isinstance(client, LlamafileClient)
        assert client.mode == "prompt"
        assert mock_setup.await_args.kwargs["mode"] == "native"

    def test_native_passthrough_forwarded_to_http_server(self) -> None:
        # native → native_passthrough True; prompt → False.
        assert (ProxyServer(backend_url="http://x")._backend_capability == "native")
        assert (
            ProxyServer(backend_url="http://x", backend_capability="prompt")
            ._backend_capability == "prompt"
        )


class TestLifecycle:
    """start()/stop() thread + state management."""

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
