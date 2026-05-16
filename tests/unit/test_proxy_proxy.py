"""Tests for ProxyServer construction and wiring.

HTTPServer protocol-level tests live in test_proxy_server.py. This file
covers the ProxyServer wrapper: backend validation, mode resolution, and
client/setup_backend wiring.
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
    """__init__ validation: backend, mode, and identity field rules."""

    def test_unknown_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="backend must be one of"):
            ProxyServer(backend="bogus", url="http://localhost:8080")

    def test_backend_required(self) -> None:
        with pytest.raises(TypeError):
            ProxyServer(url="http://localhost:8080")  # type: ignore[call-arg]

    # External mode rules
    def test_external_ollama_rejected(self) -> None:
        with pytest.raises(ValueError, match="not supported in external mode"):
            ProxyServer(backend="ollama", url="http://localhost:11434")

    def test_external_rejects_gguf(self) -> None:
        with pytest.raises(ValueError, match="does not accept identity fields"):
            ProxyServer(backend="llamaserver", url="http://x:8080", gguf="m.gguf")

    def test_external_rejects_model_path(self) -> None:
        with pytest.raises(ValueError, match="does not accept identity fields"):
            ProxyServer(backend="vllm", url="http://x:8000", model_path="/m")

    def test_external_rejects_model(self) -> None:
        with pytest.raises(ValueError, match="does not accept identity fields"):
            ProxyServer(backend="llamaserver", url="http://x:8080", model="m")

    def test_external_llamaserver_ok(self) -> None:
        proxy = ProxyServer(backend="llamaserver", url="http://x:8080")
        assert proxy._backend == "llamaserver"
        assert proxy._url == "http://x:8080"

    def test_external_vllm_ok(self) -> None:
        proxy = ProxyServer(backend="vllm", url="http://x:8000")
        assert proxy._backend == "vllm"

    # Managed mode rules
    def test_managed_no_identity_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires exactly one identity field"):
            ProxyServer(backend="vllm")

    def test_managed_multiple_identities_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires exactly one identity field"):
            ProxyServer(backend="vllm", model_path="/m", gguf="x.gguf")

    def test_managed_ollama_requires_model(self) -> None:
        # gguf set instead of model — caught by per-backend check
        with pytest.raises(ValueError, match="backend='ollama' requires model"):
            ProxyServer(backend="ollama", gguf="x.gguf")

    def test_managed_llamaserver_requires_gguf(self) -> None:
        with pytest.raises(ValueError, match="requires gguf"):
            ProxyServer(backend="llamaserver", model="m")

    def test_managed_llamafile_requires_gguf(self) -> None:
        with pytest.raises(ValueError, match="requires gguf"):
            ProxyServer(backend="llamafile", model_path="/m")

    def test_managed_vllm_requires_model_path(self) -> None:
        with pytest.raises(ValueError, match="requires model_path"):
            ProxyServer(backend="vllm", gguf="x.gguf")

    def test_managed_ok(self) -> None:
        ProxyServer(backend="llamaserver", gguf="m.gguf")
        ProxyServer(backend="llamafile", gguf="m.gguf")
        ProxyServer(backend="vllm", model_path="/m")
        ProxyServer(backend="ollama", model="llama3")

    # Serialize auto-detection
    def test_serialize_auto_managed_true(self) -> None:
        proxy = ProxyServer(backend="vllm", model_path="/m")
        assert proxy._serialize is True

    def test_serialize_auto_external_false(self) -> None:
        proxy = ProxyServer(backend="vllm", url="http://x:8000")
        assert proxy._serialize is False

    def test_serialize_override(self) -> None:
        proxy = ProxyServer(backend="vllm", url="http://x:8000", serialize=True)
        assert proxy._serialize is True


class TestSetupExternal:
    """External mode constructs the right client and resolves budget."""

    @pytest.mark.asyncio
    async def test_llamaserver_uses_llamafile_client(self) -> None:
        proxy = ProxyServer(
            backend="llamaserver", url="http://localhost:8080", budget_tokens=8192,
        )
        client, ctx = await proxy._setup_external()
        assert isinstance(client, LlamafileClient)
        assert client.base_url == "http://localhost:8080/v1"
        assert ctx.budget_tokens == 8192

    @pytest.mark.asyncio
    async def test_llamafile_uses_llamafile_client(self) -> None:
        proxy = ProxyServer(
            backend="llamafile", url="http://localhost:8080", budget_tokens=8192,
        )
        client, _ = await proxy._setup_external()
        assert isinstance(client, LlamafileClient)

    @pytest.mark.asyncio
    async def test_vllm_uses_vllm_client(self) -> None:
        proxy = ProxyServer(
            backend="vllm", url="http://localhost:8000", budget_tokens=8192,
        )
        client, ctx = await proxy._setup_external()
        assert isinstance(client, VLLMClient)
        assert client.base_url == "http://localhost:8000/v1"
        assert ctx.budget_tokens == 8192

    @pytest.mark.asyncio
    async def test_url_v1_suffix_preserved(self) -> None:
        proxy = ProxyServer(
            backend="vllm", url="http://localhost:8000/v1", budget_tokens=8192,
        )
        client, _ = await proxy._setup_external()
        assert client.base_url == "http://localhost:8000/v1"

    @pytest.mark.asyncio
    async def test_url_trailing_slash_stripped(self) -> None:
        proxy = ProxyServer(
            backend="vllm", url="http://localhost:8000/", budget_tokens=8192,
        )
        client, _ = await proxy._setup_external()
        assert client.base_url == "http://localhost:8000/v1"

    @pytest.mark.asyncio
    async def test_budget_from_backend_when_unspecified(self) -> None:
        proxy = ProxyServer(backend="vllm", url="http://localhost:8000")
        with patch.object(
            VLLMClient, "get_context_length",
            new_callable=AsyncMock, return_value=32768,
        ):
            _, ctx = await proxy._setup_external()
        assert ctx.budget_tokens == 32768

    @pytest.mark.asyncio
    async def test_budget_unresolvable_raises(self) -> None:
        proxy = ProxyServer(backend="llamaserver", url="http://localhost:8080")
        with patch.object(
            LlamafileClient, "get_context_length",
            new_callable=AsyncMock, return_value=None,
        ), pytest.raises(RuntimeError, match="did not report a context length"):
            await proxy._setup_external()


class TestSetupManaged:
    """Managed mode delegates to setup_backend with the right args."""

    @pytest.mark.asyncio
    async def test_llamaserver_wiring(self) -> None:
        proxy = ProxyServer(
            backend="llamaserver",
            gguf="/models/x.gguf",
            backend_port=8080,
            budget_mode=BudgetMode.FORGE_FAST,
            extra_flags=["-ngl", "99"],
        )
        mock_server = MagicMock()
        mock_ctx = ContextManager.__new__(ContextManager)
        mock_ctx.budget_tokens = 16384

        with patch(
            "forge.proxy.proxy.setup_backend",
            new_callable=AsyncMock, return_value=(mock_server, mock_ctx),
        ) as mock_setup:
            client, ctx = await proxy._setup_managed()

        assert isinstance(client, LlamafileClient)
        assert client.base_url == "http://localhost:8080/v1"
        mock_setup.assert_awaited_once()
        kwargs = mock_setup.await_args.kwargs
        assert kwargs["backend"] == "llamaserver"
        assert kwargs["gguf_path"] == "/models/x.gguf"
        assert kwargs["model"] is None
        assert kwargs["model_path"] is None
        assert kwargs["port"] == 8080
        assert kwargs["budget_mode"] == BudgetMode.FORGE_FAST
        assert kwargs["extra_flags"] == ["-ngl", "99"]
        assert kwargs["client"] is client
        assert proxy._server_manager is mock_server
        assert ctx is mock_ctx

    @pytest.mark.asyncio
    async def test_llamafile_wiring(self) -> None:
        proxy = ProxyServer(backend="llamafile", gguf="/m/x.gguf", backend_port=9090)
        mock_ctx = ContextManager.__new__(ContextManager)
        mock_ctx.budget_tokens = 8192
        with patch(
            "forge.proxy.proxy.setup_backend",
            new_callable=AsyncMock, return_value=(MagicMock(), mock_ctx),
        ) as mock_setup:
            client, _ = await proxy._setup_managed()
        assert isinstance(client, LlamafileClient)
        assert client.base_url == "http://localhost:9090/v1"
        assert mock_setup.await_args.kwargs["backend"] == "llamafile"

    @pytest.mark.asyncio
    async def test_vllm_wiring(self) -> None:
        proxy = ProxyServer(
            backend="vllm", model_path="/models/awq", backend_port=8000,
            budget_tokens=113000, budget_mode=BudgetMode.MANUAL,
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
        kwargs = mock_setup.await_args.kwargs
        assert kwargs["backend"] == "vllm"
        assert kwargs["model_path"] == "/models/awq"
        assert kwargs["gguf_path"] is None
        assert kwargs["model"] is None
        assert kwargs["manual_tokens"] == 113000
        assert kwargs["budget_mode"] == BudgetMode.MANUAL

    @pytest.mark.asyncio
    async def test_ollama_wiring(self) -> None:
        proxy = ProxyServer(backend="ollama", model="ministral-3:14b")
        mock_ctx = ContextManager.__new__(ContextManager)
        mock_ctx.budget_tokens = 4096
        with patch(
            "forge.proxy.proxy.setup_backend",
            new_callable=AsyncMock, return_value=(MagicMock(), mock_ctx),
        ) as mock_setup:
            client, _ = await proxy._setup_managed()
        assert isinstance(client, OllamaClient)
        kwargs = mock_setup.await_args.kwargs
        assert kwargs["backend"] == "ollama"
        assert kwargs["model"] == "ministral-3:14b"
        # Client is passed through so setup_backend can wire num_ctx
        assert kwargs["client"] is client


class TestLifecycle:
    """start()/stop() thread + state management."""

    def test_url_property(self) -> None:
        proxy = ProxyServer(
            backend="vllm", url="http://localhost:8000",
            host="0.0.0.0", port=9000,
        )
        assert proxy.url == "http://0.0.0.0:9000"

    def test_stop_before_start_noop(self) -> None:
        proxy = ProxyServer(backend="vllm", url="http://localhost:8000")
        # Should not raise
        proxy.stop()

    def test_start_twice_idempotent(self) -> None:
        proxy = ProxyServer(backend="vllm", url="http://localhost:8000")
        proxy._started = True
        # Second start should return immediately without spawning a thread
        proxy.start()
        assert proxy._thread is None
