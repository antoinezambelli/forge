"""Tests for forge.server — BudgetMode enum and ServerManager."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest

from forge.context.manager import ContextManager
from forge.context.strategies import TieredCompact
from forge.errors import BackendError, BudgetResolutionError
from forge.server import BudgetMode, ServerManager, setup_backend


# ── BudgetMode ──────────────────────────────────────────────────


class TestBudgetMode:
    """BudgetMode enum basics."""

    def test_budget_mode_values(self) -> None:
        assert BudgetMode.BACKEND.value == "backend"
        assert BudgetMode.MANUAL.value == "manual"
        assert BudgetMode.FORGE_FULL.value == "forge-full"
        assert BudgetMode.FORGE_FAST.value == "forge-fast"

    def test_budget_mode_is_string_enum(self) -> None:
        assert BudgetMode.BACKEND == "backend"
        assert BudgetMode.MANUAL == "manual"
        assert BudgetMode.FORGE_FULL == "forge-full"
        assert BudgetMode.FORGE_FAST == "forge-fast"


# ── ServerManager construction ──────────────────────────────────


class TestServerManagerInit:
    """Constructor / attribute checks."""

    def test_init_ollama(self) -> None:
        sm = ServerManager(backend="ollama")
        assert sm._backend == "ollama"
        assert sm._proc is None
        assert sm._current_model is None

    def test_init_llamaserver(self) -> None:
        sm = ServerManager(backend="llamaserver", port=9090, models_dir="/models")
        assert sm._backend == "llamaserver"
        assert sm._port == 9090
        assert sm._models_dir is not None


# ── ServerManager.start() ───────────────────────────────────────


class TestServerManagerStart:
    """start() process launching and reuse logic."""

    @pytest.fixture()
    def sm(self) -> ServerManager:
        return ServerManager(backend="llamaserver", port=8080)

    @pytest.mark.asyncio
    async def test_start_launches_process(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start("llama3", "/models/llama3.gguf")

        args = mock_popen.call_args[0][0]
        assert "llama-server" in args
        assert "-m" in args
        assert "/models/llama3.gguf" in args
        assert "-ngl" in args
        assert "999" in args
        assert "--port" in args
        assert "8080" in args

    @pytest.mark.asyncio
    async def test_start_native_mode_adds_jinja(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start("llama3", "/models/llama3.gguf", mode="native")

        args = mock_popen.call_args[0][0]
        assert "--jinja" in args

    @pytest.mark.asyncio
    async def test_start_prompt_mode_no_jinja(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start("llama3", "/models/llama3.gguf", mode="prompt")

        args = mock_popen.call_args[0][0]
        assert "--jinja" not in args

    @pytest.mark.asyncio
    async def test_start_with_extra_flags(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start(
                "qwen3", "/models/qwen3.gguf",
                extra_flags=["--reasoning-format", "auto"],
            )

        args = mock_popen.call_args[0][0]
        assert "--reasoning-format" in args
        assert "auto" in args

    @pytest.mark.asyncio
    async def test_start_with_ctx_override(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start("llama3", "/models/llama3.gguf", ctx_override=8000)

        args = mock_popen.call_args[0][0]
        assert "-c" in args
        assert "8000" in args

    @pytest.mark.asyncio
    async def test_start_reuses_same_config(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start("llama3", "/models/llama3.gguf", mode="native")
            await sm.start("llama3", "/models/llama3.gguf", mode="native")

        assert mock_popen.call_count == 1

    @pytest.mark.asyncio
    async def test_start_restarts_on_mode_change(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
            patch.object(sm, "stop", new_callable=AsyncMock) as mock_stop,
        ):
            # First start — stop is called but nothing to stop
            await sm.start("llama3", "/models/llama3.gguf", mode="native")
            # Simulate state after first start
            sm._current_model = "llama3"
            sm._current_mode = "native"
            sm._current_ctx = None

            # Second start with different mode — should restart
            await sm.start("llama3", "/models/llama3.gguf", mode="prompt")

        assert mock_popen.call_count == 2
        # stop() called before each start
        assert mock_stop.call_count >= 2

    @pytest.mark.asyncio
    async def test_start_restarts_on_ctx_change(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
            patch.object(sm, "stop", new_callable=AsyncMock) as mock_stop,
        ):
            await sm.start("llama3", "/models/llama3.gguf")
            sm._current_model = "llama3"
            sm._current_mode = "native"
            sm._current_ctx = None

            await sm.start("llama3", "/models/llama3.gguf", ctx_override=8000)

        assert mock_popen.call_count == 2

    @pytest.mark.asyncio
    async def test_start_restarts_on_extra_flags_change(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
            patch.object(sm, "stop", new_callable=AsyncMock) as mock_stop,
        ):
            await sm.start("llama3", "/models/llama3.gguf", mode="native")
            sm._current_model = "llama3"
            sm._current_mode = "native"
            sm._current_ctx = None

            # Same model/mode/ctx but different extra_flags — should restart
            await sm.start(
                "llama3", "/models/llama3.gguf", mode="native",
                extra_flags=["--reasoning-format", "auto"],
            )

        assert mock_popen.call_count == 2

    @pytest.mark.asyncio
    async def test_start_reuses_same_config_with_extra_flags(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start(
                "llama3", "/models/llama3.gguf", mode="native",
                extra_flags=["--reasoning-format", "auto"],
            )
            await sm.start(
                "llama3", "/models/llama3.gguf", mode="native",
                extra_flags=["--reasoning-format", "auto"],
            )

        assert mock_popen.call_count == 1

    @pytest.mark.asyncio
    async def test_start_noop_for_ollama(self) -> None:
        sm = ServerManager(backend="ollama")
        with patch("forge.server.subprocess.Popen") as mock_popen:
            await sm.start("llama3", "/models/llama3.gguf")

        mock_popen.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_llamafile_uses_runtime_binary(self, tmp_path: Path) -> None:
        sm = ServerManager(backend="llamafile", port=8080)
        # Create a fake llamafile runtime in tmp_path
        runtime = tmp_path / "llamafile-0.9.2.exe"
        runtime.touch()
        model_path = tmp_path / "Model.Q4_K_M.llamafile"
        model_path.touch()

        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start("llama3", str(model_path), mode="prompt")

        args = mock_popen.call_args[0][0]
        assert str(runtime) in args
        assert "--server" in args
        assert "--nobrowser" in args
        assert "-m" in args
        assert str(model_path) in args
        assert "llama-server" not in args

    @pytest.mark.asyncio
    async def test_start_llamafile_no_runtime_raises(self, tmp_path: Path) -> None:
        sm = ServerManager(backend="llamafile", port=8080)
        model_path = tmp_path / "Model.Q4_K_M.llamafile"
        model_path.touch()
        # No llamafile-* runtime in tmp_path
        with pytest.raises(FileNotFoundError, match="No llamafile runtime"):
            await sm.start("llama3", str(model_path), mode="prompt")

    @pytest.mark.asyncio
    async def test_start_llamafile_picks_highest_version(self, tmp_path: Path) -> None:
        sm = ServerManager(backend="llamafile", port=8080)
        # Create multiple versions
        (tmp_path / "llamafile-0.8.0.exe").touch()
        (tmp_path / "llamafile-0.9.2.exe").touch()
        (tmp_path / "llamafile-0.9.0.exe").touch()
        model_path = tmp_path / "Model.llamafile"
        model_path.touch()

        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start("llama3", str(model_path), mode="prompt")

        args = mock_popen.call_args[0][0]
        assert str(tmp_path / "llamafile-0.9.2.exe") in args

    @pytest.mark.asyncio
    async def test_start_with_cache_type_k_v(self) -> None:
        sm = ServerManager(backend="llamaserver", port=8080)
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start(
                "llama3", "/models/llama3.gguf",
                cache_type_k="q8_0", cache_type_v="q8_0",
            )

        cmd = mock_popen.call_args[0][0]
        assert "--cache-type-k" in cmd
        assert "q8_0" in cmd
        assert "--cache-type-v" in cmd
        # Flags appear in order
        k_idx = cmd.index("--cache-type-k")
        v_idx = cmd.index("--cache-type-v")
        assert cmd[k_idx + 1] == "q8_0"
        assert cmd[v_idx + 1] == "q8_0"

    @pytest.mark.asyncio
    async def test_start_without_cache_type_omits_flags(self) -> None:
        sm = ServerManager(backend="llamaserver", port=8080)
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start("llama3", "/models/llama3.gguf")

        cmd = mock_popen.call_args[0][0]
        assert "--cache-type-k" not in cmd
        assert "--cache-type-v" not in cmd

    @pytest.mark.asyncio
    async def test_start_restarts_on_cache_type_change(self) -> None:
        sm = ServerManager(backend="llamaserver", port=8080)
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc),
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
            patch("forge.server.asyncio.sleep", new_callable=AsyncMock),
        ):
            await sm.start("llama3", "/models/llama3.gguf", cache_type_k="q8_0")
            assert sm._current_cache_type_k == "q8_0"

            # Same config — should reuse (no restart)
            await sm.start("llama3", "/models/llama3.gguf", cache_type_k="q8_0")

            # Different cache type — should restart
            await sm.start("llama3", "/models/llama3.gguf", cache_type_k="q4_0")
            assert sm._current_cache_type_k == "q4_0"

    @pytest.mark.asyncio
    async def test_start_with_n_slots(self) -> None:
        sm = ServerManager(backend="llamaserver", port=8080)
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start("llama3", "/models/llama3.gguf", n_slots=2)

        cmd = mock_popen.call_args[0][0]
        assert "--parallel" in cmd
        idx = cmd.index("--parallel")
        assert cmd[idx + 1] == "2"
        assert sm._current_n_slots == 2

    @pytest.mark.asyncio
    async def test_start_without_n_slots_omits_flag(self) -> None:
        sm = ServerManager(backend="llamaserver", port=8080)
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start("llama3", "/models/llama3.gguf")

        cmd = mock_popen.call_args[0][0]
        assert "--parallel" not in cmd


# ── ServerManager.stop() ────────────────────────────────────────


class TestServerManagerStop:
    """stop() termination and cleanup."""

    @pytest.mark.asyncio
    async def test_stop_terminates_process(self) -> None:
        sm = ServerManager(backend="llamaserver")
        mock_proc = MagicMock()
        sm._proc = mock_proc

        with patch("forge.server.asyncio.sleep", new_callable=AsyncMock):
            await sm.stop()

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=10)

    @pytest.mark.asyncio
    async def test_stop_kills_on_timeout(self) -> None:
        sm = ServerManager(backend="llamaserver")
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 10), None]
        sm._proc = mock_proc

        with patch("forge.server.asyncio.sleep", new_callable=AsyncMock):
            await sm.stop()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_ollama_runs_stop_command(self) -> None:
        sm = ServerManager(backend="ollama")
        sm._current_model = "ministral:14b"

        with patch("forge.server.subprocess.run") as mock_run:
            await sm.stop()

        mock_run.assert_called_once_with(["ollama", "stop", "ministral:14b"])

    @pytest.mark.asyncio
    async def test_stop_ollama_noop_when_no_model(self) -> None:
        sm = ServerManager(backend="ollama")

        with patch("forge.server.subprocess.run") as mock_run:
            await sm.stop()

        mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_clears_state(self) -> None:
        sm = ServerManager(backend="llamaserver")
        mock_proc = MagicMock()
        sm._proc = mock_proc
        sm._current_model = "llama3"
        sm._current_mode = "native"
        sm._current_ctx = 8000
        sm._current_flags = ("--reasoning-format", "auto")

        with patch("forge.server.asyncio.sleep", new_callable=AsyncMock):
            await sm.stop()

        assert sm._proc is None
        assert sm._current_model is None
        assert sm._current_mode is None
        assert sm._current_ctx is None
        assert sm._current_flags == ()
        assert sm._current_cache_type_k is None
        assert sm._current_cache_type_v is None
        assert sm._current_n_slots is None


# ── ServerManager.get_server_context() ──────────────────────────


class TestGetServerContext:
    """get_server_context() /props parsing."""

    @pytest.mark.asyncio
    async def test_get_server_context_parses_props(self) -> None:
        sm = ServerManager(backend="llamaserver")
        props_response = {"default_generation_settings": {"n_ctx": 13568}}

        with patch.object(sm, "query_props", new_callable=AsyncMock, return_value=props_response):
            result = await sm.get_server_context()

        assert result == 13568

    @pytest.mark.asyncio
    async def test_get_server_context_raises_on_missing_field(self) -> None:
        sm = ServerManager(backend="llamaserver")
        props_response = {"default_generation_settings": {}}

        with patch.object(sm, "query_props", new_callable=AsyncMock, return_value=props_response):
            with pytest.raises(BudgetResolutionError):
                await sm.get_server_context()

    @pytest.mark.asyncio
    async def test_get_server_context_raises_on_connect_error(self) -> None:
        sm = ServerManager(backend="llamaserver")

        with patch.object(sm, "query_props", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")):
            with pytest.raises(BudgetResolutionError) as exc_info:
                await sm.get_server_context()
            assert exc_info.value.__cause__ is not None


# ── ServerManager.resolve_budget() ──────────────────────────────


class TestResolveBudget:
    """resolve_budget() mode × backend matrix."""

    # -- backend mode --

    @pytest.mark.asyncio
    async def test_resolve_budget_backend_ollama(self) -> None:
        sm = ServerManager(backend="ollama")
        hw = MagicMock(vram_total_gb=12.0)
        with patch("forge.server.detect_hardware", return_value=hw):
            result = await sm.resolve_budget(BudgetMode.BACKEND)
        assert result == 4096

    @pytest.mark.asyncio
    async def test_resolve_budget_backend_ollama_24gb(self) -> None:
        sm = ServerManager(backend="ollama")
        hw = MagicMock(vram_total_gb=24.0)
        with patch("forge.server.detect_hardware", return_value=hw):
            result = await sm.resolve_budget(BudgetMode.BACKEND)
        assert result == 32768

    @pytest.mark.asyncio
    async def test_resolve_budget_backend_ollama_48gb(self) -> None:
        sm = ServerManager(backend="ollama")
        hw = MagicMock(vram_total_gb=48.0)
        with patch("forge.server.detect_hardware", return_value=hw):
            result = await sm.resolve_budget(BudgetMode.BACKEND)
        assert result == 262144

    @pytest.mark.asyncio
    async def test_resolve_budget_backend_ollama_no_gpu(self) -> None:
        sm = ServerManager(backend="ollama")
        with patch("forge.server.detect_hardware", return_value=None):
            result = await sm.resolve_budget(BudgetMode.BACKEND)
        assert result == 4096

    @pytest.mark.asyncio
    async def test_resolve_budget_backend_llamaserver(self) -> None:
        sm = ServerManager(backend="llamaserver")
        with patch.object(sm, "get_server_context", new_callable=AsyncMock, return_value=13568):
            result = await sm.resolve_budget(BudgetMode.BACKEND)
        assert result == 13568

    # -- manual mode --

    @pytest.mark.asyncio
    async def test_resolve_budget_manual_ollama_returns_manual_tokens(self) -> None:
        sm = ServerManager(backend="ollama")
        result = await sm.resolve_budget(BudgetMode.MANUAL, manual_tokens=8000)
        assert result == 8000

    @pytest.mark.asyncio
    async def test_resolve_budget_manual_llamaserver_returns_server_context(self) -> None:
        sm = ServerManager(backend="llamaserver")
        with patch.object(sm, "get_server_context", new_callable=AsyncMock, return_value=8000):
            result = await sm.resolve_budget(BudgetMode.MANUAL, manual_tokens=8000)
        assert result == 8000

    @pytest.mark.asyncio
    async def test_resolve_budget_manual_no_tokens_raises(self) -> None:
        sm = ServerManager(backend="ollama")
        with pytest.raises(ValueError, match="manual mode requires manual_tokens"):
            await sm.resolve_budget(BudgetMode.MANUAL)

    # -- forge-full mode --

    @pytest.mark.asyncio
    async def test_resolve_budget_forge_full_ollama(self) -> None:
        sm = ServerManager(backend="ollama")
        hw = MagicMock(vram_total_gb=12.0)
        with patch("forge.server.detect_hardware", return_value=hw):
            result = await sm.resolve_budget(BudgetMode.FORGE_FULL)
        assert result == 4096  # same as backend for Ollama

    @pytest.mark.asyncio
    async def test_resolve_budget_forge_full_llamaserver(self) -> None:
        sm = ServerManager(backend="llamaserver")
        with patch.object(sm, "get_server_context", new_callable=AsyncMock, return_value=13568):
            result = await sm.resolve_budget(BudgetMode.FORGE_FULL)
        assert result == 13568

    # -- forge-fast mode --

    @pytest.mark.asyncio
    async def test_resolve_budget_forge_fast_ollama(self) -> None:
        sm = ServerManager(backend="ollama")
        hw = MagicMock(vram_total_gb=12.0)
        with patch("forge.server.detect_hardware", return_value=hw):
            result = await sm.resolve_budget(BudgetMode.FORGE_FAST)
        assert result == 2048  # half of 4096 tier for <24 GB

    @pytest.mark.asyncio
    async def test_resolve_budget_forge_fast_llamaserver(self) -> None:
        sm = ServerManager(backend="llamaserver")
        with patch.object(sm, "get_server_context", new_callable=AsyncMock, return_value=6784):
            result = await sm.resolve_budget(BudgetMode.FORGE_FAST)
        assert result == 6784  # caller already restarted with -c

    # -- error cases --

    @pytest.mark.asyncio
    async def test_resolve_budget_llamaserver_no_context_raises(self) -> None:
        sm = ServerManager(backend="llamaserver")
        with patch.object(sm, "get_server_context", new_callable=AsyncMock, side_effect=BudgetResolutionError()):
            with pytest.raises(BudgetResolutionError):
                await sm.resolve_budget(BudgetMode.BACKEND)

    @pytest.mark.asyncio
    async def test_resolve_budget_manual_llamaserver_no_context_raises(self) -> None:
        sm = ServerManager(backend="llamaserver")
        with patch.object(sm, "get_server_context", new_callable=AsyncMock, side_effect=BudgetResolutionError()):
            with pytest.raises(BudgetResolutionError):
                await sm.resolve_budget(BudgetMode.MANUAL, manual_tokens=8000)


# ── ServerManager.start_with_budget() ─────────────────────────────


class TestStartWithBudget:
    """start_with_budget() mode-specific startup dance."""

    @pytest.mark.asyncio
    async def test_start_with_budget_backend(self) -> None:
        sm = ServerManager(backend="llamaserver")
        with (
            patch.object(sm, "start", new_callable=AsyncMock) as mock_start,
            patch.object(sm, "get_server_context", new_callable=AsyncMock, return_value=13568),
        ):
            result = await sm.start_with_budget(
                "llama3", "/models/llama3.gguf",
                budget_mode=BudgetMode.BACKEND,
            )

        mock_start.assert_called_once_with(
            "llama3", "/models/llama3.gguf", "native", None, ctx_override=None,
            cache_type_k=None, cache_type_v=None, n_slots=None, kv_unified=False,
        )
        assert result == 13568

    @pytest.mark.asyncio
    async def test_start_with_budget_manual(self) -> None:
        sm = ServerManager(backend="llamaserver")
        with (
            patch.object(sm, "start", new_callable=AsyncMock) as mock_start,
            patch.object(sm, "get_server_context", new_callable=AsyncMock, return_value=8000),
        ):
            result = await sm.start_with_budget(
                "llama3", "/models/llama3.gguf",
                budget_mode=BudgetMode.MANUAL,
                manual_tokens=8000,
            )

        mock_start.assert_called_once_with(
            "llama3", "/models/llama3.gguf", "native", None, ctx_override=8000,
            cache_type_k=None, cache_type_v=None, n_slots=None, kv_unified=False,
        )
        assert result == 8000

    @pytest.mark.asyncio
    async def test_start_with_budget_manual_no_tokens_raises(self) -> None:
        sm = ServerManager(backend="llamaserver")
        with pytest.raises(ValueError, match="manual mode requires manual_tokens"):
            await sm.start_with_budget(
                "llama3", "/models/llama3.gguf",
                budget_mode=BudgetMode.MANUAL,
            )

    @pytest.mark.asyncio
    async def test_start_with_budget_forge_full(self) -> None:
        sm = ServerManager(backend="llamaserver")
        with (
            patch.object(sm, "start", new_callable=AsyncMock) as mock_start,
            patch.object(sm, "get_server_context", new_callable=AsyncMock, return_value=13568),
        ):
            result = await sm.start_with_budget(
                "llama3", "/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FULL,
            )

        mock_start.assert_called_once_with(
            "llama3", "/models/llama3.gguf", "native", None, ctx_override=None,
            cache_type_k=None, cache_type_v=None, n_slots=None, kv_unified=False,
        )
        assert result == 13568

    @pytest.mark.asyncio
    async def test_start_with_budget_forge_fast(self) -> None:
        """FORGE_FAST: start → read 13568 → restart with 6784 → read 6784."""
        sm = ServerManager(backend="llamaserver")
        # First get_server_context returns 13568 (auto-tuned max),
        # second returns 6784 (after restart with -c 6784)
        with (
            patch.object(sm, "start", new_callable=AsyncMock) as mock_start,
            patch.object(
                sm, "get_server_context", new_callable=AsyncMock,
                side_effect=[13568, 6784],
            ),
        ):
            result = await sm.start_with_budget(
                "llama3", "/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FAST,
            )

        assert mock_start.call_count == 2
        # Phase 1: start without -c
        mock_start.assert_any_call(
            "llama3", "/models/llama3.gguf", "native", None, ctx_override=None,
            cache_type_k=None, cache_type_v=None, n_slots=None, kv_unified=False,
        )
        # Phase 2: restart with half (13568 // 2 = 6784)
        mock_start.assert_any_call(
            "llama3", "/models/llama3.gguf", "native", None, ctx_override=6784,
            cache_type_k=None, cache_type_v=None, n_slots=None, kv_unified=False,
        )
        assert result == 6784

    @pytest.mark.asyncio
    async def test_start_with_budget_forge_fast_no_ctx_raises(self) -> None:
        sm = ServerManager(backend="llamaserver")
        with (
            patch.object(sm, "start", new_callable=AsyncMock),
            patch.object(sm, "get_server_context", new_callable=AsyncMock, side_effect=BudgetResolutionError()),
        ):
            with pytest.raises(BudgetResolutionError):
                await sm.start_with_budget(
                    "llama3", "/models/llama3.gguf",
                    budget_mode=BudgetMode.FORGE_FAST,
                )

    @pytest.mark.asyncio
    async def test_start_with_budget_ollama_backend(self) -> None:
        sm = ServerManager(backend="ollama")
        hw = MagicMock(vram_total_gb=12.0)
        with patch("forge.server.detect_hardware", return_value=hw):
            result = await sm.start_with_budget(
                "llama3", "/models/llama3.gguf",
                budget_mode=BudgetMode.BACKEND,
            )
        assert result == 4096
        assert sm._current_model == "llama3"

    @pytest.mark.asyncio
    async def test_start_with_budget_ollama_forge_fast_halves(self) -> None:
        """Ollama forge-fast returns half of VRAM tier."""
        sm = ServerManager(backend="ollama")
        hw = MagicMock(vram_total_gb=12.0)
        with patch("forge.server.detect_hardware", return_value=hw):
            result = await sm.start_with_budget(
                "llama3", "/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FAST,
            )
        assert result == 2048  # half of 4096 tier for <24 GB

    @pytest.mark.asyncio
    async def test_forge_fast_multi_slot_no_double_divide(self) -> None:
        """FORGE_FAST with 2 slots: recovers total before halving."""
        sm = ServerManager(backend="llamaserver")
        # /props reports 35K per-slot (from 70K total / 2 slots)
        # After restart with -c 35K (half of 70K), /props reports 17.5K per-slot
        with (
            patch.object(sm, "start", new_callable=AsyncMock) as mock_start,
            patch.object(
                sm, "get_server_context", new_callable=AsyncMock,
                side_effect=[35000, 17500],
            ),
        ):
            result = await sm.start_with_budget(
                "llama3", "/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FAST,
                n_slots=2,
            )

        assert mock_start.call_count == 2
        # Phase 2: -c should be 35000 (half of 70K total), NOT 17500 (half of per-slot)
        mock_start.assert_any_call(
            "llama3", "/models/llama3.gguf", "native", None, ctx_override=35000,
            cache_type_k=None, cache_type_v=None, n_slots=2, kv_unified=False,
        )
        # Budget returned is per-slot (non-unified): 17500
        assert result == 17500

    @pytest.mark.asyncio
    async def test_forge_fast_single_slot_unchanged(self) -> None:
        """FORGE_FAST with 1 slot: same behavior as before."""
        sm = ServerManager(backend="llamaserver")
        with (
            patch.object(sm, "start", new_callable=AsyncMock) as mock_start,
            patch.object(
                sm, "get_server_context", new_callable=AsyncMock,
                side_effect=[13568, 6784],
            ),
        ):
            result = await sm.start_with_budget(
                "llama3", "/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FAST,
                n_slots=1,
            )

        mock_start.assert_any_call(
            "llama3", "/models/llama3.gguf", "native", None, ctx_override=6784,
            cache_type_k=None, cache_type_v=None, n_slots=1, kv_unified=False,
        )
        assert result == 6784

    @pytest.mark.asyncio
    async def test_kv_unified_returns_full_budget(self) -> None:
        """kv_unified=True with 2 slots: /props reports full context, budget matches."""
        sm = ServerManager(backend="llamaserver")

        async def fake_start(*args, **kwargs):
            sm._current_kv_unified = kwargs.get("kv_unified", False)
            sm._current_n_slots = kwargs.get("n_slots")

        # With kv_unified, /props reports full context (not divided by slots)
        with (
            patch.object(sm, "start", side_effect=fake_start),
            patch.object(sm, "get_server_context", new_callable=AsyncMock, return_value=70000),
        ):
            result = await sm.start_with_budget(
                "llama3", "/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FULL,
                n_slots=2,
                kv_unified=True,
            )

        # Budget is what /props reports — no multiplication needed
        assert result == 70000

    @pytest.mark.asyncio
    async def test_kv_unified_single_slot_no_change(self) -> None:
        """kv_unified with 1 slot: same as without — /props reports full context."""
        sm = ServerManager(backend="llamaserver")

        with (
            patch.object(sm, "start", new_callable=AsyncMock),
            patch.object(sm, "get_server_context", new_callable=AsyncMock, return_value=70000),
        ):
            result = await sm.start_with_budget(
                "llama3", "/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FULL,
                n_slots=1,
                kv_unified=True,
            )

        assert result == 70000

    @pytest.mark.asyncio
    async def test_non_unified_returns_per_slot_budget(self) -> None:
        """Without kv_unified, 2 slots: budget is per-slot context."""
        sm = ServerManager(backend="llamaserver")
        with (
            patch.object(sm, "start", new_callable=AsyncMock),
            patch.object(sm, "get_server_context", new_callable=AsyncMock, return_value=35000),
        ):
            result = await sm.start_with_budget(
                "llama3", "/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FULL,
                n_slots=2,
                kv_unified=False,
            )

        # Non-unified: per-slot is the correct budget
        assert result == 35000

    @pytest.mark.asyncio
    async def test_kv_unified_injects_flag(self) -> None:
        """kv_unified=True passes --kv-unified to start()."""
        sm = ServerManager(backend="llamaserver")
        with (
            patch.object(sm, "start", new_callable=AsyncMock) as mock_start,
            patch.object(sm, "get_server_context", new_callable=AsyncMock, return_value=35000),
        ):
            await sm.start_with_budget(
                "llama3", "/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FULL,
                n_slots=2,
                kv_unified=True,
            )

        mock_start.assert_called_once_with(
            "llama3", "/models/llama3.gguf", "native", None, ctx_override=None,
            cache_type_k=None, cache_type_v=None, n_slots=2, kv_unified=True,
        )

    @pytest.mark.asyncio
    async def test_forge_fast_kv_unified_multi_slot(self) -> None:
        """FORGE_FAST + kv_unified + 2 slots: /props reports full context."""
        sm = ServerManager(backend="llamaserver")

        # With unified, /props reports full context (not per-slot).
        # Phase 1: /props reports 70K (full). FORGE_FAST halves: -c 35K.
        # Phase 2: /props reports 35K (full at new size). Budget = 35K.
        with (
            patch.object(sm, "start", new_callable=AsyncMock),
            patch.object(
                sm, "get_server_context", new_callable=AsyncMock,
                side_effect=[70000, 35000],
            ),
        ):
            result = await sm.start_with_budget(
                "llama3", "/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FAST,
                n_slots=2,
                kv_unified=True,
            )

        assert result == 35000

    @pytest.mark.asyncio
    async def test_forge_fast_kv_unified_single_slot(self) -> None:
        """FORGE_FAST + kv_unified + 1 slot: straightforward halving."""
        sm = ServerManager(backend="llamaserver")
        with (
            patch.object(sm, "start", new_callable=AsyncMock) as mock_start,
            patch.object(
                sm, "get_server_context", new_callable=AsyncMock,
                side_effect=[70000, 35000],
            ),
        ):
            result = await sm.start_with_budget(
                "llama3", "/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FAST,
                n_slots=1,
                kv_unified=True,
            )

        mock_start.assert_any_call(
            "llama3", "/models/llama3.gguf", "native", None, ctx_override=35000,
            cache_type_k=None, cache_type_v=None, n_slots=1, kv_unified=True,
        )
        assert result == 35000


# ── setup_backend() ──────────────────────────────────────────────


class TestSetupBackend:
    """setup_backend() convenience function."""

    @pytest.mark.asyncio
    async def test_setup_backend_returns_manager_and_ctx(self) -> None:
        with (
            patch.object(
                ServerManager, "start_with_budget",
                new_callable=AsyncMock, return_value=13568,
            ),
        ):
            server, ctx = await setup_backend(
                backend="llamaserver",
                gguf_path="/models/llama3.gguf",
            )

        assert isinstance(server, ServerManager)
        assert isinstance(ctx, ContextManager)
        assert ctx.budget_tokens == 13568

    @pytest.mark.asyncio
    async def test_setup_backend_ctx_uses_tiered_compact(self) -> None:
        with (
            patch.object(
                ServerManager, "start_with_budget",
                new_callable=AsyncMock, return_value=13568,
            ),
        ):
            _, ctx = await setup_backend(
                backend="llamaserver",
                gguf_path="/models/llama3.gguf",
            )

        assert isinstance(ctx.strategy, TieredCompact)

    @pytest.mark.asyncio
    async def test_setup_backend_passes_compact_threshold(self) -> None:
        with (
            patch.object(
                ServerManager, "start_with_budget",
                new_callable=AsyncMock, return_value=13568,
            ),
        ):
            _, ctx = await setup_backend(
                backend="llamaserver",
                gguf_path="/models/llama3.gguf",
                compact_threshold=0.5,
            )

        # compact_threshold now lives on the strategy, not ContextManager
        assert ctx.strategy._phase_triggers == (0.5, 0.5, 0.5)

    @pytest.mark.asyncio
    async def test_setup_backend_passes_on_compact(self) -> None:
        callback = MagicMock()
        with (
            patch.object(
                ServerManager, "start_with_budget",
                new_callable=AsyncMock, return_value=13568,
            ),
        ):
            _, ctx = await setup_backend(
                backend="llamaserver",
                gguf_path="/models/llama3.gguf",
                on_compact=callback,
            )

        assert ctx.on_compact is callback

    @pytest.mark.asyncio
    async def test_setup_backend_ollama_wires_num_ctx(self) -> None:
        """setup_backend sets client.set_num_ctx(budget) for Ollama."""
        mock_client = MagicMock()
        mock_client.set_num_ctx = MagicMock()
        with patch.object(
            ServerManager, "start_with_budget",
            new_callable=AsyncMock, return_value=4096,
        ):
            _, ctx = await setup_backend(
                backend="ollama",
                model="llama3",
                client=mock_client,
            )
        mock_client.set_num_ctx.assert_called_once_with(4096)
        assert ctx.budget_tokens == 4096

    @pytest.mark.asyncio
    async def test_setup_backend_llamaserver_ignores_client(self) -> None:
        """setup_backend does NOT call set_num_ctx for non-Ollama backends."""
        mock_client = MagicMock()
        mock_client.set_num_ctx = MagicMock()
        with patch.object(
            ServerManager, "start_with_budget",
            new_callable=AsyncMock, return_value=13568,
        ):
            await setup_backend(
                backend="llamaserver",
                client=mock_client,
                gguf_path="/models/llama3.gguf",
            )
        mock_client.set_num_ctx.assert_not_called()

    @pytest.mark.asyncio
    async def test_setup_backend_ollama_no_client_ok(self) -> None:
        """setup_backend without client still works (no crash)."""
        with patch.object(
            ServerManager, "start_with_budget",
            new_callable=AsyncMock, return_value=4096,
        ):
            server, ctx = await setup_backend(
                backend="ollama",
                model="llama3",
            )
        assert ctx.budget_tokens == 4096

    @pytest.mark.asyncio
    async def test_setup_backend_ollama_rejects_gguf_path(self) -> None:
        """Ollama backend must not accept gguf_path."""
        with pytest.raises(ValueError, match="ollama.*does not accept gguf_path"):
            await setup_backend(
                backend="ollama", model="llama3", gguf_path="/models/x.gguf",
            )

    @pytest.mark.asyncio
    async def test_setup_backend_ollama_requires_model(self) -> None:
        """Ollama backend must have a model."""
        with pytest.raises(ValueError, match="ollama.*requires model"):
            await setup_backend(backend="ollama")

    @pytest.mark.asyncio
    async def test_setup_backend_llamaserver_rejects_model(self) -> None:
        """llamaserver/llamafile must not accept model (use gguf_path)."""
        with pytest.raises(ValueError, match="does not accept model"):
            await setup_backend(
                backend="llamaserver", model="llama3", gguf_path="/x.gguf",
            )

    @pytest.mark.asyncio
    async def test_setup_backend_llamaserver_requires_gguf(self) -> None:
        """llamaserver/llamafile must have a gguf_path."""
        with pytest.raises(ValueError, match="requires gguf_path"):
            await setup_backend(backend="llamaserver")


# ── Full workflow wiring (integration-style, mocked) ─────────────


class TestFullWorkflowWiring:
    """Sanity check that all types wire together."""

    @pytest.mark.asyncio
    async def test_full_workflow_wiring(self) -> None:
        """Create mocked ServerManager + OllamaClient + ContextManager + WorkflowRunner."""
        from forge.clients.ollama import OllamaClient
        from forge.context.manager import ContextManager
        from forge.context.strategies import TieredCompact
        from forge.core.runner import WorkflowRunner

        sm = ServerManager(backend="ollama")
        hw = MagicMock(vram_total_gb=12.0)
        with patch("forge.server.detect_hardware", return_value=hw):
            budget = await sm.resolve_budget(BudgetMode.BACKEND)

        client = OllamaClient(model="llama3")
        ctx = ContextManager(
            strategy=TieredCompact(),
            budget_tokens=budget,
        )
        runner = WorkflowRunner(client=client, context_manager=ctx)

        assert runner is not None
        assert ctx.budget_tokens == 4096
