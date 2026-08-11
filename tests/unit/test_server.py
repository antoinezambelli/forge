"""Tests for forge.server — BudgetMode enum and ServerManager."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from forge._backend_profiles import managed_profile
from forge._endpoint_layouts import ConnectionInputKind
from forge._resolved_backend import resolve_backend
from forge.context.manager import ContextManager
from forge.context.strategies import TieredCompact
from forge.errors import BudgetResolutionError
from forge.server import (
    BudgetMode,
    ServerManager,
    _setup_managed_backend,
    setup_backend,
)


# ── BudgetMode ──────────────────────────────────────────────────


class TestBudgetMode:
    """BudgetMode enum basics."""

    def test_values_and_string_behavior(self) -> None:
        assert BudgetMode.BACKEND.value == "backend"
        assert BudgetMode.MANUAL.value == "manual"
        assert BudgetMode.FORGE_FULL.value == "forge-full"
        assert BudgetMode.FORGE_FAST.value == "forge-fast"
        assert BudgetMode.BACKEND == "backend"
        assert BudgetMode.MANUAL == "manual"
        assert BudgetMode.FORGE_FULL == "forge-full"
        assert BudgetMode.FORGE_FAST == "forge-fast"


# ── ServerManager construction ──────────────────────────────────


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
            await sm.start("llama3", gguf_path="/models/llama3.gguf")

        args = mock_popen.call_args[0][0]
        assert "llama-server" in args
        assert "-m" in args
        assert "/models/llama3.gguf" in args
        assert "-ngl" in args
        assert "999" in args
        assert "--port" in args
        assert "8080" in args

    @pytest.mark.asyncio
    async def test_start_readiness_failure_cleans_spawned_process(
        self, sm: ServerManager,
    ) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc),
            patch.object(
                sm,
                "_wait_healthy",
                new_callable=AsyncMock,
                side_effect=RuntimeError("not ready"),
            ),
            patch("forge.server.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError, match="not ready"),
        ):
            await sm.start("llama3", gguf_path="/models/llama3.gguf")

        mock_proc.terminate.assert_called_once()
        assert sm._proc is None

    @pytest.mark.asyncio
    async def test_start_native_mode_adds_jinja(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start("llama3", gguf_path="/models/llama3.gguf", mode="native")

        args = mock_popen.call_args[0][0]
        assert args.count("--jinja") == 1

    @pytest.mark.asyncio
    async def test_start_prompt_mode_no_jinja(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start("llama3", gguf_path="/models/llama3.gguf", mode="prompt")

        args = mock_popen.call_args[0][0]
        assert args.count("--jinja") == 0

    @pytest.mark.asyncio
    async def test_start_with_extra_flags(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start(
                "qwen3", gguf_path="/models/qwen3.gguf",
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
            await sm.start("llama3", gguf_path="/models/llama3.gguf", ctx_override=8000)

        args = mock_popen.call_args[0][0]
        assert "-c" in args
        assert "8000" in args

    @pytest.mark.asyncio
    async def test_start_reuses_same_config(self) -> None:
        cases = [
            ("base", {}),
            ("extra flags", {"extra_flags": ["--reasoning-format", "auto"]}),
        ]
        for label, options in cases:
            sm = ServerManager(backend="llamaserver", port=8080)
            mock_proc = MagicMock()
            with (
                patch(
                    "forge.server.subprocess.Popen", return_value=mock_proc
                ) as mock_popen,
                patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
                patch.object(
                    sm, "is_healthy", new_callable=AsyncMock, return_value=True,
                ),
            ):
                await sm.start(
                    "llama3", gguf_path="/models/llama3.gguf", mode="native",
                    **options,
                )
                await sm.start(
                    "llama3", gguf_path="/models/llama3.gguf", mode="native",
                    **options,
                )
            assert mock_popen.call_count == 1, label

    @pytest.mark.asyncio
    async def test_start_restarts_when_launch_configuration_changes(self) -> None:
        cases = [
            ("mode", {"mode": "native"}, {"mode": "prompt"}),
            ("context", {}, {"ctx_override": 8000}),
            (
                "extra flags",
                {"mode": "native"},
                {"mode": "native", "extra_flags": ["--reasoning-format", "auto"]},
            ),
        ]
        for label, initial_options, changed_options in cases:
            sm = ServerManager(backend="llamaserver", port=8080)
            mock_proc = MagicMock()
            with (
                patch(
                    "forge.server.subprocess.Popen", return_value=mock_proc
                ) as mock_popen,
                patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
                patch.object(sm, "stop", new_callable=AsyncMock),
            ):
                await sm.start(
                    "llama3", gguf_path="/models/llama3.gguf", **initial_options
                )
                await sm.start(
                    "llama3", gguf_path="/models/llama3.gguf", **changed_options
                )
            assert mock_popen.call_count == 2, label

    @pytest.mark.asyncio
    async def test_start_noop_for_ollama(self) -> None:
        sm = ServerManager(backend="ollama")
        with patch("forge.server.subprocess.Popen") as mock_popen:
            await sm.start(
                "llama3",
                gguf_path="/models/llama3.gguf",
                mode="prompt",
            )

        mock_popen.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"extra_flags": ["--flag"]}, "extra_flags"),
            ({"cache_type_k": "q8_0"}, "cache_type_k"),
            ({"cache_type_v": "q8_0"}, "cache_type_k"),
            ({"n_slots": 2}, "n_slots"),
            ({"kv_unified": True}, "kv_unified"),
        ],
    )
    async def test_start_ollama_rejects_ignored_controls(
        self, kwargs: dict[str, object], message: str,
    ) -> None:
        sm = ServerManager(backend="ollama")
        with pytest.raises(ValueError, match=message):
            await sm.start("llama3", **kwargs)

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
            await sm.start("llama3", gguf_path=str(model_path), mode="prompt")

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
            await sm.start("llama3", gguf_path=str(model_path), mode="prompt")

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
            await sm.start("llama3", gguf_path=str(model_path), mode="prompt")

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
                "llama3", gguf_path="/models/llama3.gguf",
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
            await sm.start("llama3", gguf_path="/models/llama3.gguf")

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
            await sm.start("llama3", gguf_path="/models/llama3.gguf", cache_type_k="q8_0")
            assert sm._current_cache_type_k == "q8_0"

            # Same config — should reuse (no restart)
            await sm.start("llama3", gguf_path="/models/llama3.gguf", cache_type_k="q8_0")

            # Different cache type — should restart
            await sm.start("llama3", gguf_path="/models/llama3.gguf", cache_type_k="q4_0")
            assert sm._current_cache_type_k == "q4_0"

    @pytest.mark.asyncio
    async def test_start_with_n_slots(self) -> None:
        sm = ServerManager(backend="llamaserver", port=8080)
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start("llama3", gguf_path="/models/llama3.gguf", n_slots=2)

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
            await sm.start("llama3", gguf_path="/models/llama3.gguf")

        cmd = mock_popen.call_args[0][0]
        assert "--parallel" not in cmd


# ── ServerManager.start() — vllm backend ────────────────────────


class TestServerManagerStartVllm:
    """start() vLLM-specific path: cmd build, validation, /v1/models discovery."""

    @pytest.fixture()
    def sm(self) -> ServerManager:
        return ServerManager(backend="vllm", port=8000)

    @pytest.mark.asyncio
    async def test_start_launches_vllm_serve(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start(
                "gemma-4-AWQ",
                model_path="/models/gemma-4-26B-A4B-it-AWQ-4bit",
            )

        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "vllm"
        assert cmd[1] == "serve"
        assert "/models/gemma-4-26B-A4B-it-AWQ-4bit" in cmd
        assert "--port" in cmd
        assert "8000" in cmd

    @pytest.mark.asyncio
    async def test_start_does_not_pass_llamacpp_flags(self, sm: ServerManager) -> None:
        """vLLM should never receive llama.cpp-specific flags like --jinja."""
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start(
                "gemma-4-AWQ",
                model_path="/models/gemma",
                mode="native",
            )

        cmd = mock_popen.call_args[0][0]
        assert "--jinja" not in cmd
        assert "-c" not in cmd  # llama.cpp ctx flag
        assert "-m" not in cmd  # llama.cpp model flag

    @pytest.mark.asyncio
    async def test_ctx_override_passes_max_model_len(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start(
                "gemma-4-AWQ",
                model_path="/models/gemma",
                ctx_override=113000,
            )

        cmd = mock_popen.call_args[0][0]
        assert "--max-model-len" in cmd
        idx = cmd.index("--max-model-len")
        assert cmd[idx + 1] == "113000"

    @pytest.mark.asyncio
    async def test_extra_flags_appended(self, sm: ServerManager) -> None:
        mock_proc = MagicMock()
        with (
            patch("forge.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(sm, "_wait_healthy", new_callable=AsyncMock),
        ):
            await sm.start(
                "gemma-4-AWQ",
                model_path="/models/gemma",
                extra_flags=[
                    "--tensor-parallel-size", "2",
                    "--reasoning-parser", "gemma4",
                    "--tool-call-parser", "gemma4",
                    "--enable-auto-tool-choice",
                ],
            )

        cmd = mock_popen.call_args[0][0]
        assert "--tensor-parallel-size" in cmd
        assert "--reasoning-parser" in cmd
        assert "gemma4" in cmd

    @pytest.mark.asyncio
    async def test_rejects_gguf_path(self, sm: ServerManager) -> None:
        with pytest.raises(ValueError, match="does not accept gguf_path"):
            await sm.start("x", gguf_path="/models/x.gguf")

    @pytest.mark.asyncio
    async def test_requires_model_path(self, sm: ServerManager) -> None:
        with pytest.raises(ValueError, match="requires model_path"):
            await sm.start("x")

    @pytest.mark.asyncio
    async def test_rejects_llamacpp_runtime_controls(self) -> None:
        cases = [
            ("cache type k", {"cache_type_k": "q8_0"}, "cache_type"),
            ("cache type v", {"cache_type_v": "q8_0"}, "cache_type"),
            ("slots", {"n_slots": 2}, "n_slots"),
            ("unified kv", {"kv_unified": True}, "n_slots"),
        ]
        for label, options, expected in cases:
            sm = ServerManager(backend="vllm", port=8000)
            with pytest.raises(ValueError) as exc_info:
                await sm.start("x", model_path="/m", **options)
            assert expected in str(exc_info.value), label

    @pytest.mark.asyncio
    async def test_llamaserver_rejects_model_path(self) -> None:
        """Symmetry: llamaserver should reject model_path (it's a vllm-only param)."""
        sm = ServerManager(backend="llamaserver")
        with pytest.raises(ValueError, match="does not accept model_path"):
            await sm.start("x", model_path="/models/x")

    @pytest.mark.asyncio
    async def test_unknown_backend_raises(self) -> None:
        sm = ServerManager(backend="bogus")
        with pytest.raises(ValueError, match="unsupported backend"):
            await sm.start("x", gguf_path="/models/x.gguf")


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

        proc = AsyncMock()
        with patch(
            "forge.server.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ) as mock_exec:
            await sm.stop()

        mock_exec.assert_called_once_with("ollama", "stop", "ministral:14b")
        proc.wait.assert_awaited_once()

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


# ── ServerManager.get_server_context() — vllm backend ───────────


class TestGetServerContextVllm:
    """get_server_context() vllm path: parses max_model_len from /v1/models."""

    @pytest.mark.asyncio
    async def test_reads_max_model_len_from_models_endpoint(self) -> None:
        sm = ServerManager(backend="vllm", port=8000)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"id": "/models/x", "max_model_len": 113000}],
        }
        with patch("forge.server.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = await sm.get_server_context()

        assert result == 113000
        mock_client.get.assert_awaited_once_with("http://localhost:8000/v1/models")


class TestServerOperationAddresses:
    @pytest.mark.asyncio
    async def test_query_props_uses_shared_llama_address(self) -> None:
        sm = ServerManager(backend="llamaserver", port=9090)
        response = MagicMock(status_code=200)
        response.json.return_value = {"default_generation_settings": {"n_ctx": 1}}
        with patch("forge.server.httpx.AsyncClient") as client_cls:
            client = AsyncMock()
            client.__aenter__.return_value = client
            client.get.return_value = response
            client_cls.return_value = client
            await sm.query_props()
        client.get.assert_awaited_once_with("http://localhost:9090/props")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("backend", "port", "body", "expected"),
        [
            (
                "llamaserver", 9090,
                {"default_generation_settings": {}},
                "http://localhost:9090/props",
            ),
            ("vllm", 8000, {"data": [{"id": "m"}]}, "http://localhost:8000/v1/models"),
        ],
    )
    async def test_readiness_uses_shared_startup_probe_address(
        self, backend: str, port: int, body: dict, expected: str,
    ) -> None:
        sm = ServerManager(backend=backend, port=port)
        response = MagicMock(status_code=200)
        response.json.return_value = body
        with patch("forge.server.httpx.AsyncClient") as client_cls:
            client = AsyncMock()
            client.__aenter__.return_value = client
            client.get.return_value = response
            client_cls.return_value = client
            await sm._wait_healthy(timeout=0.1)
        client.get.assert_awaited_once_with(expected)

    @pytest.mark.asyncio
    async def test_raises_on_empty_data(self) -> None:
        sm = ServerManager(backend="vllm", port=8000)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        with patch("forge.server.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            with pytest.raises(BudgetResolutionError):
                await sm.get_server_context()

    @pytest.mark.asyncio
    async def test_raises_on_missing_max_model_len(self) -> None:
        sm = ServerManager(backend="vllm", port=8000)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "/models/x"}]}
        with patch("forge.server.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            with pytest.raises(BudgetResolutionError):
                await sm.get_server_context()

    @pytest.mark.asyncio
    async def test_raises_on_non_200(self) -> None:
        sm = ServerManager(backend="vllm", port=8000)
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "service unavailable"
        with patch("forge.server.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            with pytest.raises(BudgetResolutionError):
                await sm.get_server_context()


# ── ServerManager.resolve_budget() ──────────────────────────────


class TestResolveBudget:
    """resolve_budget() mode × backend matrix."""

    # -- backend mode --

    @pytest.mark.asyncio
    async def test_resolve_budget_backend_ollama_vram_tiers(self) -> None:
        cases = [
            ("no gpu", None, 4096),
            ("12 GB", 12.0, 4096),
            ("24 GB", 24.0, 32768),
            ("48 GB", 48.0, 262144),
        ]
        for label, vram_gb, expected in cases:
            sm = ServerManager(backend="ollama")
            hardware = None if vram_gb is None else MagicMock(vram_total_gb=vram_gb)
            with patch("forge.server.detect_hardware", return_value=hardware):
                result = await sm.resolve_budget(BudgetMode.BACKEND)
            assert result == expected, label

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
    async def test_resolve_budget_manual_requires_positive_tokens(self) -> None:
        for manual_tokens in [None, 0, -1]:
            sm = ServerManager(backend="ollama")
            with pytest.raises(ValueError) as exc_info:
                await sm.resolve_budget(
                    BudgetMode.MANUAL, manual_tokens=manual_tokens
                )
            assert "manual" in str(exc_info.value), f"manual_tokens={manual_tokens}"

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
    async def test_resolve_budget_propagates_missing_server_context(self) -> None:
        cases = [
            ("backend", BudgetMode.BACKEND, None),
            ("manual", BudgetMode.MANUAL, 8000),
        ]
        for label, mode, manual_tokens in cases:
            sm = ServerManager(backend="llamaserver")
            with (
                patch.object(
                    sm,
                    "get_server_context",
                    new_callable=AsyncMock,
                    side_effect=BudgetResolutionError(),
                ),
                pytest.raises(BudgetResolutionError),
            ):
                await sm.resolve_budget(mode, manual_tokens=manual_tokens)


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
                "llama3", gguf_path="/models/llama3.gguf",
                budget_mode=BudgetMode.BACKEND,
            )

        mock_start.assert_called_once_with(
            "llama3", gguf_path="/models/llama3.gguf", model_path=None,
            mode="native", extra_flags=None, ctx_override=None,
            cache_type_k=None, cache_type_v=None, n_slots=None, kv_unified=False,
        )
        assert result == 13568

    @pytest.mark.asyncio
    async def test_ollama_restart_replays_last_attached_model(self) -> None:
        sm = ServerManager(backend="ollama")
        with patch.object(
            sm, "resolve_budget", new_callable=AsyncMock, return_value=4096,
        ):
            await sm.start_with_budget("llama3", budget_mode=BudgetMode.BACKEND)

        with patch.object(sm, "stop", new_callable=AsyncMock) as stop:
            await sm.restart()

        stop.assert_awaited_once()
        assert sm._current_model == "llama3"

    @pytest.mark.asyncio
    async def test_start_with_budget_manual(self) -> None:
        sm = ServerManager(backend="llamaserver")
        with (
            patch.object(sm, "start", new_callable=AsyncMock) as mock_start,
            patch.object(sm, "get_server_context", new_callable=AsyncMock, return_value=8000),
        ):
            result = await sm.start_with_budget(
                "llama3", gguf_path="/models/llama3.gguf",
                budget_mode=BudgetMode.MANUAL,
                manual_tokens=8000,
            )

        mock_start.assert_called_once_with(
            "llama3", gguf_path="/models/llama3.gguf", model_path=None,
            mode="native", extra_flags=None, ctx_override=8000,
            cache_type_k=None, cache_type_v=None, n_slots=None, kv_unified=False,
        )
        assert result == 8000

    @pytest.mark.asyncio
    async def test_start_with_budget_manual_requires_positive_tokens(self) -> None:
        for manual_tokens in [None, 0, -1]:
            sm = ServerManager(backend="llamaserver")
            with pytest.raises(ValueError) as exc_info:
                await sm.start_with_budget(
                    "llama3",
                    gguf_path="/models/llama3.gguf",
                    budget_mode=BudgetMode.MANUAL,
                    manual_tokens=manual_tokens,
                )
            assert "manual" in str(exc_info.value), f"manual_tokens={manual_tokens}"

    @pytest.mark.asyncio
    async def test_nonmanual_dormant_manual_tokens_are_ignored(self) -> None:
        sm = ServerManager(backend="ollama")
        with patch("forge.server.detect_hardware", return_value=None):
            result = await sm.start_with_budget(
                "llama3",
                budget_mode=BudgetMode.BACKEND,
                manual_tokens=-1,
            )
        assert result == 4096

    @pytest.mark.asyncio
    async def test_start_with_budget_ollama_forwards_option_validation(self) -> None:
        sm = ServerManager(backend="ollama")
        with pytest.raises(ValueError, match="backend='ollama' does not support"):
            await sm.start_with_budget("llama3", extra_flags=["--flag"])

    @pytest.mark.asyncio
    async def test_start_with_budget_forge_full(self) -> None:
        sm = ServerManager(backend="llamaserver")
        with (
            patch.object(sm, "start", new_callable=AsyncMock) as mock_start,
            patch.object(sm, "get_server_context", new_callable=AsyncMock, return_value=13568),
        ):
            result = await sm.start_with_budget(
                "llama3", gguf_path="/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FULL,
            )

        mock_start.assert_called_once_with(
            "llama3", gguf_path="/models/llama3.gguf", model_path=None,
            mode="native", extra_flags=None, ctx_override=None,
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
                "llama3", gguf_path="/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FAST,
            )

        assert mock_start.call_count == 2
        # Phase 1: start without -c
        mock_start.assert_any_call(
            "llama3", gguf_path="/models/llama3.gguf", model_path=None,
            mode="native", extra_flags=None, ctx_override=None,
            cache_type_k=None, cache_type_v=None, n_slots=None, kv_unified=False,
        )
        # Phase 2: restart with half (13568 // 2 = 6784)
        mock_start.assert_any_call(
            "llama3", gguf_path="/models/llama3.gguf", model_path=None,
            mode="native", extra_flags=None, ctx_override=6784,
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
                    "llama3", gguf_path="/models/llama3.gguf",
                    budget_mode=BudgetMode.FORGE_FAST,
                )

    @pytest.mark.asyncio
    async def test_start_with_budget_ollama_backend(self) -> None:
        sm = ServerManager(backend="ollama")
        hw = MagicMock(vram_total_gb=12.0)
        with patch("forge.server.detect_hardware", return_value=hw):
            result = await sm.start_with_budget(
                "llama3", gguf_path="/models/llama3.gguf",
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
                "llama3", gguf_path="/models/llama3.gguf",
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
                "llama3", gguf_path="/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FAST,
                n_slots=2,
            )

        assert mock_start.call_count == 2
        # Phase 2: -c should be 35000 (half of 70K total), NOT 17500 (half of per-slot)
        mock_start.assert_any_call(
            "llama3", gguf_path="/models/llama3.gguf", model_path=None,
            mode="native", extra_flags=None, ctx_override=35000,
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
                "llama3", gguf_path="/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FAST,
                n_slots=1,
            )

        mock_start.assert_any_call(
            "llama3", gguf_path="/models/llama3.gguf", model_path=None,
            mode="native", extra_flags=None, ctx_override=6784,
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
                "llama3", gguf_path="/models/llama3.gguf",
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
                "llama3", gguf_path="/models/llama3.gguf",
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
                "llama3", gguf_path="/models/llama3.gguf",
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
                "llama3", gguf_path="/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FULL,
                n_slots=2,
                kv_unified=True,
            )

        mock_start.assert_called_once_with(
            "llama3", gguf_path="/models/llama3.gguf", model_path=None,
            mode="native", extra_flags=None, ctx_override=None,
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
                "llama3", gguf_path="/models/llama3.gguf",
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
                "llama3", gguf_path="/models/llama3.gguf",
                budget_mode=BudgetMode.FORGE_FAST,
                n_slots=1,
                kv_unified=True,
            )

        mock_start.assert_any_call(
            "llama3", gguf_path="/models/llama3.gguf", model_path=None,
            mode="native", extra_flags=None, ctx_override=35000,
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
        assert isinstance(ctx.strategy, TieredCompact)

    @pytest.mark.asyncio
    async def test_setup_failure_stops_unreturned_manager(self) -> None:
        with (
            patch.object(
                ServerManager,
                "start_with_budget",
                new_callable=AsyncMock,
                side_effect=BudgetResolutionError(),
            ),
            patch.object(ServerManager, "stop", new_callable=AsyncMock) as stop,
            pytest.raises(BudgetResolutionError),
        ):
            await setup_backend(
                backend="llamaserver",
                gguf_path="/models/llama3.gguf",
            )

        stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_setup_backend_passes_compact_threshold_and_callback(self) -> None:
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
                compact_threshold=0.5,
                on_compact=callback,
            )

        # compact_threshold now lives on the strategy, not ContextManager
        assert ctx.strategy._phase_triggers == (0.5, 0.5, 0.5)
        assert ctx.on_compact is callback

    @pytest.mark.asyncio
    async def test_setup_backend_preserves_phase_and_context_callbacks(self) -> None:
        threshold_callback = MagicMock(return_value=None)
        with patch.object(
            ServerManager,
            "start_with_budget",
            new_callable=AsyncMock,
            return_value=13568,
        ):
            _, ctx = await setup_backend(
                backend="llamaserver",
                gguf_path="/models/llama3.gguf",
                phase_thresholds=(0.5, 0.7, 0.9),
                context_thresholds=[0.4, 0.8],
                on_context_threshold=threshold_callback,
            )

        assert ctx.strategy._phase_triggers == (0.5, 0.7, 0.9)
        assert ctx._context_thresholds == [0.4, 0.8]
        assert ctx._on_context_threshold is threshold_callback

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
    async def test_setup_backend_manual_requires_positive_tokens(self) -> None:
        for manual_tokens in [None, 0, -1]:
            with pytest.raises(ValueError) as exc_info:
                await setup_backend(
                    backend="ollama",
                    model="llama3",
                    budget_mode=BudgetMode.MANUAL,
                    manual_tokens=manual_tokens,
                )
            assert "manual_tokens > 0" in str(
                exc_info.value
            ), f"manual_tokens={manual_tokens}"

    @pytest.mark.asyncio
    async def test_setup_backend_ollama_forwards_option_validation(self) -> None:
        with pytest.raises(ValueError, match="backend='ollama' does not support"):
            await setup_backend(backend="ollama", model="llama3", n_slots=2)

    @pytest.mark.asyncio
    async def test_setup_backend_ollama_preserves_mode_noop(self) -> None:
        with patch.object(
            ServerManager,
            "start_with_budget",
            new_callable=AsyncMock,
            return_value=4096,
        ) as start:
            await setup_backend(backend="ollama", model="llama3", mode="prompt")
        assert start.await_args.kwargs["mode"] == "prompt"

    @pytest.mark.asyncio
    async def test_setup_backend_enforces_backend_identity_rules(self) -> None:
        cases = [
            (
                "ollama rejects gguf",
                {"backend": "ollama", "model": "llama3", "gguf_path": "/x"},
                "does not accept gguf_path",
            ),
            ("ollama requires model", {"backend": "ollama"}, "requires model"),
            (
                "ollama rejects model path",
                {"backend": "ollama", "model": "tag", "model_path": "/x"},
                "does not accept model_path",
            ),
            (
                "llamaserver rejects model",
                {"backend": "llamaserver", "model": "tag", "gguf_path": "/x"},
                "does not accept model",
            ),
            (
                "llamaserver requires gguf",
                {"backend": "llamaserver"},
                "requires gguf_path",
            ),
            (
                "llamaserver rejects model path",
                {
                    "backend": "llamaserver",
                    "gguf_path": "/x",
                    "model_path": "/y",
                },
                "does not accept model_path",
            ),
            ("vllm requires model path", {"backend": "vllm"}, "requires model_path"),
            (
                "vllm rejects gguf",
                {"backend": "vllm", "model_path": "/m", "gguf_path": "/x"},
                "does not accept gguf_path",
            ),
            (
                "vllm rejects model",
                {"backend": "vllm", "model_path": "/m", "model": "tag"},
                "does not accept model",
            ),
            ("unknown backend", {"backend": "bogus"}, "unsupported backend"),
        ]
        for label, kwargs, expected in cases:
            with pytest.raises(ValueError) as exc_info:
                await setup_backend(**kwargs)
            assert expected in str(exc_info.value), label

    @pytest.mark.asyncio
    async def test_vllm_setup_returns_manager_and_ctx(self) -> None:
        with (
            patch.object(
                ServerManager, "start_with_budget",
                new_callable=AsyncMock, return_value=113000,
            ),
        ):
            server, ctx = await setup_backend(
                backend="vllm",
                model_path="/models/gemma-4-AWQ",
            )

        assert isinstance(server, ServerManager)
        assert isinstance(ctx, ContextManager)
        assert ctx.budget_tokens == 113000


class TestPrivateManagedSetup:
    @pytest.mark.asyncio
    async def test_backend_mode_can_report_an_unavailable_non_ollama_window(
        self,
    ) -> None:
        with patch.object(
            ServerManager,
            "start_with_budget",
            new_callable=AsyncMock,
            side_effect=BudgetResolutionError(),
        ):
            result = await _setup_managed_backend(
                backend="llamaserver",
                gguf_path="/models/x.gguf",
                budget_mode=BudgetMode.BACKEND,
                allow_missing_backend_window=True,
            )
        assert isinstance(result.server, ServerManager)
        assert result.context_window_tokens is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "budget_mode",
        [BudgetMode.MANUAL, BudgetMode.FORGE_FULL, BudgetMode.FORGE_FAST],
    )
    async def test_explicit_policy_window_failure_remains_loud(
        self, budget_mode: BudgetMode,
    ) -> None:
        with patch.object(
            ServerManager,
            "start_with_budget",
            new_callable=AsyncMock,
            side_effect=BudgetResolutionError(),
        ), pytest.raises(BudgetResolutionError):
            await _setup_managed_backend(
                backend="llamaserver",
                gguf_path="/models/x.gguf",
                budget_mode=budget_mode,
                manual_tokens=8192 if budget_mode == BudgetMode.MANUAL else None,
                allow_missing_backend_window=True,
            )

    @pytest.mark.asyncio
    async def test_native_facade_never_suppresses_backend_window_failure(self) -> None:
        with patch.object(
            ServerManager,
            "start_with_budget",
            new_callable=AsyncMock,
            side_effect=BudgetResolutionError(),
        ), pytest.raises(BudgetResolutionError):
            await setup_backend(
                backend="llamaserver",
                gguf_path="/models/x.gguf",
                budget_mode=BudgetMode.BACKEND,
            )

    @pytest.mark.asyncio
    async def test_ollama_backend_window_failure_is_never_suppressed(self) -> None:
        with patch.object(
            ServerManager,
            "start_with_budget",
            new_callable=AsyncMock,
            side_effect=BudgetResolutionError(),
        ), pytest.raises(BudgetResolutionError):
            await _setup_managed_backend(
                backend="ollama",
                model="llama3",
                budget_mode=BudgetMode.BACKEND,
                allow_missing_backend_window=True,
            )


# ── Full workflow wiring (integration-style, mocked) ─────────────


class TestOllamaDaemonTarget:
    @pytest.mark.asyncio
    async def test_proxy_override_is_used_for_target_aware_stop(self) -> None:
        manager = ServerManager(backend="ollama")
        target = resolve_backend(
            managed_profile("ollama"),
            "http://localhost:22445",
            ConnectionInputKind.OLLAMA_DAEMON_ROOT,
        )
        manager._set_resolved_daemon_target(target)
        manager._current_model = "tag"
        process = MagicMock()
        process.wait = AsyncMock()
        with patch(
            "forge.server.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=process,
        ) as create:
            await manager.stop()
        assert create.await_args.args == ("ollama", "stop", "tag")
        assert create.await_args.kwargs["env"]["OLLAMA_HOST"] == (
            "http://localhost:22445"
        )

    @pytest.mark.asyncio
    async def test_native_setup_ignores_common_port_for_ollama(self) -> None:
        with patch.object(
            ServerManager,
            "start_with_budget",
            new_callable=AsyncMock,
            return_value=4096,
        ):
            server, _ = await setup_backend(
                backend="ollama", model="tag", port=22445,
            )
        assert server._resolved_backend is not None
        assert server._resolved_backend.connection.mount_root == (
            "http://localhost:11434"
        )
        assert server._daemon_target_overridden is False
