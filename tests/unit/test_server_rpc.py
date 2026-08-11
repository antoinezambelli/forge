"""Model-free lifecycle tests for Forge-managed llama.cpp RPC."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from forge.errors import BudgetResolutionError
from forge.rpc import LlamaCppRpcConfig, LlamaCppRpcWorkerConfig
from forge.server import BudgetMode, ServerManager, setup_backend


@pytest.fixture()
def rpc_config(tmp_path: Path) -> LlamaCppRpcConfig:
    return LlamaCppRpcConfig(
        worker=LlamaCppRpcWorkerConfig(
            ssh_target="antoine@10.35.0.5",
            rpc_host="10.35.0.5",
            executable="/opt/llama/build/bin/ggml-rpc-server",
            device="Vulkan0",
            environment=(
                ("AMD_VULKAN_ICD", "RADV"),
                ("RADV_PERFTEST", "nogttspill"),
            ),
        ),
        coordinator_executable="/opt/llama/build/bin/llama-server",
        coordinator_environment=(
            ("AMD_VULKAN_ICD", "RADV"),
            ("RADV_PERFTEST", "nogttspill"),
        ),
        devices=("Vulkan0", "RPC0"),
        tensor_split=(1, 1),
        startup_timeout=1800,
        log_directory=tmp_path,
    )


def _process() -> MagicMock:
    process = MagicMock()
    process.poll.return_value = None
    return process


@pytest.mark.asyncio
async def test_rpc_starts_worker_then_coordinator(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    manager = ServerManager("llamaserver", port=8080)
    worker = _process()
    coordinator = _process()
    with (
        patch(
            "forge.server.subprocess.Popen",
            side_effect=[worker, coordinator],
        ) as popen,
        patch.object(manager, "_wait_rpc_worker", new_callable=AsyncMock) as wait_rpc,
        patch.object(manager, "_wait_healthy", new_callable=AsyncMock) as wait_server,
    ):
        await manager.start(
            "/models/deepseek.gguf",
            gguf_path="/models/deepseek.gguf",
            extra_flags=["--fit", "off"],
            rpc=rpc_config,
        )

    assert popen.call_count == 2
    assert popen.call_args_list[0].args[0][0] == "ssh"
    coordinator_command = popen.call_args_list[1].args[0]
    assert coordinator_command[:3] == [
        "/opt/llama/build/bin/llama-server",
        "-m",
        "/models/deepseek.gguf",
    ]
    assert coordinator_command[coordinator_command.index("--rpc") + 1] == (
        "10.35.0.5:50052"
    )
    assert coordinator_command[coordinator_command.index("--device") + 1] == (
        "Vulkan0,RPC0"
    )
    assert popen.call_args_list[1].kwargs["env"]["AMD_VULKAN_ICD"] == "RADV"
    wait_rpc.assert_awaited_once_with(rpc_config)
    wait_server.assert_awaited_once_with(timeout=1800)
    assert manager.client_base_url == "http://localhost:8080/v1"
    assert manager.rpc_log_paths == (
        Path(rpc_config.log_directory) / "rpc-worker.log",
        Path(rpc_config.log_directory) / "rpc-coordinator.log",
    )

    with patch("forge.server.asyncio.sleep", new_callable=AsyncMock):
        await manager.stop()


@pytest.mark.asyncio
async def test_rpc_stop_is_coordinator_then_worker(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    manager = ServerManager("llamaserver")
    events: list[str] = []
    worker = _process()
    coordinator = _process()
    worker.terminate.side_effect = lambda: events.append("worker")
    coordinator.terminate.side_effect = lambda: events.append("coordinator")
    with (
        patch("forge.server.subprocess.Popen", side_effect=[worker, coordinator]),
        patch.object(manager, "_wait_rpc_worker", new_callable=AsyncMock),
        patch.object(manager, "_wait_healthy", new_callable=AsyncMock),
    ):
        await manager.start("/m.gguf", gguf_path="/m.gguf", rpc=rpc_config)

    handles = manager._rpc_log_handles
    assert handles is not None
    with patch("forge.server.asyncio.sleep", new_callable=AsyncMock):
        await manager.stop()

    assert events == ["coordinator", "worker"]
    assert all(handle.closed for handle in handles)
    assert manager._rpc_log_handles is None
    assert manager.rpc_log_paths is not None


@pytest.mark.asyncio
async def test_worker_cleanup_still_runs_if_coordinator_stop_errors(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    manager = ServerManager("llamaserver")
    worker = _process()
    coordinator = _process()
    coordinator.terminate.side_effect = OSError("coordinator stop failed")
    with (
        patch("forge.server.subprocess.Popen", side_effect=[worker, coordinator]),
        patch.object(manager, "_wait_rpc_worker", new_callable=AsyncMock),
        patch.object(manager, "_wait_healthy", new_callable=AsyncMock),
    ):
        await manager.start("/m.gguf", gguf_path="/m.gguf", rpc=rpc_config)

    with (
        patch("forge.server.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(OSError, match="coordinator stop failed"),
    ):
        await manager.stop()

    worker.terminate.assert_called_once()
    assert manager._rpc_worker_proc is None


@pytest.mark.asyncio
async def test_worker_tcp_timeout_cleans_worker_and_reports_logs(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    manager = ServerManager("llamaserver")
    short_timeout = replace(rpc_config, startup_timeout=0.1)
    worker = _process()
    clock = [0.0]

    def advance_clock() -> float:
        clock[0] += 0.05
        return clock[0]

    with (
        patch("forge.server.subprocess.Popen", return_value=worker) as popen,
        patch("forge.server.time.monotonic", side_effect=advance_clock),
        patch(
            "forge.server.asyncio.open_connection",
            new_callable=AsyncMock,
            side_effect=OSError("not listening"),
        ),
        patch("forge.server.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(RuntimeError, match="within 0.1s.*logs"),
    ):
        await manager.start("/m.gguf", gguf_path="/m.gguf", rpc=short_timeout)

    assert popen.call_count == 1
    worker.terminate.assert_called_once()
    assert manager._proc is None
    assert manager._rpc_worker_proc is None
    assert manager.rpc_log_paths is not None


@pytest.mark.asyncio
async def test_coordinator_spawn_failure_cleans_worker(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    manager = ServerManager("llamaserver")
    worker = _process()
    with (
        patch(
            "forge.server.subprocess.Popen",
            side_effect=[worker, OSError("coordinator spawn failed")],
        ),
        patch.object(manager, "_wait_rpc_worker", new_callable=AsyncMock),
        patch("forge.server.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(OSError, match="coordinator spawn failed") as exc_info,
    ):
        await manager.start("/m.gguf", gguf_path="/m.gguf", rpc=rpc_config)

    worker.terminate.assert_called_once()
    assert manager._rpc_worker_proc is None
    assert any("RPC logs:" in note for note in exc_info.value.__notes__)


@pytest.mark.asyncio
async def test_coordinator_props_timeout_cleans_both(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    manager = ServerManager("llamaserver")
    short_timeout = replace(rpc_config, startup_timeout=0.1)
    events: list[str] = []
    worker = _process()
    coordinator = _process()
    worker.terminate.side_effect = lambda: events.append("worker")
    coordinator.terminate.side_effect = lambda: events.append("coordinator")
    clock = [0.0]

    def advance_clock() -> float:
        clock[0] += 0.05
        return clock[0]

    with (
        patch("forge.server.subprocess.Popen", side_effect=[worker, coordinator]),
        patch.object(manager, "_wait_rpc_worker", new_callable=AsyncMock),
        patch.object(
            manager,
            "_probe_readiness",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch("forge.server.time.monotonic", side_effect=advance_clock),
        patch("forge.server.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(RuntimeError, match="within 0.1s"),
    ):
        await manager.start("/m.gguf", gguf_path="/m.gguf", rpc=short_timeout)

    assert events == ["coordinator", "worker"]


@pytest.mark.asyncio
async def test_exited_child_after_readiness_rejects_pair(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    for dead_child, message in (
        ("worker", "worker exited"),
        ("coordinator", "Coordinator exited"),
    ):
        manager = ServerManager("llamaserver")
        worker = _process()
        coordinator = _process()
        if dead_child == "worker":
            worker.poll.return_value = 1
        else:
            coordinator.poll.return_value = 1

        with (
            patch("forge.server.subprocess.Popen", side_effect=[worker, coordinator]),
            patch.object(manager, "_wait_rpc_worker", new_callable=AsyncMock),
            patch.object(manager, "_wait_healthy", new_callable=AsyncMock),
            patch("forge.server.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError, match=message),
        ):
            await manager.start("/m.gguf", gguf_path="/m.gguf", rpc=rpc_config)

        coordinator.terminate.assert_called_once()
        worker.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_matching_healthy_rpc_launch_is_reused(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    manager = ServerManager("llamaserver")
    worker = _process()
    coordinator = _process()
    with (
        patch("forge.server.subprocess.Popen", side_effect=[worker, coordinator]) as popen,
        patch.object(manager, "_wait_rpc_worker", new_callable=AsyncMock),
        patch.object(manager, "_wait_healthy", new_callable=AsyncMock),
        patch.object(
            manager, "is_healthy", new_callable=AsyncMock, return_value=True,
        ) as healthy,
    ):
        await manager.start("/m.gguf", gguf_path="/m.gguf", rpc=rpc_config)
        await manager.start("/m.gguf", gguf_path="/m.gguf", rpc=rpc_config)

    assert popen.call_count == 2
    healthy.assert_awaited_once()
    with patch("forge.server.asyncio.sleep", new_callable=AsyncMock):
        await manager.stop()


@pytest.mark.asyncio
async def test_matching_unhealthy_rpc_launch_is_restarted(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    manager = ServerManager("llamaserver")
    first_worker, first_coordinator = _process(), _process()
    second_worker, second_coordinator = _process(), _process()
    with (
        patch(
            "forge.server.subprocess.Popen",
            side_effect=[
                first_worker,
                first_coordinator,
                second_worker,
                second_coordinator,
            ],
        ) as popen,
        patch.object(manager, "_wait_rpc_worker", new_callable=AsyncMock),
        patch.object(manager, "_wait_healthy", new_callable=AsyncMock),
        patch.object(
            manager, "is_healthy", new_callable=AsyncMock, return_value=False,
        ),
        patch("forge.server.asyncio.sleep", new_callable=AsyncMock),
    ):
        await manager.start("/m.gguf", gguf_path="/m.gguf", rpc=rpc_config)
        await manager.start("/m.gguf", gguf_path="/m.gguf", rpc=rpc_config)
        await manager.stop()

    assert popen.call_count == 4
    first_coordinator.terminate.assert_called_once()
    first_worker.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_health_rejects_dead_worker_without_probe(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    manager = ServerManager("llamaserver")
    worker = _process()
    coordinator = _process()
    with (
        patch("forge.server.subprocess.Popen", side_effect=[worker, coordinator]),
        patch.object(manager, "_wait_rpc_worker", new_callable=AsyncMock),
        patch.object(manager, "_wait_healthy", new_callable=AsyncMock),
    ):
        await manager.start("/m.gguf", gguf_path="/m.gguf", rpc=rpc_config)

    worker.poll.return_value = 1
    with patch.object(
        manager, "_probe_readiness", new_callable=AsyncMock,
    ) as probe:
        assert await manager.is_healthy() is False
    probe.assert_not_awaited()
    with patch("forge.server.asyncio.sleep", new_callable=AsyncMock):
        await manager.stop()


@pytest.mark.asyncio
async def test_health_turns_expected_probe_error_into_false(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    manager = ServerManager("llamaserver")
    worker = _process()
    coordinator = _process()
    with (
        patch("forge.server.subprocess.Popen", side_effect=[worker, coordinator]),
        patch.object(manager, "_wait_rpc_worker", new_callable=AsyncMock),
        patch.object(manager, "_wait_healthy", new_callable=AsyncMock),
    ):
        await manager.start("/m.gguf", gguf_path="/m.gguf", rpc=rpc_config)

    request = httpx.Request("GET", "http://localhost:8080/props")
    with patch.object(
        manager,
        "_probe_readiness",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("offline", request=request),
    ):
        assert await manager.is_healthy() is False
    with patch("forge.server.asyncio.sleep", new_callable=AsyncMock):
        await manager.stop()


@pytest.mark.asyncio
async def test_restart_replays_last_resolved_rpc_launch_once(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    manager = ServerManager("llamaserver")
    worker = _process()
    coordinator = _process()
    with (
        patch("forge.server.subprocess.Popen", side_effect=[worker, coordinator]),
        patch.object(manager, "_wait_rpc_worker", new_callable=AsyncMock),
        patch.object(manager, "_wait_healthy", new_callable=AsyncMock),
    ):
        await manager.start(
            "/m.gguf",
            gguf_path="/m.gguf",
            ctx_override=8192,
            extra_flags=["--fit", "off"],
            rpc=rpc_config,
        )
    with patch("forge.server.asyncio.sleep", new_callable=AsyncMock):
        await manager.stop()

    with patch.object(manager, "start", new_callable=AsyncMock) as start:
        await manager.restart()

    start.assert_awaited_once_with(
        "/m.gguf",
        gguf_path="/m.gguf",
        model_path=None,
        mode="native",
        extra_flags=["--fit", "off"],
        ctx_override=8192,
        cache_type_k=None,
        cache_type_v=None,
        n_slots=None,
        kv_unified=False,
        rpc=rpc_config,
    )


@pytest.mark.asyncio
async def test_rpc_forge_fast_preserves_two_phase_sequence(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    manager = ServerManager("llamaserver")
    with (
        patch.object(manager, "start", new_callable=AsyncMock) as start,
        patch.object(
            manager,
            "get_server_context",
            new_callable=AsyncMock,
            side_effect=[16_384, 8_192],
        ),
    ):
        result = await manager.start_with_budget(
            "/m.gguf",
            gguf_path="/m.gguf",
            budget_mode=BudgetMode.FORGE_FAST,
            rpc=rpc_config,
        )

    assert result == 8_192
    assert start.await_count == 2
    assert start.await_args_list[0].kwargs["ctx_override"] is None
    assert start.await_args_list[1].kwargs["ctx_override"] == 8_192
    assert all(call.kwargs["rpc"] is rpc_config for call in start.await_args_list)


@pytest.mark.asyncio
async def test_rpc_forge_fast_budget_failures_stop_pair(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    for context_results, expected_starts in (
        ([BudgetResolutionError()], 1),
        ([16_384, BudgetResolutionError()], 2),
    ):
        manager = ServerManager("llamaserver")
        with (
            patch.object(manager, "start", new_callable=AsyncMock) as start,
            patch.object(
                manager,
                "get_server_context",
                new_callable=AsyncMock,
                side_effect=context_results,
            ),
            patch.object(manager, "stop", new_callable=AsyncMock) as stop,
            pytest.raises(BudgetResolutionError),
        ):
            await manager.start_with_budget(
                "/m.gguf",
                gguf_path="/m.gguf",
                budget_mode=BudgetMode.FORGE_FAST,
                rpc=rpc_config,
            )
        assert start.await_count == expected_starts
        stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_setup_backend_threads_rpc_to_manager(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    with patch.object(
        ServerManager,
        "start_with_budget",
        new_callable=AsyncMock,
        return_value=262_144,
    ) as start:
        server, context = await setup_backend(
            backend="llamaserver",
            gguf_path="/m.gguf",
            rpc=rpc_config,
        )

    assert isinstance(server, ServerManager)
    assert context.budget_tokens == 262_144
    assert start.await_args.kwargs["rpc"] is rpc_config


@pytest.mark.asyncio
async def test_rpc_rejects_non_llamaserver_before_launch(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    with (
        patch("forge.server.subprocess.Popen") as popen,
        pytest.raises(ValueError, match="only by backend='llamaserver'"),
    ):
        await setup_backend(
            backend="llamafile",
            gguf_path="/m.gguf",
            rpc=rpc_config,
        )
    popen.assert_not_called()


@pytest.mark.asyncio
async def test_rpc_rejects_duplicate_topology_before_launch(
    rpc_config: LlamaCppRpcConfig,
) -> None:
    with (
        patch("forge.server.subprocess.Popen") as popen,
        pytest.raises(ValueError, match="RPC topology owns --rpc"),
    ):
        await setup_backend(
            backend="llamaserver",
            gguf_path="/m.gguf",
            rpc=rpc_config,
            extra_flags=["--rpc=10.35.0.5:50052"],
        )
    popen.assert_not_called()
