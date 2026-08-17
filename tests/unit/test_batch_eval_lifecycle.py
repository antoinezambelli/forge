"""Managed-backend lifecycle coverage for the batch eval harness."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import tests.eval.batch_eval as batch_eval
from forge.clients.sampling_defaults import get_sampling_defaults
from forge.server import BudgetMode
from tests.eval.batch_eval import BatchConfig, _BatchServerRecipe, run_batch
from tests.eval.eval_runner import RunResult
from tests.eval.scenarios import basic_2step


def _result(
    *,
    completed: bool = True,
    error_type: str | None = None,
    error_message: str | None = None,
) -> RunResult:
    return RunResult(
        scenario_name=basic_2step.name,
        completed=completed,
        iterations_used=3 if completed else 0,
        correct=True if completed else None,
        error_type=error_type,
        error_message=error_message,
    )


class _Server:
    def __init__(self, base_url: str, budget: int) -> None:
        self.client_base_url = base_url
        self.budget = budget
        self.stop_count = 0
        self.restart_count = 0

    async def stop(self) -> None:
        self.stop_count += 1

    async def restart(self) -> None:
        self.restart_count += 1

    async def resolve_budget(self, *args: Any) -> int:
        return self.budget


def _prepare_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch_eval, "ALL_SCENARIOS", [basic_2step])
    monkeypatch.setattr(batch_eval, "_check_model_available", lambda *args: None)


@pytest.mark.asyncio
async def test_each_config_uses_one_public_setup_and_its_own_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_batch(monkeypatch)
    rpc_marker = object()
    recipe = _BatchServerRecipe(("--no-mmap",), rpc_marker)  # type: ignore[arg-type]
    configs = [
        BatchConfig(
            model="M", backend="llamaserver", mode="native", think=None,
            gguf_filename="M.gguf", server_recipe=recipe,
            reasoning_level=reasoning_level,
        )
        for reasoning_level in ("default", "high")
    ]
    servers = [
        _Server("http://coordinator-1:8080/v1", 4096),
        _Server("http://coordinator-2:8080/v1", 8192),
    ]
    setup_calls: list[dict[str, Any]] = []

    async def setup(**kwargs: Any) -> tuple[_Server, SimpleNamespace]:
        setup_calls.append(kwargs)
        server = servers[len(setup_calls) - 1]
        return server, SimpleNamespace(budget_tokens=server.budget)

    clients: list[object] = []
    build_calls: list[tuple[str, float]] = []

    class Client:
        async def aclose(self) -> None:
            pytest.fail("batch_eval must preserve its current no-aclose behavior")

    def build_client(
        config: BatchConfig, models_dir: Path, base_url: str, timeout: float,
    ) -> object:
        build_calls.append((base_url, timeout))
        client = Client()
        clients.append(client)
        return client

    run_calls: list[tuple[object, int | None, float]] = []

    async def run_once(
        client: object,
        scenario: Any,
        eval_config: Any,
        ablation: Any,
        run_timeout: float,
    ) -> RunResult:
        run_calls.append((client, eval_config.budget_override, run_timeout))
        return _result()

    monkeypatch.setattr(batch_eval, "setup_backend", setup)
    monkeypatch.setattr(batch_eval, "_build_client", build_client)
    monkeypatch.setattr(batch_eval, "_run_with_timeout", run_once)

    await run_batch(
        configs=configs,
        runs_per_scenario=2,
        output_path=tmp_path / "results.jsonl",
        models_dir=tmp_path,
        budget_mode=BudgetMode.MANUAL,
        manual_tokens=262_144,
        request_timeout=7200,
        run_timeout=7300,
    )

    assert len(setup_calls) == 2
    assert [server.stop_count for server in servers] == [1, 1]
    assert setup_calls[0] == setup_calls[1]
    assert setup_calls[0] == {
        "backend": "llamaserver",
        "model": None,
        "gguf_path": str(tmp_path / "M.gguf"),
        "mode": "native",
        "port": batch_eval._eval_port(),
        "budget_mode": BudgetMode.MANUAL,
        "manual_tokens": 262_144,
        "extra_flags": ["--no-mmap"],
        "rpc": rpc_marker,
    }
    assert build_calls == [
        ("http://coordinator-1:8080/v1", 7200),
        ("http://coordinator-2:8080/v1", 7200),
    ]
    assert run_calls == [
        (clients[0], 4096, 7300),
        (clients[0], 4096, 7300),
        (clients[1], 8192, 7300),
        (clients[1], 8192, 7300),
    ]


@pytest.mark.asyncio
async def test_prelaunch_skips_and_dry_run_never_own_a_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(batch_eval, "ALL_SCENARIOS", [basic_2step])
    availability_calls: list[str] = []

    def availability(config: BatchConfig, models_dir: Path) -> str | None:
        availability_calls.append(config.model)
        return "file not found" if config.model == "missing" else None

    def fail(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("backend/client construction was reached")

    monkeypatch.setattr(batch_eval, "_check_model_available", availability)
    monkeypatch.setattr(batch_eval, "setup_backend", fail)
    monkeypatch.setattr(batch_eval, "_build_client", fail)
    monkeypatch.setattr(batch_eval, "_append_jsonl_row", fail)
    configs = [
        BatchConfig(model="missing", backend="ollama", mode="native", think=None),
        BatchConfig(model="resumed", backend="ollama", mode="native", think=None),
    ]

    await run_batch(
        configs=configs,
        runs_per_scenario=1,
        output_path=tmp_path / "results.jsonl",
        dry_run=True,
    )

    assert availability_calls == ["missing", "resumed"]
    assert not (tmp_path / "results.jsonl").exists()

    output = tmp_path / "resumed.jsonl"
    output.write_text(
        json.dumps(
            batch_eval._run_result_to_row(
                _result(), configs[1], basic_2step, 1, generation=0,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    before = output.read_bytes()
    availability_calls.clear()

    await run_batch(configs=configs, runs_per_scenario=1, output_path=output)

    assert availability_calls == ["missing", "resumed", "missing", "resumed"]
    assert output.read_bytes() == before


def test_client_factory_uses_manager_url_and_request_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from forge.clients import llamafile, ollama

    constructed: list[tuple[str, dict[str, Any]]] = []

    def ollama_client(**kwargs: Any) -> object:
        constructed.append(("ollama", kwargs))
        return object()

    def llama_client(**kwargs: Any) -> object:
        constructed.append(("llamaserver", kwargs))
        return object()

    monkeypatch.setattr(ollama, "OllamaClient", ollama_client)
    monkeypatch.setattr(llamafile, "LlamafileClient", llama_client)
    batch_eval._build_client(
        BatchConfig(model="tag", backend="ollama", mode="native", think=None),
        tmp_path,
        "http://ollama:11434",
        600,
    )
    batch_eval._build_client(
        BatchConfig(
            model="M", backend="llamaserver", mode="native", think=None,
            gguf_filename="M.gguf",
        ),
        tmp_path,
        "http://coordinator:8080/v1",
        7200,
    )

    assert constructed[0][1]["base_url"] == "http://ollama:11434"
    assert constructed[0][1]["timeout"] == 600
    assert constructed[1][1]["base_url"] == "http://coordinator:8080/v1"
    assert constructed[1][1]["timeout"] == 7200


@pytest.mark.asyncio
async def test_rpc_campaign_topology_client_and_dry_run_are_one_integrated_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    topology_path = tmp_path / "deepseek-rpc.json"
    topology_path.write_text(
        json.dumps({
            "worker": {
                "ssh_target": "operator@10.35.0.5",
                "rpc_host": "10.35.0.5",
                "rpc_port": 50052,
                "executable": "/opt/llama/build/bin/ggml-rpc-server",
                "device": "Vulkan0",
                "tensor_cache": True,
                "environment": [
                    ["AMD_VULKAN_ICD", "RADV"],
                    ["RADV_PERFTEST", "nogttspill"],
                ],
            },
            "coordinator_executable": "/opt/llama/build/bin/llama-server",
            "coordinator_environment": [
                ["AMD_VULKAN_ICD", "RADV"],
                ["RADV_PERFTEST", "nogttspill"],
            ],
            "devices": ["Vulkan0", "RPC0"],
            "tensor_split": [1, 1],
            "split_mode": "layer",
            "startup_timeout": 1800,
            "log_directory": str(tmp_path / "logs"),
        }),
        encoding="utf-8",
    )
    model_path = (
        tmp_path
        / "DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf"
    )
    model_path.touch()

    rpc = batch_eval._load_rpc_topology(topology_path)
    assert rpc.worker.tensor_cache is True
    assert rpc.worker.environment == (
        ("AMD_VULKAN_ICD", "RADV"),
        ("RADV_PERFTEST", "nogttspill"),
    )
    assert rpc.devices == ("Vulkan0", "RPC0")
    assert rpc.tensor_split == (1, 1)
    assert rpc.startup_timeout == 1800

    base_config = batch_eval.DEEPSEEK_V4_RPC_CONFIGS[0]
    with pytest.raises(ValueError, match="requires an attached RPC topology"):
        await run_batch(
            [base_config], 1, tmp_path / "unconfigured.jsonl", dry_run=True,
        )
    attached = batch_eval._attach_rpc_topology([base_config], rpc)
    assert base_config.server_recipe.rpc is None
    assert attached[0] is not base_config
    assert attached[0].server_recipe.rpc is rpc

    client = batch_eval._build_client(
        attached[0], tmp_path, "http://localhost:8080/v1", 7200,
    )
    selected_effort = get_sampling_defaults(base_config.model)[
        "chat_template_kwargs"
    ]["reasoning_effort"]
    assert selected_effort in {"low", "high", "max"}
    assert client.temperature == 1.0
    assert client.top_p == 0.95
    assert client.chat_template_kwargs == {"reasoning_effort": selected_effort}
    await client.aclose()

    inkling_path = tmp_path / "Inkling-Small-UD-IQ4_XS-00001-of-00004.gguf"
    inkling_path.touch()
    inkling_config = batch_eval.INKLING_SMALL_RPC_CONFIGS[0]
    with pytest.raises(ValueError, match="requires an attached RPC topology"):
        await run_batch(
            [inkling_config], 1, tmp_path / "inkling-unconfigured.jsonl",
            dry_run=True,
        )
    inkling_attached = batch_eval._attach_rpc_topology([inkling_config], rpc)
    inkling_client = batch_eval._build_client(
        inkling_attached[0], tmp_path, "http://localhost:8080/v1", 7200,
    )
    assert inkling_client.temperature == 1.0
    assert inkling_client.top_p == 1.0
    assert inkling_client.min_p == 0.0
    assert inkling_client.chat_template_kwargs == {"reasoning_effort": "max"}
    await inkling_client.aclose()

    def fail(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("dry-run reached backend, client, subprocess, or output activity")

    monkeypatch.setattr(batch_eval, "setup_backend", fail)
    monkeypatch.setattr(batch_eval, "_build_client", fail)
    monkeypatch.setattr(batch_eval, "_append_jsonl_row", fail)
    monkeypatch.setattr(batch_eval.subprocess, "run", fail)
    output_path = tmp_path / "must-not-exist.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "batch_eval",
        "--config", "deepseek-v4-rpc",
        "--rpc-topology", str(topology_path),
        "--tags", "plumbing", "model_quality", "advanced_reasoning",
        "--runs", "50",
        "--budget-mode", "manual",
        "--num-ctx", "262144",
        "--request-timeout", "7200",
        "--run-timeout", "7200",
        "--ablation", "reforged",
        "--reasoning-replay", "none",
        "--generation", "3",
        "--models-dir", str(tmp_path),
        "--output", str(output_path),
        "--dry-run",
    ])

    await batch_eval.main()

    output = capsys.readouterr().out
    assert "Config set:    deepseek-v4-rpc (1 configs)" in output
    assert "Scenarios:     26" in output
    assert "Total max runs: 1300" in output
    assert (
        f"reasoning effort (sampling registry): {selected_effort}" in output
    )
    assert "startup timeout: 1800s" in output
    worker_line = next(
        line for line in output.splitlines() if "worker command:" in line
    )
    coordinator_line = next(
        line for line in output.splitlines() if "coordinator command:" in line
    )
    assert worker_line.endswith("-H 10.35.0.5 -p 50052 -d Vulkan0 -c'")
    for owned_arg in (
        "-ngl 999", "--jinja", "--fit off", "-b 2048", "-ub 128",
        "--cache-type-k q8_0", "--cache-type-v q8_0", "--no-mmap",
        "-fa on", "--reasoning-budget 32768", "--reasoning-format auto",
        "--no-prefill-assistant", "--rpc 10.35.0.5:50052",
        "--device Vulkan0,RPC0", "--split-mode layer",
        "--tensor-split 1,1", "-c 262144",
    ):
        assert coordinator_line.count(owned_arg) == 1, owned_arg
    assert not output_path.exists()

    inkling_output_path = tmp_path / "inkling-must-not-exist.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "batch_eval",
        "--config", "inkling-small-rpc",
        "--rpc-topology", str(topology_path),
        "--tags", "plumbing", "model_quality", "advanced_reasoning",
        "--runs", "50",
        "--budget-mode", "manual",
        "--num-ctx", "262144",
        "--request-timeout", "7200",
        "--run-timeout", "7200",
        "--ablation", "reforged",
        "--reasoning-replay", "none",
        "--generation", "3",
        "--models-dir", str(tmp_path),
        "--output", str(inkling_output_path),
        "--dry-run",
    ])
    await batch_eval.main()

    inkling_output = capsys.readouterr().out
    assert "Config set:    inkling-small-rpc (1 configs)" in inkling_output
    assert "Scenarios:     26" in inkling_output
    assert "Total max runs: 1300" in inkling_output
    assert "reasoning effort (sampling registry): max" in inkling_output
    inkling_coordinator = next(
        line for line in inkling_output.splitlines()
        if "coordinator command:" in line
    )
    for owned_arg in (
        "-ngl 999", "--jinja", "--fit off", "-b 512", "-ub 128",
        "--cache-type-k f16", "--cache-type-v f16", "--no-mmap",
        "-fa on", "--parallel 1", "--rpc 10.35.0.5:50052",
        "--device Vulkan0,RPC0", "--split-mode layer",
        "--tensor-split 1,1", "-c 262144",
    ):
        assert inkling_coordinator.count(owned_arg) == 1, owned_arg
    for deepseek_only in (
        "q8_0", "--reasoning-budget", "--no-prefill-assistant",
    ):
        assert deepseek_only not in inkling_coordinator
    assert not inkling_output_path.exists()


@pytest.mark.asyncio
async def test_setup_failure_is_single_call_and_returns_no_owned_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_batch(monkeypatch)
    setup_count = 0

    async def setup(**kwargs: Any) -> Any:
        nonlocal setup_count
        setup_count += 1
        raise RuntimeError("startup timeout")

    monkeypatch.setattr(batch_eval, "setup_backend", setup)
    monkeypatch.setattr(
        batch_eval,
        "_build_client",
        lambda *args: pytest.fail("client construction was reached"),
    )

    await run_batch(
        configs=[BatchConfig(model="M", backend="ollama", mode="native", think=None)],
        runs_per_scenario=1,
        output_path=tmp_path / "results.jsonl",
    )

    assert setup_count == 1
    assert not (tmp_path / "results.jsonl").exists()


@pytest.mark.asyncio
async def test_infrastructure_failure_restarts_rebuilds_and_retries_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_batch(monkeypatch)
    server = _Server("http://localhost:11434", 4096)

    async def setup(**kwargs: Any) -> tuple[_Server, SimpleNamespace]:
        return server, SimpleNamespace(budget_tokens=server.budget)

    clients: list[object] = []

    def build_client(*args: Any) -> object:
        client = object()
        clients.append(client)
        return client

    results = [
        _result(
            completed=False,
            error_type="BackendError",
            error_message="ConnectError: worker disconnected",
        ),
        _result(),
    ]
    run_clients: list[object] = []

    async def run_once(*args: Any) -> RunResult:
        run_clients.append(args[0])
        return results.pop(0)

    monkeypatch.setattr(batch_eval, "setup_backend", setup)
    monkeypatch.setattr(batch_eval, "_build_client", build_client)
    monkeypatch.setattr(batch_eval, "_run_with_timeout", run_once)
    monkeypatch.setattr(batch_eval, "_RECOVERY_BACKOFFS", [0, 0, 0])

    output = tmp_path / "results.jsonl"
    await run_batch(
        configs=[BatchConfig(model="M", backend="ollama", mode="native", think=None)],
        runs_per_scenario=1,
        output_path=output,
    )

    assert server.restart_count == 1
    assert server.stop_count == 2  # recovery pre-stop, then configuration boundary
    assert run_clients == clients
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.asyncio
async def test_recovery_backoffs_circuit_and_whole_run_timeout_remain_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server("http://localhost:8080/v1", 4096)
    sleeps: list[int] = []

    async def sleep(seconds: int) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(batch_eval.asyncio, "sleep", sleep)
    for crash_count in (1, 2, 3):
        assert await batch_eval._recover_server(
            server, crash_count, BudgetMode.BACKEND, None,
        ) == 4096
    assert await batch_eval._recover_server(
        server, 4, BudgetMode.BACKEND, None,
    ) is None
    assert sleeps == [30, 60, 300]
    assert server.restart_count == 3

    async def fail_budget(*args: Any) -> int:
        raise RuntimeError("props unavailable")

    monkeypatch.setattr(server, "resolve_budget", fail_budget)
    assert await batch_eval._recover_server(
        server, 1, BudgetMode.BACKEND, None,
    ) is None

    async def never_returns(*args: Any, **kwargs: Any) -> RunResult:
        await batch_eval.asyncio.Future()

    monkeypatch.setattr(batch_eval, "run_scenario", never_returns)
    timeout_result = await batch_eval._run_with_timeout(
        object(), basic_2step, SimpleNamespace(), None, run_timeout=0.001,
    )
    assert timeout_result.error_type == "Timeout"
    assert timeout_result.error_message == "Exceeded 0.001s"
    assert not batch_eval._is_server_error(timeout_result)
