"""Tests for public llama.cpp RPC configuration and command rendering."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from forge import LlamaCppRpcConfig, LlamaCppRpcWorkerConfig
from forge.rpc import (
    render_rpc_coordinator_args,
    render_rpc_worker_command,
    validate_rpc_extra_flags,
)


@pytest.fixture()
def rig_rpc() -> LlamaCppRpcConfig:
    worker = LlamaCppRpcWorkerConfig(
        ssh_target="antoine@10.35.0.5",
        rpc_host="10.35.0.5",
        executable=(
            "/home/antoine/Documents/llama.cpp-dsv4-hc-ccbc178/"
            "build/bin/ggml-rpc-server"
        ),
        device="Vulkan0",
        environment=(
            ("AMD_VULKAN_ICD", "RADV"),
            ("RADV_PERFTEST", "nogttspill"),
        ),
    )
    return LlamaCppRpcConfig(
        worker=worker,
        coordinator_executable=(
            "/home/antoine/Documents/llama.cpp-dsv4-hc-ccbc178/"
            "build/bin/llama-server"
        ),
        coordinator_environment=(
            ("AMD_VULKAN_ICD", "RADV"),
            ("RADV_PERFTEST", "nogttspill"),
        ),
        devices=("Vulkan0", "RPC0"),
        tensor_split=(1, 1),
        startup_timeout=1800,
    )


def test_render_worker_matches_proven_rig_command(
    rig_rpc: LlamaCppRpcConfig,
) -> None:
    assert render_rpc_worker_command(rig_rpc.worker) == [
        "ssh",
        "-tt",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "antoine@10.35.0.5",
        (
            "exec env AMD_VULKAN_ICD=RADV RADV_PERFTEST=nogttspill "
            "/home/antoine/Documents/llama.cpp-dsv4-hc-ccbc178/"
            "build/bin/ggml-rpc-server -H 10.35.0.5 -p 50052 -d Vulkan0 -c"
        ),
    ]


def test_render_worker_quotes_remote_values() -> None:
    worker = LlamaCppRpcWorkerConfig(
        ssh_target="worker",
        rpc_host="10.0.0.5",
        executable="/opt/llama cpp/ggml-rpc-server",
        device="Vulkan 0",
        tensor_cache=False,
        environment=(("TRACE_LABEL", "rpc smoke"),),
        ssh_options=(),
    )
    assert render_rpc_worker_command(worker) == [
        "ssh",
        "-tt",
        "worker",
        (
            "exec env 'TRACE_LABEL=rpc smoke' '/opt/llama cpp/ggml-rpc-server' "
            "-H 10.0.0.5 -p 50052 -d 'Vulkan 0'"
        ),
    ]


def test_render_coordinator_matches_proven_rig_topology(
    rig_rpc: LlamaCppRpcConfig,
) -> None:
    assert render_rpc_coordinator_args(rig_rpc) == [
        "--rpc",
        "10.35.0.5:50052",
        "--device",
        "Vulkan0,RPC0",
        "--split-mode",
        "layer",
        "--tensor-split",
        "1,1",
    ]


def test_rpc_topology_flag_ownership() -> None:
    duplicate_flags = [
        ["--rpc", "10.0.0.5:50052"],
        ["--device=Vulkan0,RPC0"],
        ["-sm", "layer"],
        ["--tensor-split=1,1"],
    ]
    for flags in duplicate_flags:
        with pytest.raises(ValueError, match="RPC topology owns"):
            validate_rpc_extra_flags(flags)
    validate_rpc_extra_flags(["--fit", "off", "-fa", "on"])


def test_config_is_frozen(rig_rpc: LlamaCppRpcConfig) -> None:
    with pytest.raises(FrozenInstanceError):
        rig_rpc.startup_timeout = 12  # type: ignore[misc]


def test_invalid_worker_config_fails_early() -> None:
    invalid_cases = [
        ({"rpc_host": "0.0.0.0"}, "direct-link"),
        ({"rpc_port": 0}, "rpc_port"),
        ({"ssh_target": ""}, "ssh_target"),
        ({"environment": (("BAD=KEY", "x"),)}, "invalid variable"),
    ]
    for kwargs, message in invalid_cases:
        values: dict[str, object] = {
            "ssh_target": "worker",
            "rpc_host": "10.0.0.5",
            "rpc_port": 50052,
            "executable": "/opt/ggml-rpc-server",
            "device": "Vulkan0",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=message):
            LlamaCppRpcWorkerConfig(**values)  # type: ignore[arg-type]


def test_tensor_split_must_match_devices(
    rig_rpc: LlamaCppRpcConfig,
) -> None:
    with pytest.raises(ValueError, match="one value per device"):
        LlamaCppRpcConfig(
            worker=rig_rpc.worker,
            coordinator_executable=rig_rpc.coordinator_executable,
            devices=("Vulkan0", "RPC0"),
            tensor_split=(1,),
        )
