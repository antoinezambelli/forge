"""Configuration and command rendering for experimental llama.cpp RPC.

Forge currently supports one remote ``ggml-rpc-server`` worker connected to
one local ``llama-server`` coordinator.  The worker is a foreground SSH child;
its lifetime is expected to follow that SSH channel.
"""

from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path


_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOPOLOGY_FLAGS = {
    "--rpc",
    "--device",
    "-dev",
    "--split-mode",
    "-sm",
    "--tensor-split",
    "-ts",
}


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _validate_environment(
    environment: tuple[tuple[str, str], ...],
    field_name: str,
) -> None:
    seen: set[str] = set()
    for key, _ in environment:
        if not _ENVIRONMENT_KEY.fullmatch(key):
            raise ValueError(f"{field_name} contains invalid variable name {key!r}")
        if key in seen:
            raise ValueError(f"{field_name} contains duplicate variable {key!r}")
        seen.add(key)


@dataclass(frozen=True)
class LlamaCppRpcWorkerConfig:
    """One remote llama.cpp RPC worker launched through foreground SSH."""

    ssh_target: str
    rpc_host: str
    executable: str
    device: str
    rpc_port: int = 50052
    tensor_cache: bool = True
    environment: tuple[tuple[str, str], ...] = ()
    ssh_options: tuple[str, ...] = (
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
    )

    def __post_init__(self) -> None:
        _require_text(self.ssh_target, "ssh_target")
        _require_text(self.rpc_host, "rpc_host")
        _require_text(self.executable, "executable")
        _require_text(self.device, "device")
        if self.rpc_host in {"0.0.0.0", "::", "*"}:
            raise ValueError("rpc_host must be the worker's direct-link address")
        if not 1 <= self.rpc_port <= 65535:
            raise ValueError("rpc_port must be between 1 and 65535")
        if any(not option for option in self.ssh_options):
            raise ValueError("ssh_options must not contain empty arguments")
        _validate_environment(self.environment, "environment")


@dataclass(frozen=True)
class LlamaCppRpcConfig:
    """Experimental one-worker llama.cpp RPC topology.

    Abrupt SSH transport loss that leaves a remote worker behind is outside
    this lightweight lifecycle.  Forge owns and stops the foreground SSH child
    it starts; it does not search for or attach to arbitrary remote processes.
    """

    worker: LlamaCppRpcWorkerConfig
    coordinator_executable: str
    devices: tuple[str, ...]
    tensor_split: tuple[float, ...]
    split_mode: str = "layer"
    startup_timeout: float = 180.0
    coordinator_environment: tuple[tuple[str, str], ...] = ()
    log_directory: str | Path | None = None

    def __post_init__(self) -> None:
        _require_text(self.coordinator_executable, "coordinator_executable")
        if not self.devices or any(not device.strip() for device in self.devices):
            raise ValueError("devices must contain at least one non-empty device")
        if len(self.tensor_split) != len(self.devices):
            raise ValueError("tensor_split must have one value per device")
        if any(not math.isfinite(value) or value < 0 for value in self.tensor_split):
            raise ValueError("tensor_split values must be finite and non-negative")
        if not any(value > 0 for value in self.tensor_split):
            raise ValueError("tensor_split must contain at least one positive value")
        if self.split_mode not in {"none", "layer", "row", "tensor"}:
            raise ValueError("split_mode must be none, layer, row, or tensor")
        if not math.isfinite(self.startup_timeout) or self.startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")
        _validate_environment(
            self.coordinator_environment,
            "coordinator_environment",
        )
        if self.log_directory is not None:
            object.__setattr__(self, "log_directory", Path(self.log_directory))


def render_rpc_worker_command(config: LlamaCppRpcWorkerConfig) -> list[str]:
    """Render the exact local argv for the foreground SSH worker."""

    remote_argv = [
        "env",
        *(f"{key}={value}" for key, value in config.environment),
        config.executable,
        "-H",
        config.rpc_host,
        "-p",
        str(config.rpc_port),
        "-d",
        config.device,
    ]
    if config.tensor_cache:
        remote_argv.append("-c")
    remote_command = f"exec {shlex.join(remote_argv)}"
    # Force a remote PTY so terminating Forge's foreground SSH child sends a
    # hangup to the exec'd worker instead of leaving it listening remotely.
    return ["ssh", "-tt", *config.ssh_options, config.ssh_target, remote_command]


def render_rpc_coordinator_args(config: LlamaCppRpcConfig) -> list[str]:
    """Render llama-server arguments owned by the RPC topology."""

    tensor_split = ",".join(format(value, ".15g") for value in config.tensor_split)
    return [
        "--rpc",
        f"{config.worker.rpc_host}:{config.worker.rpc_port}",
        "--device",
        ",".join(config.devices),
        "--split-mode",
        config.split_mode,
        "--tensor-split",
        tensor_split,
    ]


def validate_rpc_extra_flags(extra_flags: list[str] | None) -> None:
    """Reject topology options that Forge would otherwise render twice."""

    for token in extra_flags or ():
        flag = token.split("=", 1)[0]
        if flag in _TOPOLOGY_FLAGS:
            raise ValueError(
                f"RPC topology owns {flag}; do not also pass it in extra_flags"
            )
