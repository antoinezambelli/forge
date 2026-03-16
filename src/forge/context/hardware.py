"""Hardware detection for GPU capabilities.

detect_hardware() reads total VRAM from nvidia-smi (a stable number that
doesn't change with allocations).  Used by ServerManager for VRAM tier
lookup.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from forge.errors import HardwareDetectionError

# Bits-per-weight for common GGUF quantisation levels.
_QUANT_BPW: dict[str, float] = {
    "Q4_0": 4.0,
    "Q4_K_M": 4.83,
    "Q4_K_S": 4.58,
    "Q5_0": 5.0,
    "Q5_K_M": 5.68,
    "Q5_K_S": 5.52,
    "Q6_K": 6.56,
    "Q8_0": 8.0,
    "F16": 16.0,
}


@dataclass
class HardwareProfile:
    """Detected GPU capabilities (total VRAM only — a stable value)."""

    gpu_name: str
    vram_total_mb: int

    @property
    def vram_total_gb(self) -> float:
        return self.vram_total_mb / 1024


def detect_hardware() -> HardwareProfile | None:
    """Auto-detect GPU via nvidia-smi. Returns None if detection fails."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    try:
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Expected 2 CSV fields, got {len(parts)}: {line!r}")

        return HardwareProfile(
            gpu_name=parts[0],
            vram_total_mb=int(parts[1]),
        )
    except (ValueError, IndexError) as exc:
        raise HardwareDetectionError(exc) from exc
