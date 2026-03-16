"""Tests for forge.context.hardware — detection + VRAM budget estimation."""

from unittest.mock import patch
import subprocess

import pytest

from forge.context.hardware import (
    HardwareProfile,
    detect_hardware,
)
from forge.errors import HardwareDetectionError


class TestHardwareProfile:
    """Tests for HardwareProfile dataclass."""

    def test_vram_total_gb(self) -> None:
        hw = HardwareProfile(gpu_name="RTX 5070", vram_total_mb=12288)
        assert hw.vram_total_gb == 12.0

    def test_vram_total_gb_fractional(self) -> None:
        hw = HardwareProfile(gpu_name="RTX 5070", vram_total_mb=12000)
        assert hw.vram_total_gb == 12000 / 1024


class TestDetectHardware:
    """Tests for detect_hardware() function."""

    def test_returns_none_when_nvidia_smi_not_found(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert detect_hardware() is None

    def test_returns_none_on_nonzero_exit(self) -> None:
        result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")
        with patch("subprocess.run", return_value=result):
            assert detect_hardware() is None

    def test_raises_on_malformed_output(self) -> None:
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout="garbage", stderr="")
        with patch("subprocess.run", return_value=result):
            with pytest.raises(HardwareDetectionError) as exc_info:
                detect_hardware()
            assert isinstance(exc_info.value.cause, ValueError)

    def test_returns_none_on_timeout(self) -> None:
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)):
            assert detect_hardware() is None

    def test_returns_profile_on_success(self) -> None:
        nvidia_output = "NVIDIA GeForce RTX 5070, 12288\n"
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout=nvidia_output, stderr="")
        with patch("subprocess.run", return_value=result):
            hw = detect_hardware()
            assert hw is not None
            assert hw.gpu_name == "NVIDIA GeForce RTX 5070"
            assert hw.vram_total_mb == 12288

    def test_handles_multi_gpu_uses_first(self) -> None:
        nvidia_output = (
            "NVIDIA GeForce RTX 3090, 24576\n"
            "NVIDIA GeForce RTX 3060, 12288\n"
        )
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout=nvidia_output, stderr="")
        with patch("subprocess.run", return_value=result):
            hw = detect_hardware()
            assert hw is not None
            assert hw.gpu_name == "NVIDIA GeForce RTX 3090"
            assert hw.vram_total_mb == 24576


