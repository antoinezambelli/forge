"""Tests for forge.context.hardware — detection + VRAM budget estimation."""

import logging
import subprocess
from pathlib import Path
from unittest.mock import patch

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

    def test_defaults_to_nvidia_discrete(self) -> None:
        hw = HardwareProfile(gpu_name="RTX 5070", vram_total_mb=12288)
        assert hw.gpu_vendor == "nvidia"
        assert hw.memory_kind == "discrete"


class TestDetectHardwareNvidia:
    """nvidia-smi probe path."""

    def test_returns_none_when_nvidia_smi_not_found_and_no_amd(self, tmp_path: Path) -> None:
        # FileNotFoundError on nvidia-smi + empty drm root => fall through to None.
        with patch("subprocess.run", side_effect=FileNotFoundError), \
             patch("forge.context.hardware.Path", return_value=tmp_path):
            assert detect_hardware() is None

    def test_returns_none_on_nonzero_exit_and_no_amd(self, tmp_path: Path) -> None:
        result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")
        with patch("subprocess.run", return_value=result), \
             patch("forge.context.hardware.Path", return_value=tmp_path):
            assert detect_hardware() is None

    def test_raises_on_malformed_output(self) -> None:
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout="garbage", stderr="")
        with patch("subprocess.run", return_value=result):
            with pytest.raises(HardwareDetectionError) as exc_info:
                detect_hardware()
            assert isinstance(exc_info.value.cause, ValueError)

    def test_returns_none_on_timeout_and_no_amd(self, tmp_path: Path) -> None:
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)), \
             patch("forge.context.hardware.Path", return_value=tmp_path):
            assert detect_hardware() is None

    def test_returns_profile_on_success(self) -> None:
        nvidia_output = "NVIDIA GeForce RTX 5070, 12288\n"
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout=nvidia_output, stderr="")
        with patch("subprocess.run", return_value=result):
            hw = detect_hardware()
            assert hw is not None
            assert hw.gpu_name == "NVIDIA GeForce RTX 5070"
            assert hw.vram_total_mb == 12288
            assert hw.gpu_vendor == "nvidia"
            assert hw.memory_kind == "discrete"

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


def _make_amd_card(drm_root: Path, card_name: str, vram_bytes: int, vendor: str = "0x1002", with_uevent: bool = True) -> None:
    """Build a minimal /sys/class/drm/<card>/device/ fixture under *drm_root*."""
    device_dir = drm_root / card_name / "device"
    device_dir.mkdir(parents=True)
    (device_dir / "vendor").write_text(f"{vendor}\n")
    (device_dir / "mem_info_vram_total").write_text(f"{vram_bytes}\n")
    if with_uevent:
        (device_dir / "uevent").write_text(
            "DRIVER=amdgpu\nPCI_CLASS=30000\nPCI_ID=1002:150E\n"
        )


class TestDetectHardwareAmdSysfs:
    """AMD sysfs probe path — fires when nvidia-smi is unavailable."""

    @pytest.fixture
    def fake_drm(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        drm_root = tmp_path / "drm"
        drm_root.mkdir()
        # Stub the Path("/sys/class/drm") call inside _detect_amd_sysfs.
        original_path = Path

        def fake_path(arg: str) -> Path:
            if arg == "/sys/class/drm":
                return drm_root
            return original_path(arg)

        monkeypatch.setattr("forge.context.hardware.Path", fake_path)
        return drm_root

    def test_amd_sysfs_when_nvidia_absent(self, fake_drm: Path) -> None:
        # 64 GiB unified — Strix Halo BIOS carve-out.
        _make_amd_card(fake_drm, "card1", vram_bytes=68_719_476_736)
        with patch("subprocess.run", side_effect=FileNotFoundError):
            hw = detect_hardware()
            assert hw is not None
            assert hw.gpu_vendor == "amd"
            assert hw.memory_kind == "unified"
            assert hw.vram_total_mb == 65_536  # 64 GiB
            assert hw.vram_total_gb == 64.0
            assert "1002:150E" in hw.gpu_name

    def test_amd_sysfs_skips_non_amd_card(self, fake_drm: Path) -> None:
        # Intel iGPU first, AMD discrete/APU second — must pick AMD.
        _make_amd_card(fake_drm, "card0", vram_bytes=2_147_483_648, vendor="0x8086", with_uevent=False)
        _make_amd_card(fake_drm, "card1", vram_bytes=68_719_476_736)
        with patch("subprocess.run", side_effect=FileNotFoundError):
            hw = detect_hardware()
            assert hw is not None
            assert hw.gpu_vendor == "amd"
            assert hw.vram_total_mb == 65_536

    def test_no_card_when_drm_empty(self, fake_drm: Path, caplog: pytest.LogCaptureFixture) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError), \
             caplog.at_level(logging.WARNING, logger="forge.context.hardware"):
            assert detect_hardware() is None
            assert any("GPU detection failed" in rec.message for rec in caplog.records)

    def test_no_amd_card_falls_through_with_warn(self, fake_drm: Path, caplog: pytest.LogCaptureFixture) -> None:
        # Only an Intel card present.
        _make_amd_card(fake_drm, "card0", vram_bytes=2_147_483_648, vendor="0x8086", with_uevent=False)
        with patch("subprocess.run", side_effect=FileNotFoundError), \
             caplog.at_level(logging.WARNING, logger="forge.context.hardware"):
            assert detect_hardware() is None
            assert any("GPU detection failed" in rec.message for rec in caplog.records)

    def test_warn_lists_attempted_probes(self, fake_drm: Path, caplog: pytest.LogCaptureFixture) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError), \
             caplog.at_level(logging.WARNING, logger="forge.context.hardware"):
            detect_hardware()
            warns = [rec.message for rec in caplog.records if rec.levelno == logging.WARNING]
            assert len(warns) == 1
            assert "nvidia-smi: not installed" in warns[0]
            assert "amd-sysfs" in warns[0]
