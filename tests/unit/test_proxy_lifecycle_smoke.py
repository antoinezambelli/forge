"""Selected-artifact lifecycle smoke orchestration tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.standalone import lifecycle_smoke
from scripts.standalone.inputs import SUPPORTED_TARGETS
from scripts.standalone.release import artifact_name


def test_windows_shim_command_uses_cmd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifecycle_smoke.os, "name", "nt")
    assert lifecycle_smoke.command(Path("forge-proxy.cmd"), ["check"]) == [
        "cmd",
        "/d",
        "/c",
        "forge-proxy.cmd",
        "check",
    ]


def test_expected_failure_is_a_successful_gate_observation(tmp_path: Path) -> None:
    def runner(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(arguments, 2, "", "download unavailable")

    record = lifecycle_smoke.run_process(
        ["forge-proxy", "update"],
        cwd=tmp_path,
        env={},
        runner=runner,
        expected_error="download unavailable",
    )
    assert record["status"] == 2
    assert record["expected_failure"] == "download unavailable"


def test_unexpected_success_fails_a_failure_gate(tmp_path: Path) -> None:
    def runner(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(arguments, 0, "", "")

    with pytest.raises(RuntimeError, match="unexpectedly succeeded"):
        lifecycle_smoke.run_process(
            ["forge-proxy", "update"],
            cwd=tmp_path,
            env={},
            runner=runner,
            expected_error="download unavailable",
        )


def test_missing_stable_pointer_means_no_published_baseline(tmp_path: Path) -> None:
    def reader(url: str, *, missing_ok: bool = False) -> bytes | None:
        assert url == "https://fixture.invalid/pointer"
        assert missing_ok is True
        return None

    assert (
        lifecycle_smoke.retrievable_published_baseline(
            "windows-x86_64",
            tmp_path,
            pointer_url="https://fixture.invalid/pointer",
            reader=reader,
        )
        is None
    )


def test_unavailable_published_manifest_means_no_cross_version_checks(
    tmp_path: Path,
) -> None:
    def reader(url: str, *, missing_ok: bool = False) -> bytes | None:
        if url == "https://fixture.invalid/pointer":
            assert missing_ok is True
            return b"1.2.3\n"
        raise RuntimeError(f"download unavailable: {url}")

    assert (
        lifecycle_smoke.retrievable_published_baseline(
            "windows-x86_64",
            tmp_path,
            pointer_url="https://fixture.invalid/pointer",
            release_base_url="https://fixture.invalid/releases",
            reader=reader,
        )
        is None
    )


def test_invalid_published_manifest_means_no_cross_version_checks(
    tmp_path: Path,
) -> None:
    def reader(url: str, *, missing_ok: bool = False) -> bytes | None:
        if url == "https://fixture.invalid/pointer":
            assert missing_ok is True
            return b"1.2.3\n"
        return b"not a manifest"

    assert (
        lifecycle_smoke.retrievable_published_baseline(
            "windows-x86_64",
            tmp_path,
            pointer_url="https://fixture.invalid/pointer",
            release_base_url="https://fixture.invalid/releases",
            reader=reader,
        )
        is None
    )


def test_published_baseline_is_manifest_verified(tmp_path: Path) -> None:
    target = "windows-x86_64"
    payload = b"published baseline bytes"
    digest = hashlib.sha256(payload).hexdigest()
    version = "1.2.3"
    artifacts = {
        item: {
            "name": artifact_name(item),
            "sha256": digest if item == target else "a" * 64,
            "size": len(payload) if item == target else 1,
        }
        for item in SUPPORTED_TARGETS
    }
    routes = {
        "https://fixture.invalid/pointer": f"{version}\n".encode(),
        f"https://fixture.invalid/releases/v{version}/proxy-{version}.json": (
            json.dumps({"version": version, "artifacts": artifacts}).encode()
        ),
        f"https://fixture.invalid/releases/v{version}/{artifact_name(target)}": payload,
    }

    def reader(url: str, *, missing_ok: bool = False) -> bytes | None:
        del missing_ok
        return routes[url]

    baseline = lifecycle_smoke.resolve_published_baseline(
        target,
        tmp_path,
        pointer_url="https://fixture.invalid/pointer",
        release_base_url="https://fixture.invalid/releases",
        reader=reader,
    )
    assert baseline is not None
    assert baseline.version == version
    assert baseline.sha256 == digest
    assert baseline.path.read_bytes() == payload


def test_local_release_routes_can_declare_a_bad_checksum(tmp_path: Path) -> None:
    path = tmp_path / "forge-proxy-windows-x86_64.exe"
    path.write_bytes(b"candidate")
    artifact = lifecycle_smoke.ReleaseArtifact(
        path,
        "1.2.4",
        lifecycle_smoke.artifact_sha256(path),
        "windows-x86_64",
    )
    routes: dict[str, tuple[int, bytes]] = {}
    lifecycle_smoke.set_release_routes(routes, artifact, sha256="f" * 64)
    manifest = json.loads(routes["/v1.2.4/proxy-1.2.4.json"][1])
    assert manifest["artifacts"][artifact.target]["sha256"] == "f" * 64
    assert routes[f"/v1.2.4/{artifact.name}"][1] == b"candidate"


def test_inaugural_version_still_has_a_higher_failure_target() -> None:
    assert lifecycle_smoke.next_patch_version("0.9.0") == "0.9.1"
    with pytest.raises(ValueError, match="invalid release version"):
        lifecycle_smoke.next_patch_version("0.09.0")


def test_bootstrap_arguments_are_exact_version_and_noninteractive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forge-proxy"
    path.write_bytes(b"candidate")
    artifact = lifecycle_smoke.ReleaseArtifact(
        path, "1.2.3", "a" * 64, "linux-x86_64-gnu"
    )
    arguments = lifecycle_smoke.bootstrap_arguments(artifact, tmp_path / "root")
    assert arguments[:2] == ["sh", str(lifecycle_smoke.ROOT / "install.sh")]
    assert arguments[arguments.index("--version") + 1] == "1.2.3"
    assert "--no-init" in arguments
    assert arguments[arguments.index("--install-root") + 1] == str(tmp_path / "root")
