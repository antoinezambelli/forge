"""Windows acceptance checks that execute installer-owned native processes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from scripts.standalone.release import project_version
from tests.integration._bootstrap_support import ROOT, fixture_server, run_powershell
from tests.unit.test_proxy_installer import (
    FakeRunner,
    adapter,
    artifact,
    install,
    windows_paths,
)


pytestmark = [pytest.mark.integration, pytest.mark.acceptance]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_snapshot(root: Path) -> tuple[tuple[str, str], ...] | None:
    if not root.exists():
        return None
    rows: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        rows.append((relative, sha256(path) if path.is_file() else "directory"))
    return tuple(rows)


def user_path() -> str | None:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return str(winreg.QueryValueEx(key, "Path")[0])
    except FileNotFoundError:
        return None


def installed_bytes(root: Path, version: str) -> tuple[bytes, bytes, bytes]:
    return (
        (root / "bin" / "forge-proxy.cmd").read_bytes(),
        (root / "state.json").read_bytes(),
        (root / "versions" / version / "forge-proxy.exe").read_bytes(),
    )


def wait_for_uninstall(root: Path) -> None:
    deadline = time.monotonic() + 15
    while (root / "state.json").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not (root / "state.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows packaged-artifact lifecycle")
def test_windows_frozen_artifact_install_failure_and_uninstall(
    tmp_path: Path,
) -> None:
    frozen = ROOT / "standalone-dist" / "windows-x86_64" / "onefile" / "forge-proxy.exe"
    if not frozen.is_file():
        pytest.skip("packaged Windows artifact is unavailable")
    version = project_version()
    name = "forge-proxy-windows-x86_64.exe"
    manifest = json.dumps(
        {
            "artifacts": {
                "windows-x86_64": {
                    "name": name,
                    "sha256": sha256(frozen),
                    "size": frozen.stat().st_size,
                }
            },
            "version": version,
        },
        indent=2,
        sort_keys=True,
    ).encode()
    routes = {
        "/pointer": (200, f"{version}\n".encode()),
        f"/v{version}/proxy-{version}.json": (200, manifest),
        f"/v{version}/{name}": (200, frozen.read_bytes()),
    }

    real_local = Path(os.environ["LOCALAPPDATA"]) / "Forge"
    real_tree_before = tree_snapshot(real_local)
    real_path_before = user_path()
    redirected = {
        "APPDATA": str(tmp_path / "appdata"),
        "LOCALAPPDATA": str(tmp_path / "localappdata"),
        "FORGE_PROXY_PATH_FILE": str(tmp_path / "fixture-path.txt"),
    }
    Path(redirected["APPDATA"]).mkdir()
    Path(redirected["LOCALAPPDATA"]).mkdir()
    Path(redirected["FORGE_PROXY_PATH_FILE"]).write_text(
        "C:\\Existing", encoding="utf-8"
    )

    with fixture_server(routes) as (_server, base_url):
        install_root = tmp_path / "exact root with spaces"
        result = run_powershell(
            tmp_path,
            base_url,
            ["-Version", version, "-NoInit", "-InstallRoot", str(install_root)],
            extra_env=redirected,
        )
        assert result.returncode == 0, result.stderr
        assert "forge-proxy init --non-interactive" in result.stdout
        before_failure = installed_bytes(install_root, version)

        bad_document = json.loads(manifest)
        bad_document["artifacts"]["windows-x86_64"]["sha256"] = "f" * 64
        routes[f"/v{version}/proxy-{version}.json"] = (
            200,
            json.dumps(bad_document).encode(),
        )
        failed = run_powershell(
            tmp_path,
            base_url,
            ["-Version", version, "-NoInit", "-InstallRoot", str(install_root)],
            extra_env=redirected,
        )
        assert failed.returncode == 1
        assert "checksum mismatch" in failed.stderr
        assert installed_bytes(install_root, version) == before_failure

        profile = Path(redirected["APPDATA"]) / "Forge" / "profiles" / "default.toml"
        profile.parent.mkdir(parents=True)
        profile.write_text("backend = 'openai'\n", encoding="utf-8")
        uninstall_env = os.environ.copy()
        uninstall_env.update(redirected)
        uninstall = subprocess.run(
            [
                str(install_root / "versions" / version / "forge-proxy.exe"),
                "uninstall",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=uninstall_env,
            timeout=30,
        )
        assert uninstall.returncode == 0, uninstall.stderr
        wait_for_uninstall(install_root)

    assert profile.read_text(encoding="utf-8") == "backend = 'openai'\n"
    assert Path(redirected["FORGE_PROXY_PATH_FILE"]).read_text() == "C:\\Existing"
    assert not list((tmp_path / "bootstrap temp").iterdir())
    assert user_path() == real_path_before
    assert tree_snapshot(real_local) == real_tree_before


@pytest.mark.skipif(os.name != "nt", reason="Windows file locking behavior")
def test_windows_locked_slot_preserves_uninstall_retry_path(tmp_path: Path) -> None:
    paths = windows_paths(tmp_path, "locked native root")
    path_file = adapter(tmp_path, "C:\\Existing")
    source, sha = artifact(tmp_path, "1.0.0")
    install(source, sha, "1.0.0", paths, FakeRunner(), path_file)

    with paths.slot("1.0.0").open("rb"):
        subprocess.run(
            [str(paths.uninstaller), "999999"],
            capture_output=True,
            text=True,
            check=False,
            shell=True,
            timeout=15,
        )
        assert paths.command.is_file()
        assert paths.state.is_file()
        assert paths.marker.is_file()
        assert paths.uninstaller.is_file()
        assert str(paths.command_dir) in (tmp_path / "user-path.txt").read_text()

    result = subprocess.run(
        [str(paths.uninstaller), "999999"],
        capture_output=True,
        text=True,
        check=False,
        shell=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    deadline = time.monotonic() + 10
    while paths.state.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not paths.state.exists()
    assert not paths.command.exists()
    assert (tmp_path / "user-path.txt").read_text() == "C:\\Existing"
