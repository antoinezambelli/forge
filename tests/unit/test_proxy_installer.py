"""Fixture-only coverage for the installed standalone Proxy lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge.proxy import _installer


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, list[str]]] = []
        self.fail_payloads: set[bytes] = set()
        self.profile_incompatible = False

    def run(
        self, executable: Path, arguments: list[str]
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((executable, arguments))
        payload = executable.read_bytes()
        if arguments[0] == "_installer-self-check":
            expected = arguments[arguments.index("--expected-version") + 1]
            embedded = payload.decode().split()[1]
            if payload in self.fail_payloads:
                return subprocess.CompletedProcess([], 1, "", "health failed")
            if embedded != expected:
                return subprocess.CompletedProcess([], 2, "", "version mismatch")
        if arguments[0] == "_installer-profile-check" and self.profile_incompatible:
            return subprocess.CompletedProcess(
                [], 1, "Incompatible managed profile old: unsupported field\n", ""
            )
        return subprocess.CompletedProcess([], 0, "", "")


class FixtureTransport:
    def __init__(self, values: dict[str, bytes | Exception]) -> None:
        self.values = values
        self.reads: list[str] = []

    def read(self, url: str) -> bytes:
        self.reads.append(url)
        value = self.values[url]
        if isinstance(value, Exception):
            raise value
        return value


def artifact(tmp_path: Path, version: str, suffix: str = "") -> tuple[Path, str]:
    path = tmp_path / f"source-{version}-{suffix}.exe"
    path.write_bytes(f"artifact {version} {suffix}".encode())
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def windows_paths(tmp_path: Path, name: str = "install") -> _installer.InstallPaths:
    root = tmp_path / name
    return _installer.InstallPaths(root, root / "bin", "Windows")


def adapter(tmp_path: Path, initial: str = "") -> _installer.WindowsPathAdapter:
    representation = tmp_path / "user-path.txt"
    representation.write_text(initial, encoding="utf-8")
    return _installer.WindowsPathAdapter(representation)


def install(
    source: Path,
    checksum: str,
    version: str,
    paths: _installer.InstallPaths,
    runner: FakeRunner,
    path_adapter: _installer.PathAdapter,
    **kwargs: object,
) -> dict[str, object]:
    kwargs.setdefault("environ", {"PATH": ""})
    return _installer.install_artifact(
        source,
        version,
        checksum,
        paths=paths,
        runner=runner,
        path_adapter=path_adapter,
        no_init=True,
        output=lambda _line: None,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("value", ["1.2", "v1.2.3", "1.2.3 ", "01.2.3", "1.2.3-beta"])
def test_version_is_strict(value: str) -> None:
    with pytest.raises(_installer.InstallerError, match="expected X.Y.Z"):
        _installer.parse_version(value)


def test_numeric_versions_do_not_compare_lexically() -> None:
    assert _installer.parse_version("0.9.9") < _installer.parse_version("0.10.0")


@pytest.mark.parametrize("payload", [b"1.2.3", b"1.2.3\n"])
def test_pointer_accepts_one_bare_version(payload: bytes) -> None:
    assert _installer.parse_pointer(payload) == "1.2.3"


@pytest.mark.parametrize("payload", [b" 1.2.3\n", b"1.2.3\r\n", b"1.2.3\nextra\n"])
def test_pointer_rejects_non_bare_content(payload: bytes) -> None:
    with pytest.raises(_installer.InstallerError):
        _installer.parse_pointer(payload)


def test_url_transport_distinguishes_missing_pointer_from_unavailable_fetch() -> None:
    missing = _installer.urllib.error.HTTPError(
        _installer.STABLE_POINTER_URL, 404, "missing", {}, None
    )
    with (
        patch.object(_installer.urllib.request, "urlopen", side_effect=missing),
        pytest.raises(_installer.InstallerError, match="has been published"),
    ):
        _installer.UrlTransport().read(_installer.STABLE_POINTER_URL)

    unavailable = _installer.urllib.error.HTTPError(
        _installer.STABLE_POINTER_URL, 503, "unavailable", {}, None
    )
    with (
        patch.object(_installer.urllib.request, "urlopen", side_effect=unavailable),
        pytest.raises(_installer.InstallerError, match="download unavailable"),
    ):
        _installer.UrlTransport().read(_installer.STABLE_POINTER_URL)


def test_manifest_schema_and_target_are_strict() -> None:
    sha = "a" * 64
    valid = json.dumps(
        {
            "version": "1.2.3",
            "artifacts": {
                "windows-x86_64": {"name": "proxy.exe", "sha256": sha, "size": 3}
            },
        }
    ).encode()
    assert _installer.parse_manifest(valid, "1.2.3")["windows-x86_64"]["sha256"] == sha
    invalid = json.dumps(
        {
            "version": "1.2.3",
            "artifacts": {
                "windows-arm64": {"name": "proxy.exe", "sha256": sha, "size": 3}
            },
        }
    ).encode()
    with pytest.raises(_installer.InstallerError, match="unsupported release target"):
        _installer.parse_manifest(invalid, "1.2.3")


def test_custom_root_must_be_absolute_and_relocates_bin(tmp_path: Path) -> None:
    with pytest.raises(_installer.InstallerError, match="absolute"):
        _installer.InstallPaths.resolve(Path("relative"), system="Windows")
    root = tmp_path / "root with spaces"
    paths = _installer.InstallPaths.resolve(root, system="Windows")
    assert paths.root == root
    assert paths.command_dir == root / "bin"


def test_default_roots_match_each_ruled_platform(tmp_path: Path) -> None:
    windows = _installer.InstallPaths.resolve(
        system="Windows", environ={"LOCALAPPDATA": str(tmp_path / "local")}
    )
    assert windows.root == tmp_path / "local" / "Forge"
    assert windows.command_dir == windows.root / "bin"

    linux = _installer.InstallPaths.resolve(
        system="Linux", environ={"XDG_DATA_HOME": str(tmp_path / "xdg")}, home=tmp_path
    )
    assert linux.root == tmp_path / "xdg" / "forge"
    assert linux.command_dir == tmp_path / ".local" / "bin"

    macos = _installer.InstallPaths.resolve(system="Darwin", environ={}, home=tmp_path)
    assert macos.root == tmp_path / "Library" / "Application Support" / "Forge"
    assert macos.command_dir == tmp_path / ".local" / "bin"


def test_fresh_install_writes_owned_layout_and_windows_argv_shim(
    tmp_path: Path,
) -> None:
    source, sha = artifact(tmp_path, "1.0.0")
    paths = windows_paths(tmp_path)
    runner = FakeRunner()
    path_file = adapter(tmp_path)
    state = install(source, sha, "1.0.0", paths, runner, path_file)

    slot = paths.slot("1.0.0")
    assert slot.read_bytes() == source.read_bytes()
    assert paths.command.read_text(encoding="utf-8") == (f'@echo off\n"{slot}" %*\n')
    assert state["current_version"] == "1.0.0"
    assert "selection" not in json.dumps(state)
    assert paths.uninstaller.is_file() and paths.marker.is_file()
    assert str(paths.command_dir) in (tmp_path / "user-path.txt").read_text()


def test_fresh_install_refuses_foreign_path_and_destination_commands(
    tmp_path: Path,
) -> None:
    windows = tmp_path / "windows"
    windows.mkdir()
    foreign_dir = windows / "Python" / "Scripts"
    foreign_dir.mkdir(parents=True)
    foreign_exe = foreign_dir / "forge-proxy.exe"
    foreign_exe.write_bytes(b"pip-owned launcher")
    source, sha = artifact(windows, "1.0.0")
    paths = windows_paths(windows)
    path_file = adapter(windows, "C:\\Existing")
    with pytest.raises(_installer.InstallerError, match="unowned forge-proxy") as exc:
        install(
            source,
            sha,
            "1.0.0",
            paths,
            FakeRunner(),
            path_file,
            environ={
                "PATH": str(foreign_dir),
                "PATHEXT": ".COM;.EXE;.BAT;.CMD",
            },
        )
    assert str(foreign_exe) in str(exc.value)
    assert "same Python environment" in str(exc.value)
    assert foreign_exe.read_bytes() == b"pip-owned launcher"
    assert not paths.root.exists()
    assert (windows / "user-path.txt").read_text() == "C:\\Existing"

    posix = tmp_path / "posix"
    posix.mkdir()
    source, sha = artifact(posix, "1.0.0")
    paths = _installer.InstallPaths(posix / "app", posix / "bin", "Linux")
    paths.command_dir.mkdir()
    paths.command.write_bytes(b"pip-owned script")
    paths.command.chmod(0o755)
    path_file = adapter(posix, "existing-path")
    with pytest.raises(_installer.InstallerError, match="unowned forge-proxy"):
        install(
            source,
            sha,
            "1.0.0",
            paths,
            FakeRunner(),
            path_file,
            environ={"PATH": str(paths.command_dir)},
        )
    assert paths.command.read_bytes() == b"pip-owned script"
    assert not paths.root.exists()
    assert (posix / "user-path.txt").read_text() == "existing-path"


def test_idempotent_install_rechecks_slot_without_rewriting(tmp_path: Path) -> None:
    source, sha = artifact(tmp_path, "1.0.0")
    paths = windows_paths(tmp_path)
    runner = FakeRunner()
    path_file = adapter(tmp_path)
    install(source, sha, "1.0.0", paths, runner, path_file)
    before = paths.slot("1.0.0").stat().st_mtime_ns
    install(source, sha, "1.0.0", paths, runner, path_file)
    assert paths.slot("1.0.0").stat().st_mtime_ns == before
    assert runner.calls[-2][0] == paths.slot("1.0.0")


def test_replaced_owned_command_blocks_reinstall_update_and_survives_uninstall(
    tmp_path: Path,
) -> None:
    paths = (
        windows_paths(tmp_path)
        if os.name == "nt"
        else _installer.InstallPaths(tmp_path / "app", tmp_path / "bin", "Linux")
    )
    runner = FakeRunner()
    path_file = adapter(tmp_path, "existing-path")
    source, sha = artifact(tmp_path, "1.0.0")
    install(source, sha, "1.0.0", paths, runner, path_file)

    paths.command.unlink()
    paths.command.write_bytes(b"replacement owned elsewhere")
    paths.command.chmod(0o755)
    before = paths.command.read_bytes()

    with pytest.raises(_installer.InstallerError, match="unowned forge-proxy"):
        install(source, sha, "1.0.0", paths, runner, path_file)
    with pytest.raises(_installer.InstallerError, match="unowned forge-proxy"):
        _installer.update(
            "1.0.0",
            paths=paths,
            path_adapter=path_file,
            environ={"PATH": str(paths.command_dir)},
        )
    assert paths.command.read_bytes() == before

    _installer.uninstall_owned(paths, path_adapter=path_file)
    assert paths.command.read_bytes() == before
    assert not paths.state.exists()
    assert not paths.marker.exists()
    assert not paths.uninstaller.exists()
    assert not paths.versions.exists()
    assert (tmp_path / "user-path.txt").read_text() == "existing-path"


def test_forward_updates_retain_current_and_one_previous_slot(tmp_path: Path) -> None:
    paths = windows_paths(tmp_path)
    runner = FakeRunner()
    path_file = adapter(tmp_path)
    for version in ("1.0.0", "1.1.0", "1.2.0"):
        source, sha = artifact(tmp_path, version)
        state = install(source, sha, version, paths, runner, path_file)
    assert state["current_version"] == "1.2.0"
    assert state["previous_versions"] == ["1.1.0"]
    assert {item.name for item in paths.versions.iterdir()} == {"1.1.0", "1.2.0"}


def test_external_exact_install_can_recover_to_lower_version(tmp_path: Path) -> None:
    paths = windows_paths(tmp_path)
    runner = FakeRunner()
    path_file = adapter(tmp_path)
    high, high_sha = artifact(tmp_path, "2.0.0")
    low, low_sha = artifact(tmp_path, "1.0.0")
    install(high, high_sha, "2.0.0", paths, runner, path_file)
    state = install(low, low_sha, "1.0.0", paths, runner, path_file)
    assert state["current_version"] == "1.0.0"
    assert state["previous_versions"] == ["2.0.0"]


def test_embedded_version_mismatch_never_publishes(tmp_path: Path) -> None:
    source, sha = artifact(tmp_path, "1.0.0")
    paths = windows_paths(tmp_path)
    with pytest.raises(_installer.InstallerError, match="version"):
        install(source, sha, "2.0.0", paths, FakeRunner(), adapter(tmp_path))
    assert not paths.command.exists()
    assert not paths.state.exists()
    assert not paths.slot("2.0.0").exists()


def test_checksum_failure_preserves_existing_command_state_and_bytes(
    tmp_path: Path,
) -> None:
    source, sha = artifact(tmp_path, "1.0.0")
    paths = windows_paths(tmp_path)
    runner = FakeRunner()
    path_file = adapter(tmp_path)
    install(source, sha, "1.0.0", paths, runner, path_file)
    before = (
        paths.command.read_bytes(),
        paths.state.read_bytes(),
        paths.slot("1.0.0").read_bytes(),
    )
    bad, _ = artifact(tmp_path, "1.1.0")
    with pytest.raises(_installer.InstallerError, match="checksum mismatch"):
        install(bad, "0" * 64, "1.1.0", paths, runner, path_file)
    assert before == (
        paths.command.read_bytes(),
        paths.state.read_bytes(),
        paths.slot("1.0.0").read_bytes(),
    )


def test_fresh_staged_check_failure_has_no_promoted_state(tmp_path: Path) -> None:
    source, sha = artifact(tmp_path, "1.0.0")
    runner = FakeRunner()
    runner.fail_payloads.add(source.read_bytes())
    paths = windows_paths(tmp_path)
    with pytest.raises(_installer.InstallerError, match="health failed"):
        install(source, sha, "1.0.0", paths, runner, adapter(tmp_path))
    assert not paths.command.exists() and not paths.state.exists()
    assert not paths.slot("1.0.0").exists()


def test_same_version_failed_staging_preserves_active_bytes(tmp_path: Path) -> None:
    active, active_sha = artifact(tmp_path, "1.0.0", "active")
    replacement, replacement_sha = artifact(tmp_path, "1.0.0", "replacement")
    paths = windows_paths(tmp_path)
    runner = FakeRunner()
    path_file = adapter(tmp_path)
    install(active, active_sha, "1.0.0", paths, runner, path_file)
    runner.fail_payloads.add(replacement.read_bytes())
    before = paths.slot("1.0.0").read_bytes()
    with pytest.raises(_installer.InstallerError, match="health failed"):
        install(replacement, replacement_sha, "1.0.0", paths, runner, path_file)
    assert paths.slot("1.0.0").read_bytes() == before


def test_profile_incompatibility_reports_and_promotes_without_rewriting_source(
    tmp_path: Path,
) -> None:
    source, sha = artifact(tmp_path, "1.0.0")
    before = source.read_bytes()
    output: list[str] = []
    runner = FakeRunner()
    runner.profile_incompatible = True
    paths = windows_paths(tmp_path)
    _installer.install_artifact(
        source,
        "1.0.0",
        sha,
        paths=paths,
        runner=runner,
        path_adapter=adapter(tmp_path),
        no_init=True,
        output=output.append,
    )
    assert paths.slot("1.0.0").read_bytes() == before == source.read_bytes()
    assert any("Incompatible managed profile" in line for line in output)
    assert any("do not block" in line for line in output)


def test_promotion_failure_rolls_back_command_state_and_owned_path(
    tmp_path: Path,
) -> None:
    source, sha = artifact(tmp_path, "1.0.0")
    paths = windows_paths(tmp_path)
    runner = FakeRunner()
    path_file = adapter(tmp_path)
    install(source, sha, "1.0.0", paths, runner, path_file)
    before = (
        paths.command.read_bytes(),
        paths.state.read_bytes(),
        (tmp_path / "user-path.txt").read_bytes(),
    )
    newer, newer_sha = artifact(tmp_path, "1.1.0")
    with (
        patch.object(
            _installer, "_publish_command", side_effect=OSError("publish failed")
        ),
        pytest.raises(OSError, match="publish failed"),
    ):
        install(newer, newer_sha, "1.1.0", paths, runner, path_file)
    assert before == (
        paths.command.read_bytes(),
        paths.state.read_bytes(),
        (tmp_path / "user-path.txt").read_bytes(),
    )


def release_fixture(
    version: str, source: Path, sha: str
) -> tuple[str, str, bytes, bytes]:
    manifest_url = f"manifest/{version}"
    asset_url = f"asset/{version}/{source.name}"
    manifest = json.dumps(
        {
            "version": version,
            "artifacts": {
                "windows-x86_64": {
                    "name": source.name,
                    "sha256": sha,
                    "size": source.stat().st_size,
                }
            },
        }
    ).encode()
    return manifest_url, asset_url, manifest, source.read_bytes()


def test_production_release_urls_use_raw_pointer_and_exact_forge_tag(
    tmp_path: Path,
) -> None:
    assert _installer.STABLE_POINTER_URL == (
        "https://raw.githubusercontent.com/antoinezambelli/forge/"
        "main/installer/proxy-stable.txt"
    )
    assert _installer.RELEASE_MANIFEST_URL == (
        "https://github.com/antoinezambelli/forge/releases/download/"
        "v{version}/proxy-{version}.json"
    )
    assert _installer.RELEASE_ASSET_URL == (
        "https://github.com/antoinezambelli/forge/releases/download/v{version}/{name}"
    )

    paths = windows_paths(tmp_path)
    runner = FakeRunner()
    path_file = adapter(tmp_path)
    one, one_sha = artifact(tmp_path, "1.0.0")
    install(one, one_sha, "1.0.0", paths, runner, path_file)
    two, two_sha = artifact(tmp_path, "1.1.0")
    _, _, manifest, payload = release_fixture("1.1.0", two, two_sha)
    manifest_url = _installer.RELEASE_MANIFEST_URL.format(version="1.1.0")
    asset_url = _installer.RELEASE_ASSET_URL.format(version="1.1.0", name=two.name)
    transport = FixtureTransport(
        {
            _installer.STABLE_POINTER_URL: b"1.1.0\n",
            manifest_url: manifest,
            asset_url: payload,
        }
    )

    _installer.update(
        paths=paths,
        transport=transport,
        runner=runner,
        path_adapter=path_file,
        output=lambda _line: None,
        target="windows-x86_64",
        environ={"PATH": ""},
    )

    assert transport.reads == [
        _installer.STABLE_POINTER_URL,
        manifest_url,
        asset_url,
    ]


def test_update_forward_same_newer_than_stable_and_lower_exact(tmp_path: Path) -> None:
    paths = windows_paths(tmp_path)
    runner = FakeRunner()
    path_file = adapter(tmp_path)
    one, one_sha = artifact(tmp_path, "1.0.0")
    install(one, one_sha, "1.0.0", paths, runner, path_file)
    two, two_sha = artifact(tmp_path, "1.1.0")
    manifest_url, asset_url, manifest, payload = release_fixture("1.1.0", two, two_sha)
    transport = FixtureTransport(
        {"pointer": b"1.1.0\n", manifest_url: manifest, asset_url: payload}
    )
    state = _installer.update(
        paths=paths,
        transport=transport,
        runner=runner,
        path_adapter=path_file,
        output=lambda _line: None,
        target="windows-x86_64",
        pointer_url="pointer",
        manifest_url="manifest/{version}",
        asset_url="asset/{version}/{name}",
        environ={"PATH": ""},
    )
    assert state is not None and state["current_version"] == "1.1.0"
    assert (
        _installer.update(
            "1.1.0",
            paths=paths,
            transport=transport,
            output=lambda _line: None,
            environ={"PATH": ""},
        )
        is None
    )
    old_stable = FixtureTransport({"pointer": b"1.0.0\n"})
    assert (
        _installer.update(
            paths=paths,
            transport=old_stable,
            output=lambda _line: None,
            pointer_url="pointer",
            environ={"PATH": ""},
        )
        is None
    )
    with pytest.raises(_installer.InstallerError, match="cannot downgrade"):
        _installer.update(
            "1.0.0", paths=paths, transport=transport, environ={"PATH": ""}
        )


def test_unavailable_update_preserves_current_install(tmp_path: Path) -> None:
    source, sha = artifact(tmp_path, "1.0.0")
    paths = windows_paths(tmp_path)
    runner = FakeRunner()
    path_file = adapter(tmp_path)
    install(source, sha, "1.0.0", paths, runner, path_file)
    before = (
        paths.command.read_bytes(),
        paths.state.read_bytes(),
        paths.slot("1.0.0").read_bytes(),
    )
    transport = FixtureTransport(
        {"pointer": _installer.InstallerError("download unavailable")}
    )
    with pytest.raises(_installer.InstallerError, match="download unavailable"):
        _installer.update(
            paths=paths,
            transport=transport,
            pointer_url="pointer",
            environ={"PATH": ""},
        )
    assert before == (
        paths.command.read_bytes(),
        paths.state.read_bytes(),
        paths.slot("1.0.0").read_bytes(),
    )


def test_windows_path_preexisting_remains_unowned_across_reinstall_and_uninstall(
    tmp_path: Path,
) -> None:
    paths = windows_paths(tmp_path)
    path_file = adapter(tmp_path, str(paths.command_dir))
    runner = FakeRunner()
    source, sha = artifact(tmp_path, "1.0.0")
    state = install(source, sha, "1.0.0", paths, runner, path_file)
    assert state["path_integration"]["added"] is False  # type: ignore[index]
    install(source, sha, "1.0.0", paths, runner, path_file)
    _installer.uninstall_owned(paths, path_adapter=path_file)
    assert (tmp_path / "user-path.txt").read_text() == str(paths.command_dir)


def test_windows_registry_path_write_broadcasts_environment_change() -> None:
    registry_key = MagicMock()
    winreg = MagicMock()
    winreg.HKEY_CURRENT_USER = object()
    winreg.REG_EXPAND_SZ = object()
    winreg.CreateKey.return_value = registry_key
    with (
        patch.dict(sys.modules, {"winreg": winreg}),
        patch.object(_installer, "_broadcast_windows_environment_change") as broadcast,
    ):
        _installer.WindowsPathAdapter()._write("C:\\Forge")

    winreg.SetValueEx.assert_called_once()
    broadcast.assert_called_once_with()


def test_windows_environment_broadcast_uses_setting_change_message() -> None:
    send = MagicMock()
    windll = MagicMock()
    windll.user32.SendMessageTimeoutW = send
    with patch.object(_installer.ctypes, "windll", windll, create=True):
        _installer._broadcast_windows_environment_change()

    args = send.call_args.args
    assert args[:6] == (0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000)
    assert args[6] is not None
    assert send.argtypes[3] is _installer.wintypes.LPCWSTR
    assert send.restype is _installer.wintypes.LPARAM


def test_generated_windows_uninstaller_broadcasts_only_for_real_user_path(
    tmp_path: Path,
) -> None:
    paths = windows_paths(tmp_path)
    real_record = {
        "kind": "windows",
        "command_dir": str(paths.command_dir),
        "added": True,
        "representation": None,
    }
    real_script = _installer._render_windows_uninstaller(
        paths, "owned", real_record, paths.slot("1.0.0")
    ).decode("utf-8")
    assert "Get-FileHash" not in real_script
    assert "[System.Security.Cryptography.SHA256]::Create()" in real_script
    assert "if($commandStatus -eq 'owned')" in real_script
    assert "SetEnvironmentVariable" in real_script
    assert "SendMessageTimeout" in real_script
    assert "'Environment'" in real_script

    fixture_record = {
        **real_record,
        "representation": str(tmp_path / "path.txt"),
    }
    fixture_script = _installer._render_windows_uninstaller(
        paths, "owned", fixture_record, paths.slot("1.0.0")
    ).decode("utf-8")
    assert "SetEnvironmentVariable" not in fixture_script
    assert "SendMessageTimeout" not in fixture_script


def test_owned_uninstall_removes_only_owned_files_and_preserves_unowned(
    tmp_path: Path,
) -> None:
    paths = windows_paths(tmp_path, "root with spaces")
    path_file = adapter(tmp_path, "C:\\Existing")
    source, sha = artifact(tmp_path, "1.0.0")
    install(source, sha, "1.0.0", paths, FakeRunner(), path_file)
    unowned = paths.root / "keep.txt"
    unowned.write_text("keep", encoding="utf-8")
    external = tmp_path / "profiles" / "default.toml"
    external.parent.mkdir()
    external.write_text("profile", encoding="utf-8")
    _installer.uninstall_owned(paths, path_adapter=path_file)
    assert unowned.read_text() == "keep"
    assert external.read_text() == "profile"
    assert (tmp_path / "user-path.txt").read_text() == "C:\\Existing"
    assert not paths.command.exists() and not paths.state.exists()


def test_posix_symlink_and_marked_startup_are_owned_and_removed(tmp_path: Path) -> None:
    root = tmp_path / "app"
    paths = _installer.InstallPaths(root, tmp_path / "bin with spaces", "Linux")
    startup_home = tmp_path / "home"
    startup_home.mkdir()
    startup = startup_home / ".bashrc"
    startup.write_text("# existing\n", encoding="utf-8")
    output: list[str] = []
    path_adapter = _installer.PosixPathAdapter(
        shell="/bin/bash", home=startup_home, output=output.append
    )
    record = path_adapter.ensure(paths.command_dir, None)
    assert path_adapter.ensure(paths.command_dir, record) == record
    assert startup.read_text().count(_installer._POSIX_START) == 1
    assert output == [
        f"Updated PATH startup file: {startup.resolve()}",
        "Undo with 'forge-proxy uninstall' or remove the block from "
        f"'{_installer._POSIX_START}' through '{_installer._POSIX_END}' "
        f"in {startup.resolve()}",
    ]
    path_adapter.remove(record)
    assert startup.read_text() == "# existing\n"

    slot = paths.slot("1.0.0")
    with (
        patch.object(_installer.os, "symlink") as make_symlink,
        patch.object(_installer.os, "replace") as replace,
    ):
        _installer._publish_command(paths, slot)
    linked_target = make_symlink.call_args.args[0]
    assert linked_target == os.path.relpath(slot, paths.command_dir)
    assert not Path(linked_target).is_absolute()
    assert replace.call_args.args[1] == paths.command
    uninstaller = _installer._render_posix_uninstaller(
        paths, "owned", record, slot
    ).decode("utf-8")
    assert 'readlink "$command"' in uninstaller
    assert '[ "$owned_command" -eq 1 ] && rm -f -- "$command"' in uninstaller


def test_preexisting_posix_path_block_is_not_claimed_or_reported(
    tmp_path: Path,
) -> None:
    startup = tmp_path / ".zshrc"
    command_dir = tmp_path / "bin"
    startup.write_text(_installer.PosixPathAdapter.block(command_dir), encoding="utf-8")
    output: list[str] = []
    adapter = _installer.PosixPathAdapter(
        shell="/bin/zsh", home=tmp_path, output=output.append
    )

    record = adapter.ensure(command_dir, None)

    assert record["added"] is False
    assert output == []


def test_unknown_posix_shell_prints_exact_export_without_editing(
    tmp_path: Path,
) -> None:
    output: list[str] = []
    adapter = _installer.PosixPathAdapter(
        shell="/bin/fish", home=tmp_path, output=output.append
    )
    command_dir = tmp_path / "bin with spaces"
    record = adapter.ensure(command_dir, None)
    assert record["kind"] == "guidance"
    assert output == [f'export PATH={shlex.quote(str(command_dir))}:"$PATH"']
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("no_init", [False, True])
def test_install_never_onboards_and_prints_post_install_commands(
    tmp_path: Path,
    no_init: bool,
) -> None:
    paths = windows_paths(tmp_path)
    source, sha = artifact(tmp_path, "1.0.0")
    runner = FakeRunner()
    path_file = adapter(tmp_path)
    output: list[str] = []
    _installer.install_artifact(
        source,
        "1.0.0",
        sha,
        paths=paths,
        runner=runner,
        path_adapter=path_file,
        no_init=no_init,
        output=output.append,
    )
    assert not [args for _, args in runner.calls if args[0] in {"init", "check"}]
    assert output[-7:] == [
        "Installed forge-proxy 1.0.0 at " + str(paths.slot("1.0.0")),
        "Next, configure and verify the installation:",
        "  forge-proxy init",
        "  forge-proxy check",
        "For noninteractive unmanaged setup:",
        "  forge-proxy init --non-interactive --backend-url URL",
        "  forge-proxy check",
    ]
