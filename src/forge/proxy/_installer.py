"""Artifact-owned installation lifecycle for the standalone Forge Proxy."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


PRODUCT = "forge-proxy"
STATE_SCHEMA = 1
SUPPORTED_TARGETS = (
    "windows-x86_64",
    "linux-x86_64-gnu",
    "macos-arm64",
)
STABLE_POINTER_URL = (
    "https://raw.githubusercontent.com/antoinezambelli/forge/"
    "main/installer/proxy-stable.txt"
)
RELEASE_MANIFEST_URL = (
    "https://github.com/antoinezambelli/forge/releases/download/"
    "v{version}/proxy-{version}.json"
)
RELEASE_ASSET_URL = (
    "https://github.com/antoinezambelli/forge/releases/download/v{version}/{name}"
)
_VERSION_PART = r"(?:0|[1-9][0-9]*)"
_VERSION = re.compile(rf"{_VERSION_PART}\.{_VERSION_PART}\.{_VERSION_PART}")
_CHECKSUM = re.compile(r"[0-9a-fA-F]{64}")
_POSIX_START = "# >>> forge-proxy PATH >>>"
_POSIX_END = "# <<< forge-proxy PATH <<<"


class InstallerError(RuntimeError):
    """A supported lifecycle operation could not be completed."""


@dataclass(frozen=True)
class InstallPaths:
    root: Path
    command_dir: Path
    system: str

    @classmethod
    def resolve(
        cls,
        install_root: Path | None = None,
        *,
        system: str | None = None,
        environ: Mapping[str, str] | None = None,
        home: Path | None = None,
    ) -> "InstallPaths":
        system = system or platform.system()
        environ = os.environ if environ is None else environ
        home = Path.home() if home is None else home
        if install_root is not None:
            root = install_root.expanduser()
            if not root.is_absolute():
                raise InstallerError("--install-root must be an absolute path")
            root = root.resolve()
            return cls(root, root / "bin", system)
        if system == "Windows":
            forge_root = Path(environ["LOCALAPPDATA"]) / "Forge"
            return cls(forge_root, forge_root / "bin", system)
        if system == "Darwin":
            forge_root = home / "Library" / "Application Support" / "Forge"
            return cls(forge_root, home / ".local" / "bin", system)
        data_root = Path(environ.get("XDG_DATA_HOME", home / ".local" / "share"))
        return cls(data_root / "forge", home / ".local" / "bin", system)

    @classmethod
    def from_installed_artifact(cls, artifact: Path) -> "InstallPaths | None":
        artifact = artifact.resolve()
        # <root>/versions/<version>/forge-proxy[.exe]
        if artifact.parent.parent.name != "versions":
            return None
        root = artifact.parents[2]
        state_path = root / "state.json"
        if not state_path.is_file():
            return None
        state = read_state(state_path)
        return cls(root, Path(state["command_dir"]), str(state["system"]))

    @property
    def versions(self) -> Path:
        return self.root / "versions"

    @property
    def staging(self) -> Path:
        return self.root / ".staging"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def marker(self) -> Path:
        return self.root / "ownership.txt"

    @property
    def executable_name(self) -> str:
        return "forge-proxy.exe" if self.system == "Windows" else "forge-proxy"

    @property
    def command(self) -> Path:
        suffix = ".cmd" if self.system == "Windows" else ""
        return self.command_dir / f"forge-proxy{suffix}"

    @property
    def uninstaller(self) -> Path:
        suffix = ".cmd" if self.system == "Windows" else ".sh"
        return self.root / f"uninstall{suffix}"

    def slot(self, version: str) -> Path:
        return self.versions / version / self.executable_name


class Transport(Protocol):
    def read(self, url: str) -> bytes: ...


class UrlTransport:
    def read(self, url: str) -> bytes:
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and url == STABLE_POINTER_URL:
                raise InstallerError(
                    "no stable standalone Proxy release has been published"
                ) from exc
            raise InstallerError(f"download unavailable: {url}") from exc
        except (OSError, urllib.error.URLError) as exc:
            raise InstallerError(f"download unavailable: {url}") from exc


class ProcessRunner(Protocol):
    def run(
        self, executable: Path, arguments: list[str]
    ) -> subprocess.CompletedProcess[str]: ...


class SubprocessRunner:
    def run(
        self, executable: Path, arguments: list[str]
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(executable), *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )


class PathAdapter(Protocol):
    def ensure(
        self, command_dir: Path, previous: dict[str, Any] | None
    ) -> dict[str, Any]: ...
    def remove(self, record: Mapping[str, Any]) -> None: ...


class WindowsPathAdapter:
    """User PATH adapter; a text-file representation is the local test seam."""

    def __init__(self, representation: Path | None = None) -> None:
        self.representation = representation

    def _read(self) -> str:
        if self.representation is not None:
            if not self.representation.exists():
                return ""
            return self.representation.read_text(encoding="utf-8")
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                return str(winreg.QueryValueEx(key, "Path")[0])
        except FileNotFoundError:
            return ""

    def _write(self, value: str) -> None:
        if self.representation is not None:
            _atomic_write(self.representation, value.encode("utf-8"))
            return
        import winreg

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, value)
        _broadcast_windows_environment_change()

    def ensure(
        self, command_dir: Path, previous: dict[str, Any] | None
    ) -> dict[str, Any]:
        if previous is not None:
            return previous
        exact = str(command_dir)
        entries = [item for item in self._read().split(";") if item]
        added = exact not in entries
        if added:
            entries.append(exact)
            self._write(";".join(entries))
        return {
            "kind": "windows",
            "command_dir": exact,
            "added": added,
            "representation": (
                str(self.representation.resolve()) if self.representation else None
            ),
        }

    def remove(self, record: Mapping[str, Any]) -> None:
        if not record.get("added"):
            return
        exact = str(record["command_dir"])
        entries = [item for item in self._read().split(";") if item]
        self._write(";".join(item for item in entries if item != exact))


class PosixPathAdapter:
    def __init__(
        self,
        *,
        shell: str | None = None,
        home: Path | None = None,
        output: Callable[[str], None] = print,
    ) -> None:
        self.shell = shell if shell is not None else os.environ.get("SHELL", "")
        self.home = Path.home() if home is None else home
        self.output = output

    @staticmethod
    def block(command_dir: Path) -> str:
        return (
            f"{_POSIX_START}\n"
            f'export PATH={shlex.quote(str(command_dir))}:"$PATH"\n'
            f"{_POSIX_END}\n"
        )

    def ensure(
        self, command_dir: Path, previous: dict[str, Any] | None
    ) -> dict[str, Any]:
        if previous is not None:
            return previous
        shell_name = Path(self.shell).name
        if shell_name not in {"bash", "zsh"}:
            self.output(f'export PATH={shlex.quote(str(command_dir))}:"$PATH"')
            return {"kind": "guidance", "command_dir": str(command_dir)}
        startup = self.home / (".bashrc" if shell_name == "bash" else ".zshrc")
        block = self.block(command_dir)
        content = startup.read_text(encoding="utf-8") if startup.exists() else ""
        added = block not in content
        if added:
            if content and not content.endswith("\n"):
                content += "\n"
            _atomic_write(startup, (content + block).encode("utf-8"))
            self.output(f"Updated PATH startup file: {startup.resolve()}")
            self.output(
                "Undo with 'forge-proxy uninstall' or remove the block from "
                f"'{_POSIX_START}' through '{_POSIX_END}' in {startup.resolve()}"
            )
        return {
            "kind": "posix",
            "command_dir": str(command_dir),
            "startup_file": str(startup.resolve()),
            "block": block,
            "added": added,
        }

    def remove(self, record: Mapping[str, Any]) -> None:
        if record.get("kind") != "posix" or not record.get("added"):
            return
        startup = Path(str(record["startup_file"]))
        if startup.exists():
            content = startup.read_text(encoding="utf-8")
            _atomic_write(
                startup, content.replace(str(record["block"]), "").encode("utf-8")
            )


def _broadcast_windows_environment_change() -> None:
    send = ctypes.windll.user32.SendMessageTimeoutW  # type: ignore[attr-defined]
    send.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPCWSTR,
        wintypes.UINT,
        wintypes.UINT,
        ctypes.POINTER(wintypes.WPARAM),
    ]
    send.restype = wintypes.LPARAM
    result = wintypes.WPARAM()
    send(
        0xFFFF,
        0x001A,
        0,
        "Environment",
        0x0002,
        5000,
        ctypes.byref(result),
    )


def default_path_adapter(
    paths: InstallPaths, *, output: Callable[[str], None] = print
) -> PathAdapter:
    if paths.system == "Windows":
        representation = os.environ.get("FORGE_PROXY_PATH_FILE")
        return WindowsPathAdapter(Path(representation) if representation else None)
    return PosixPathAdapter(output=output)


def parse_version(value: str) -> tuple[int, int, int]:
    if _VERSION.fullmatch(value) is None:
        raise InstallerError(f"invalid Proxy version: {value!r}; expected X.Y.Z")
    return tuple(int(part) for part in value.split("."))  # type: ignore[return-value]


def parse_checksum(value: str) -> str:
    if _CHECKSUM.fullmatch(value) is None:
        raise InstallerError("SHA-256 must contain exactly 64 hexadecimal characters")
    return value.lower()


def parse_pointer(payload: bytes) -> str:
    try:
        value = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise InstallerError("stable pointer is not ASCII") from exc
    if value.endswith("\n"):
        value = value[:-1]
    if "\n" in value or "\r" in value:
        raise InstallerError("stable pointer must contain one bare X.Y.Z line")
    parse_version(value)
    return value


def native_target(*, system: str | None = None, machine: str | None = None) -> str:
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    key = (system, machine)
    targets = {
        ("Windows", "amd64"): "windows-x86_64",
        ("Windows", "x86_64"): "windows-x86_64",
        ("Linux", "amd64"): "linux-x86_64-gnu",
        ("Linux", "x86_64"): "linux-x86_64-gnu",
        ("Darwin", "arm64"): "macos-arm64",
        ("Darwin", "aarch64"): "macos-arm64",
    }
    try:
        return targets[key]
    except KeyError as exc:
        raise InstallerError(
            f"unsupported standalone target: {system} {machine}"
        ) from exc


def parse_manifest(payload: bytes, expected_version: str) -> dict[str, dict[str, Any]]:
    parse_version(expected_version)
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InstallerError("release manifest is not valid JSON") from exc
    if not isinstance(document, dict) or set(document) != {"version", "artifacts"}:
        raise InstallerError("release manifest must contain only version and artifacts")
    if document["version"] != expected_version or not isinstance(
        document["artifacts"], dict
    ):
        raise InstallerError(
            "release manifest version does not match the requested version"
        )
    artifacts: dict[str, dict[str, Any]] = {}
    for target, entry in document["artifacts"].items():
        if target not in SUPPORTED_TARGETS:
            raise InstallerError(f"unsupported release target: {target}")
        if not isinstance(entry, dict) or set(entry) != {"name", "sha256", "size"}:
            raise InstallerError(f"invalid release manifest entry for {target}")
        if (
            not isinstance(entry["name"], str)
            or not entry["name"]
            or Path(entry["name"]).name != entry["name"]
            or not isinstance(entry["size"], int)
            or isinstance(entry["size"], bool)
            or entry["size"] < 0
            or not isinstance(entry["sha256"], str)
        ):
            raise InstallerError(f"invalid release manifest entry for {target}")
        artifacts[target] = {
            "name": entry["name"],
            "sha256": parse_checksum(entry["sha256"]),
            "size": entry["size"],
        }
    if not artifacts:
        raise InstallerError("release manifest contains no artifacts")
    return artifacts


def _atomic_write(path: Path, content: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        if executable and os.name != "nt":
            temporary.chmod(0o755)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_state(path: Path) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallerError(f"cannot read installed state: {path}") from exc
    required = {
        "schema",
        "product",
        "ownership_id",
        "root",
        "command_dir",
        "system",
        "current_version",
        "previous_versions",
        "verified_slots",
        "path_integration",
    }
    if not isinstance(state, dict) or set(state) != required:
        raise InstallerError("installed state has an unsupported schema")
    if state["schema"] != STATE_SCHEMA or state["product"] != PRODUCT:
        raise InstallerError("installed state is not owned by Forge Proxy")
    return state


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    _atomic_write(
        path, (json.dumps(state, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def current_artifact() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve()
    return Path(sys.argv[0]).resolve()


def discover_paths(artifact: Path | None = None) -> InstallPaths:
    artifact = current_artifact() if artifact is None else artifact
    installed = InstallPaths.from_installed_artifact(artifact)
    return installed or InstallPaths.resolve()


def _checked_run(
    runner: ProcessRunner, executable: Path, arguments: list[str], label: str
) -> subprocess.CompletedProcess[str]:
    result = runner.run(executable, arguments)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no details").strip()
        raise InstallerError(f"staged {label} failed: {detail}")
    return result


def _verify_executable(
    executable: Path,
    version: str,
    runner: ProcessRunner,
    output: Callable[[str], None],
) -> None:
    _checked_run(
        runner,
        executable,
        ["_installer-self-check", "--expected-version", version],
        "runtime/health/version check",
    )
    profiles = runner.run(executable, ["_installer-profile-check"])
    for line in (profiles.stdout + profiles.stderr).splitlines():
        output(line)
    if profiles.returncode != 0:
        output("Managed profile incompatibilities do not block this update.")


def _snapshot(path: Path) -> tuple[str, bytes | str | None]:
    if path.is_symlink():
        return ("symlink", os.readlink(path))
    if path.is_file():
        return ("file", path.read_bytes())
    return ("missing", None)


def _windows_command_content(slot: Path) -> bytes:
    return f'@echo off\r\n"{slot}" %*\r\n'.encode("utf-8")


def _posix_command_target(paths: InstallPaths, slot: Path) -> str:
    return os.path.relpath(slot, paths.command_dir)


def _command_is_owned(paths: InstallPaths, state: Mapping[str, Any]) -> bool:
    command = paths.command
    slot = paths.slot(str(state["current_version"]))
    try:
        if paths.system == "Windows":
            return (
                command.is_file()
                and not command.is_symlink()
                and command.read_bytes() == _windows_command_content(slot)
            )
        return command.is_symlink() and os.readlink(command) == _posix_command_target(
            paths, slot
        )
    except OSError:
        return False


def _command_key(path: Path, system: str) -> str:
    value = os.path.abspath(path)
    return os.path.normcase(value) if system == "Windows" else value


def _path_command_names(system: str, environ: Mapping[str, str]) -> tuple[str, ...]:
    if system != "Windows":
        return (PRODUCT,)
    raw_extensions = environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    extensions: list[str] = [""]
    for item in raw_extensions.split(";"):
        extension = item.strip().lower()
        if extension and not extension.startswith("."):
            extension = f".{extension}"
        if extension not in extensions:
            extensions.append(extension)
    return tuple(f"{PRODUCT}{extension}" for extension in extensions)


def _command_conflicts(
    paths: InstallPaths,
    prior: Mapping[str, Any] | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> list[Path]:
    environ = os.environ if environ is None else environ
    candidates: list[Path] = []
    if paths.command.exists() or paths.command.is_symlink():
        candidates.append(paths.command)

    separator = ";" if paths.system == "Windows" else ":"
    for raw_directory in environ.get("PATH", "").split(separator):
        directory = raw_directory.strip().strip('"')
        if not directory:
            continue
        for name in _path_command_names(paths.system, environ):
            candidate = Path(directory) / name
            if not candidate.is_file():
                continue
            if paths.system != "Windows" and not os.access(candidate, os.X_OK):
                continue
            candidates.append(candidate)

    owned_key = _command_key(paths.command, paths.system)
    owned = prior is not None and _command_is_owned(paths, prior)
    conflicts: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _command_key(candidate, paths.system)
        if key in seen:
            continue
        seen.add(key)
        if key == owned_key and owned:
            continue
        conflicts.append(Path(os.path.abspath(candidate)))
    return conflicts


def _refuse_command_conflicts(
    paths: InstallPaths,
    prior: Mapping[str, Any] | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    conflicts = _command_conflicts(paths, prior, environ=environ)
    if not conflicts:
        return
    locations = ", ".join(f"'{path}'" for path in conflicts)
    raise InstallerError(
        f"unowned forge-proxy command already exists at {locations}; the "
        "standalone installer changed nothing and will not overwrite or compete "
        "with it. If an older forge-guardrails package created the command, "
        "upgrade that package in the same Python environment so pip removes its "
        "launcher, then retry. Otherwise remove the command through the tool "
        "that owns it, then retry."
    )


def _restore(path: Path, snapshot: tuple[str, bytes | str | None]) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()
    kind, value = snapshot
    if kind == "file":
        _atomic_write(path, value if isinstance(value, bytes) else b"")
    elif kind == "symlink":
        path.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(value), path)


def _publish_command(paths: InstallPaths, slot: Path) -> None:
    paths.command_dir.mkdir(parents=True, exist_ok=True)
    if paths.system == "Windows":
        _atomic_write(paths.command, _windows_command_content(slot))
        return
    temporary = paths.command_dir / f".forge-proxy.{uuid.uuid4().hex}"
    os.symlink(_posix_command_target(paths, slot), temporary)
    try:
        os.replace(temporary, paths.command)
    finally:
        if temporary.is_symlink():
            temporary.unlink()


def _marker_content(paths: InstallPaths, ownership_id: str) -> str:
    return (
        f"product={PRODUCT}|ownership_id={ownership_id}|root={paths.root}|"
        f"command={paths.command}"
    )


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _render_windows_uninstaller(
    paths: InstallPaths,
    ownership_id: str,
    path_record: Mapping[str, Any],
    slot: Path,
) -> bytes:
    marker = _marker_content(paths, ownership_id)
    command_sha256 = hashlib.sha256(_windows_command_content(slot)).hexdigest()
    ps = [
        "$ErrorActionPreference='SilentlyContinue'",
        "$parent=[int]%1",
        "while(Get-Process -Id $parent -ErrorAction SilentlyContinue){Start-Sleep -Milliseconds 100}",
        "Start-Sleep -Milliseconds 250",
        f"$marker={_ps_quote(str(paths.marker))}",
        f"if((Get-Content -Raw -LiteralPath $marker) -ne {_ps_quote(marker)}){{exit 2}}",
        "$commandStatus='missing'",
        "$commandHash=''",
        f"$expectedCommandHash='{command_sha256}'",
        f"$command={_ps_quote(str(paths.command))}",
        "if(Test-Path -LiteralPath $command){"
        "$commandStatus='unreadable';$attempt=0;"
        "while($commandStatus -eq 'unreadable' -and $attempt -lt 50){"
        "try{$bytes=[System.IO.File]::ReadAllBytes($command);"
        "$sha=[System.Security.Cryptography.SHA256]::Create();"
        "try{$commandHash=[System.BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-','').ToLowerInvariant()}"
        "finally{$sha.Dispose()};"
        "if($commandHash -eq $expectedCommandHash){$commandStatus='owned'}else{$commandStatus='foreign'}"
        "}catch{};$attempt++;"
        "if($commandStatus -eq 'unreadable'){Start-Sleep -Milliseconds 100}};"
        "if($commandStatus -eq 'unreadable'){Write-Output ('Locked remnant: '+$command);exit 1}}",
    ]
    targets = ",".join(
        _ps_quote(str(target)) for target in (paths.versions, paths.staging)
    )
    ps.append(
        f"$locked=$false;foreach($target in @({targets})){{$attempt=0;"
        "while((Test-Path -LiteralPath $target)-and $attempt -lt 50){"
        "Remove-Item -Recurse -Force -LiteralPath $target;$attempt++;"
        "if(Test-Path -LiteralPath $target){Start-Sleep -Milliseconds 100}};"
        "if(Test-Path -LiteralPath $target){$locked=$true;"
        "Write-Output ('Locked remnant: '+$target)}};if($locked){exit 1}"
    )
    ps.append(
        "if($commandStatus -eq 'owned'){$attempt=0;"
        "while((Test-Path -LiteralPath $command)-and $attempt -lt 50){"
        "try{Remove-Item -Force -LiteralPath $command -ErrorAction Stop}catch{};"
        "$attempt++;if(Test-Path -LiteralPath $command){Start-Sleep -Milliseconds 100}};"
        "if(Test-Path -LiteralPath $command){Write-Output ('Locked remnant: '+$command);exit 1}}"
    )
    if path_record.get("kind") == "windows" and path_record.get("added"):
        command_dir = _ps_quote(str(path_record["command_dir"]))
        representation = path_record.get("representation")
        if representation:
            rep = _ps_quote(str(representation))
            ps.extend(
                [
                    f"$p={rep}",
                    "$v=if(Test-Path -LiteralPath $p){Get-Content -Raw -LiteralPath $p}else{''}",
                    f"$v=(($v -split ';')|Where-Object{{$_ -and $_ -ne {command_dir}}}) -join ';'",
                    "Set-Content -NoNewline -LiteralPath $p -Value $v",
                ]
            )
        else:
            ps.extend(
                [
                    "$p=[Environment]::GetEnvironmentVariable('Path','User')",
                    f"$p=(($p -split ';')|Where-Object{{$_ -and $_ -ne {command_dir}}}) -join ';'",
                    "[Environment]::SetEnvironmentVariable('Path',$p,'User')",
                    "Add-Type -TypeDefinition 'using System;using System.Runtime.InteropServices;public static class ForgeEnvironment{[DllImport(\"user32.dll\",CharSet=CharSet.Unicode)]public static extern IntPtr SendMessageTimeout(IntPtr hWnd,uint msg,UIntPtr wParam,string lParam,uint flags,uint timeout,out UIntPtr result);}'",
                    "$broadcast=[UIntPtr]::Zero",
                    "[void][ForgeEnvironment]::SendMessageTimeout([IntPtr]0xffff,0x001A,[UIntPtr]::Zero,'Environment',0x0002,5000,[ref]$broadcast)",
                ]
            )
    for target in (paths.state, paths.marker):
        ps.append(f"Remove-Item -Force -LiteralPath {_ps_quote(str(target))}")
    ps.extend(
        [
            f"Remove-Item -Force -LiteralPath {_ps_quote(str(paths.uninstaller))}",
            f"Remove-Item -Force -LiteralPath {_ps_quote(str(paths.command_dir))}",
            f"Remove-Item -Force -LiteralPath {_ps_quote(str(paths.root))}",
            f"if(Test-Path -LiteralPath {_ps_quote(str(paths.root))})"
            f"{{Write-Output 'Locked remnant: {str(paths.root)}'}}",
        ]
    )
    command = ";".join(ps).replace('"', '\\"')
    return (
        f'@echo off\r\nstart "" /b powershell.exe -NoProfile -Command "{command}" '
        "& exit /b\r\n"
    ).encode("utf-8")


def _render_posix_uninstaller(
    paths: InstallPaths,
    ownership_id: str,
    path_record: Mapping[str, Any],
    slot: Path,
) -> bytes:
    q = shlex.quote
    lines = [
        "#!/bin/sh",
        "parent=$1",
        'while kill -0 "$parent" 2>/dev/null; do sleep 0.1; done',
        f"marker={q(str(paths.marker))}",
        f"expected={q(_marker_content(paths, ownership_id))}",
        '[ "$(cat "$marker" 2>/dev/null)" = "$expected" ] || exit 2',
        f"command={q(str(paths.command))}",
        f"expected_command={q(_posix_command_target(paths, slot))}",
        "owned_command=0",
        '[ -L "$command" ] && [ "$(readlink "$command")" = "$expected_command" ] && owned_command=1',
    ]
    if path_record.get("kind") == "posix" and path_record.get("added"):
        startup = q(str(path_record["startup_file"]))
        lines.extend(
            [
                f"startup={startup}",
                'if [ -f "$startup" ]; then',
                '  tmp="$startup.forge-proxy.$$"',
                f'  awk \'BEGIN{{skip=0}} $0=="{_POSIX_START}"{{skip=1;next}} '
                f'$0=="{_POSIX_END}"{{skip=0;next}} !skip{{print}}\' "$startup" > "$tmp"',
                '  mv "$tmp" "$startup"',
                "fi",
            ]
        )
    lines.extend(
        [
            '[ "$owned_command" -eq 1 ] && rm -f -- "$command"',
            f"rm -f -- {q(str(paths.state))} {q(str(paths.marker))}",
            f"rm -rf -- {q(str(paths.versions))} {q(str(paths.staging))}",
            f"rm -f -- {q(str(paths.uninstaller))}",
            f"rmdir -- {q(str(paths.command_dir))} 2>/dev/null || true",
            f"rmdir -- {q(str(paths.root))} 2>/dev/null || true",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _render_ownership_files(
    paths: InstallPaths,
    ownership_id: str,
    path_record: Mapping[str, Any],
    slot: Path,
) -> None:
    _atomic_write(paths.marker, _marker_content(paths, ownership_id).encode("utf-8"))
    content = (
        _render_windows_uninstaller(paths, ownership_id, path_record, slot)
        if paths.system == "Windows"
        else _render_posix_uninstaller(paths, ownership_id, path_record, slot)
    )
    _atomic_write(paths.uninstaller, content, executable=paths.system != "Windows")


def _previous_path_record(state: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if state is None else dict(state["path_integration"])


def install_artifact(
    artifact: Path,
    version: str,
    sha256: str,
    *,
    install_root: Path | None = None,
    no_init: bool = False,
    paths: InstallPaths | None = None,
    runner: ProcessRunner | None = None,
    path_adapter: PathAdapter | None = None,
    output: Callable[[str], None] = print,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    parse_version(version)
    sha256 = parse_checksum(sha256)
    artifact = artifact.resolve()
    paths = paths or InstallPaths.resolve(install_root)
    runner = runner or SubprocessRunner()
    path_adapter = path_adapter or default_path_adapter(paths, output=output)
    prior = read_state(paths.state) if paths.state.is_file() else None
    if prior is not None and Path(prior["root"]) != paths.root:
        raise InstallerError("installed state belongs to a different root")
    _refuse_command_conflicts(paths, prior, environ=environ)

    paths.staging.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if paths.system == "Windows" else ""
    staged = paths.staging / f"{uuid.uuid4().hex}{suffix}"
    shutil.copyfile(artifact, staged)
    if paths.system != "Windows":
        staged.chmod(staged.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    try:
        if _sha256(staged) != sha256:
            raise InstallerError("downloaded artifact checksum mismatch")
        slot = paths.slot(version)
        reusable = slot.is_file() and _sha256(slot) == sha256
        checked = slot if reusable else staged
        _verify_executable(checked, version, runner, output)
        if not reusable:
            slot.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, slot)
            if paths.system != "Windows":
                slot.chmod(0o755)

        old_current = prior["current_version"] if prior else None
        previous = list(prior["previous_versions"]) if prior else []
        if old_current and old_current != version:
            previous = [
                old_current,
                *[item for item in previous if item != old_current],
            ]
        previous = [item for item in previous if item != version][:1]
        ownership_id = prior["ownership_id"] if prior else uuid.uuid4().hex

        watched = [paths.command, paths.state, paths.marker, paths.uninstaller]
        snapshots = {path: _snapshot(path) for path in watched}
        path_record: dict[str, Any] | None = None
        try:
            path_record = path_adapter.ensure(
                paths.command_dir, _previous_path_record(prior)
            )
            verified_by_version = {
                item["version"]: item
                for item in (prior["verified_slots"] if prior else [])
            }
            verified_by_version[version] = {"version": version, "sha256": sha256}
            retained = [version, *previous]
            state = {
                "schema": STATE_SCHEMA,
                "product": PRODUCT,
                "ownership_id": ownership_id,
                "root": str(paths.root),
                "command_dir": str(paths.command_dir),
                "system": paths.system,
                "current_version": version,
                "previous_versions": previous,
                "verified_slots": [verified_by_version[item] for item in retained],
                "path_integration": path_record,
            }
            _render_ownership_files(paths, ownership_id, path_record, slot)
            _write_state(paths.state, state)
            _publish_command(paths, slot)
        except Exception:
            if path_record is not None and (
                prior is None or path_record != prior["path_integration"]
            ):
                path_adapter.remove(path_record)
            for path, snapshot in snapshots.items():
                _restore(path, snapshot)
            raise

        retained_versions = {version, *previous}
        if paths.versions.is_dir():
            for directory in paths.versions.iterdir():
                if directory.is_dir() and directory.name not in retained_versions:
                    shutil.rmtree(directory)

        output(f"Installed forge-proxy {version} at {slot}")
        output("Next, configure and verify the installation:")
        output("  forge-proxy init")
        output("  forge-proxy check")
        output("For noninteractive unmanaged setup:")
        output("  forge-proxy init --non-interactive --backend-url URL")
        output("  forge-proxy check")
        return state
    finally:
        if staged.exists():
            staged.unlink()


def update(
    version: str | None = None,
    *,
    paths: InstallPaths | None = None,
    transport: Transport | None = None,
    runner: ProcessRunner | None = None,
    path_adapter: PathAdapter | None = None,
    output: Callable[[str], None] = print,
    target: str | None = None,
    pointer_url: str = STABLE_POINTER_URL,
    manifest_url: str = RELEASE_MANIFEST_URL,
    asset_url: str = RELEASE_ASSET_URL,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    paths = paths or discover_paths()
    if not paths.state.is_file():
        raise InstallerError("forge-proxy is not installed")
    state = read_state(paths.state)
    _refuse_command_conflicts(paths, state, environ=environ)
    transport = transport or UrlTransport()
    exact = version is not None
    if version is None:
        version = parse_pointer(transport.read(pointer_url))
    else:
        parse_version(version)
    current = str(state["current_version"])
    if parse_version(version) == parse_version(current):
        output(f"forge-proxy {current} is already installed")
        return None
    if parse_version(version) < parse_version(current):
        if exact:
            raise InstallerError(
                f"update cannot downgrade {current} to {version}; use the external "
                "installer for exact-version reinstall or recovery"
            )
        output(
            f"Installed forge-proxy {current} is newer than stable {version}; no update applied"
        )
        return None
    manifest = parse_manifest(
        transport.read(manifest_url.format(version=version)), version
    )
    target = target or native_target(system=paths.system)
    if target not in manifest:
        raise InstallerError(f"release manifest has no artifact for {target}")
    entry = manifest[target]
    retained = {item["version"]: item for item in state["verified_slots"]}
    slot = paths.slot(version)
    if (
        version in retained
        and retained[version]["sha256"] == entry["sha256"]
        and slot.is_file()
        and _sha256(slot) == entry["sha256"]
    ):
        artifact = slot
    else:
        payload = transport.read(asset_url.format(version=version, name=entry["name"]))
        if len(payload) != entry["size"]:
            raise InstallerError(
                "downloaded artifact size does not match release manifest"
            )
        paths.staging.mkdir(parents=True, exist_ok=True)
        artifact = (
            paths.staging / f"download-{uuid.uuid4().hex}{Path(entry['name']).suffix}"
        )
        _atomic_write(artifact, payload, executable=paths.system != "Windows")
    try:
        return install_artifact(
            artifact,
            version,
            entry["sha256"],
            paths=paths,
            no_init=False,
            runner=runner,
            path_adapter=path_adapter,
            output=output,
            environ=environ,
        )
    finally:
        if artifact.parent == paths.staging and artifact.exists():
            artifact.unlink()


def validate_owned_install(paths: InstallPaths) -> dict[str, Any]:
    state = read_state(paths.state)
    if (
        Path(state["root"]) != paths.root
        or Path(state["command_dir"]) != paths.command_dir
    ):
        raise InstallerError("installed ownership paths do not match this installation")
    expected = _marker_content(paths, str(state["ownership_id"]))
    if (
        not paths.marker.is_file()
        or paths.marker.read_text(encoding="utf-8") != expected
    ):
        raise InstallerError("installed ownership marker does not match state")
    if not paths.uninstaller.is_file():
        raise InstallerError("installed native uninstaller is missing")
    return state


def delegate_uninstall(paths: InstallPaths | None = None) -> None:
    paths = paths or discover_paths()
    validate_owned_install(paths)
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
    }
    if paths.system == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        kwargs["shell"] = True
        command = [str(paths.uninstaller), str(os.getpid())]
    else:
        kwargs["start_new_session"] = True
        command = [str(paths.uninstaller), str(os.getpid())]
    subprocess.Popen(command, **kwargs)


def uninstall_owned(
    paths: InstallPaths,
    *,
    path_adapter: PathAdapter | None = None,
) -> None:
    """Synchronous ownership-aware equivalent used by local fixture tests."""
    state = validate_owned_install(paths)
    owned_command = _command_is_owned(paths, state)
    path_adapter = path_adapter or default_path_adapter(paths)
    path_adapter.remove(state["path_integration"])
    if owned_command:
        paths.command.unlink()
    for path in (paths.state, paths.marker, paths.uninstaller):
        if path.exists():
            path.unlink()
    for directory in (paths.versions, paths.staging):
        if directory.exists():
            shutil.rmtree(directory)
    for directory in (paths.command_dir, paths.root):
        try:
            directory.rmdir()
        except OSError:
            pass
