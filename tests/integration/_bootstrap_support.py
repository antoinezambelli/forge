"""Shared local-release fixtures for installer integration and acceptance tests."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import pytest


ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "proxy_bootstrap"
INSTALL_PS1 = ROOT / "install.ps1"
INSTALL_SH = ROOT / "install.sh"
POWERSHELL = shutil.which("powershell")
_BASH = shutil.which("bash")
BASH = Path(_BASH) if _BASH else Path(r"C:\Program Files\Git\bin\bash.exe")
VERSION = "1.2.3"


class FixtureServer(ThreadingHTTPServer):
    routes: dict[str, tuple[int, bytes]]
    requests: list[str]


class FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        server = self.server
        assert isinstance(server, FixtureServer)
        server.requests.append(self.path)
        status, payload = server.routes.get(self.path, (404, b"missing"))
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def fixture_server(
    routes: dict[str, tuple[int, bytes]],
) -> Iterator[tuple[FixtureServer, str]]:
    server = FixtureServer(("127.0.0.1", 0), FixtureHandler)
    server.routes = routes
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def bash_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if os.name != "nt":
        return value
    return f"/{value[0].lower()}{value[2:]}"


def fixture_routes(
    manifest_name: str = f"proxy-{VERSION}.json",
) -> dict[str, tuple[int, bytes]]:
    return {
        f"/v{VERSION}/proxy-{VERSION}.json": (
            200,
            (FIXTURES / manifest_name).read_bytes(),
        ),
        f"/v{VERSION}/forge-proxy-linux-x86_64-gnu": (
            200,
            (FIXTURES / "forge-proxy-linux-x86_64-gnu").read_bytes(),
        ),
        f"/v{VERSION}/forge-proxy-macos-arm64": (
            200,
            (FIXTURES / "forge-proxy-macos-arm64").read_bytes(),
        ),
        f"/v{VERSION}/forge-proxy-windows-x86_64.cmd": (
            200,
            (FIXTURES / "forge-proxy-windows-x86_64.cmd").read_bytes(),
        ),
    }


def run_powershell(
    tmp_path: Path,
    base_url: str,
    arguments: list[str],
    *,
    system: str = "Windows",
    machine: str = "AMD64",
    status: int = 0,
    input_text: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    if POWERSHELL is None:
        pytest.skip("Windows PowerShell is unavailable")
    temp_root = tmp_path / "bootstrap temp"
    temp_root.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "_FORGE_PROXY_BOOTSTRAP_TESTING": "1",
            "_FORGE_PROXY_BOOTSTRAP_SYSTEM": system,
            "_FORGE_PROXY_BOOTSTRAP_MACHINE": machine,
            "_FORGE_PROXY_BOOTSTRAP_POINTER_URL": f"{base_url}/pointer",
            "_FORGE_PROXY_BOOTSTRAP_RELEASE_BASE_URL": base_url,
            "_FORGE_PROXY_BOOTSTRAP_TEMP_ROOT": str(temp_root),
            "FORGE_BOOTSTRAP_HANDOFF_LOG": str(tmp_path / "powershell-handoff.txt"),
            "FORGE_BOOTSTRAP_HANDOFF_STATUS": str(status),
        }
    )
    # Windows PowerShell must use its own built-in modules rather than an
    # inherited PowerShell 7 module path.
    system_root = Path(env.get("SystemRoot", r"C:\Windows"))
    env["PSModulePath"] = str(
        system_root / "System32" / "WindowsPowerShell" / "v1.0" / "Modules"
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(INSTALL_PS1),
            *arguments,
        ],
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=120,
    )


def run_shell(
    tmp_path: Path,
    base_url: str,
    arguments: list[str],
    *,
    system: str | None = None,
    machine: str | None = None,
    ldd_output: str = "ldd (Ubuntu GLIBC 2.35-0ubuntu3.8) 2.35",
    status: int = 0,
) -> subprocess.CompletedProcess[str]:
    if not BASH.is_file():
        pytest.skip("Bash is unavailable")
    temp_root = tmp_path / "shell-temp"
    temp_root.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "_FORGE_PROXY_BOOTSTRAP_TESTING": "1",
            "_FORGE_PROXY_BOOTSTRAP_SYSTEM": system or platform.system(),
            "_FORGE_PROXY_BOOTSTRAP_MACHINE": machine or platform.machine(),
            "_FORGE_PROXY_BOOTSTRAP_LDD_OUTPUT": ldd_output,
            "_FORGE_PROXY_BOOTSTRAP_POINTER_URL": f"{base_url}/pointer",
            "_FORGE_PROXY_BOOTSTRAP_RELEASE_BASE_URL": base_url,
            "_FORGE_PROXY_BOOTSTRAP_TEMP_ROOT": bash_path(temp_root),
            "FORGE_BOOTSTRAP_HANDOFF_LOG": bash_path(tmp_path / "shell-handoff.txt"),
            "FORGE_BOOTSTRAP_HANDOFF_STATUS": str(status),
        }
    )
    return subprocess.run(
        [str(BASH), bash_path(INSTALL_SH), *arguments],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=30,
    )


def handoff_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()
