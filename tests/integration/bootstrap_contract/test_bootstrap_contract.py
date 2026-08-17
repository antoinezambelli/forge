"""Small subprocess contracts for the public bootstrap scripts.

Each collected case exercises exactly one bootstrap implementation. The suite
uses only fixture payloads and a localhost release server; frozen-artifact
lifecycle checks live under ``platform_acceptance``.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

import pytest

from tests.integration._bootstrap_support import (
    BASH,
    FIXTURES,
    INSTALL_PS1,
    INSTALL_SH,
    POWERSHELL,
    VERSION,
    bash_path,
    fixture_routes,
    fixture_server,
    handoff_lines,
    run_powershell,
    run_shell,
)

pytestmark = pytest.mark.integration


def require_native_runner(runner: str) -> None:
    native = "powershell" if os.name == "nt" else "shell"
    if runner != native:
        pytest.skip(f"{runner} is not the native bootstrap on this runner")


def shell_target() -> str:
    return "macos-arm64" if platform.system() == "Darwin" else "linux-x86_64-gnu"


@pytest.mark.parametrize(
    ("runner", "manifest_name"),
    [
        ("powershell", "proxy-1.2.3.json"),
        ("powershell", "proxy-1.2.3-order-whitespace.json"),
        ("shell", "proxy-1.2.3.json"),
        ("shell", "proxy-1.2.3-order-whitespace.json"),
    ],
)
def test_manifest_drives_one_bootstrap_handoff(
    tmp_path: Path, runner: str, manifest_name: str
) -> None:
    require_native_runner(runner)
    routes = fixture_routes(manifest_name)
    with fixture_server(routes) as (server, base_url):
        if runner == "powershell":
            root = tmp_path / "PowerShell root with spaces"
            result = run_powershell(
                tmp_path,
                base_url,
                ["-Version", VERSION, "-NoInit", "-InstallRoot", str(root)],
            )
            handoff = tmp_path / "powershell-handoff.txt"
            expected_sha = (
                "6f2be1b60db97e83baf60316f1a4463333aeb244e8e3c6223c2c0fa21106d873"
            )
            temp_root = tmp_path / "bootstrap temp"
        else:
            root = "/opt/forge proxy"
            result = run_shell(
                tmp_path,
                base_url,
                ["--version", VERSION, "--no-init", "--install-root", str(root)],
            )
            handoff = tmp_path / "shell-handoff.txt"
            expected_sha = (
                "9ffc2e63c6095b928c58612081e8cccab4513c2a1e25b61973b775a47e487275"
            )
            temp_root = tmp_path / "shell-temp"

    assert result.returncode == 0, result.stderr
    assert handoff_lines(handoff) == [
        "install-artifact",
        "--version",
        VERSION,
        "--sha256",
        expected_sha,
        "--no-init",
        "--install-root",
        str(root),
    ]
    assert "/pointer" not in server.requests
    assert not list(temp_root.iterdir())


@pytest.mark.parametrize("runner", ["powershell", "shell"])
def test_pointer_selection_and_public_help(tmp_path: Path, runner: str) -> None:
    require_native_runner(runner)
    routes = fixture_routes()
    routes["/pointer"] = (200, b"1.2.3\n")
    with fixture_server(routes) as (_server, base_url):
        result = (
            run_powershell(tmp_path, base_url, [])
            if runner == "powershell"
            else run_shell(tmp_path, base_url, [])
        )
    assert result.returncode == 0
    assert "handoff stdout" in result.stdout
    assert "handoff stderr" in result.stderr

    command = (
        [POWERSHELL or "powershell", "-NoProfile", "-File", str(INSTALL_PS1), "-Help"]
        if runner == "powershell"
        else [str(BASH), bash_path(INSTALL_SH), "--help"]
    )
    help_text = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert "X.Y.Z" in help_text
    assert "NoInit" in help_text or "no-init" in help_text
    assert "InstallRoot" in help_text or "install-root" in help_text
    assert "install-artifact --version X.Y.Z --sha256 HEX" in help_text
    assert "inspect" in help_text.lower()


@pytest.mark.parametrize("runner", ["powershell", "shell"])
def test_missing_stable_pointer_reports_no_published_release(
    tmp_path: Path, runner: str
) -> None:
    require_native_runner(runner)
    with fixture_server(fixture_routes()) as (server, base_url):
        result = (
            run_powershell(tmp_path, base_url, [])
            if runner == "powershell"
            else run_shell(tmp_path, base_url, [])
        )
    assert result.returncode == 1
    assert "no stable standalone Proxy release has been published" in result.stderr
    assert server.requests == ["/pointer"]
    temp_name = "bootstrap temp" if runner == "powershell" else "shell-temp"
    assert not list((tmp_path / temp_name).iterdir())


@pytest.mark.parametrize("runner", ["powershell", "shell"])
def test_unavailable_stable_pointer_is_not_reported_as_unpublished(
    tmp_path: Path, runner: str
) -> None:
    require_native_runner(runner)
    with fixture_server({"/pointer": (503, b"unavailable")}) as (server, base_url):
        result = (
            run_powershell(tmp_path, base_url, [])
            if runner == "powershell"
            else run_shell(tmp_path, base_url, [])
        )
    assert result.returncode == 1
    assert "download unavailable" in result.stderr
    assert "no stable standalone Proxy release has been published" not in result.stderr
    assert server.requests == ["/pointer"]


@pytest.mark.parametrize(
    ("runner", "arguments", "expected"),
    [
        ("powershell", ["-Version", VERSION], "404"),
        ("shell", ["--version", VERSION], "download unavailable"),
    ],
)
def test_explicit_version_download_failure_is_not_a_missing_stable_release(
    tmp_path: Path, runner: str, arguments: list[str], expected: str
) -> None:
    require_native_runner(runner)
    with fixture_server({}) as (server, base_url):
        result = (
            run_powershell(tmp_path, base_url, arguments)
            if runner == "powershell"
            else run_shell(tmp_path, base_url, arguments)
        )
    assert result.returncode == 1
    assert expected in result.stderr
    assert "no stable standalone Proxy release has been published" not in result.stderr
    assert server.requests == [f"/v{VERSION}/proxy-{VERSION}.json"]
    temp_name = "bootstrap temp" if runner == "powershell" else "shell-temp"
    assert not list((tmp_path / temp_name).iterdir())


@pytest.mark.parametrize(
    ("runner", "kwargs", "message", "exact"),
    [
        (
            "powershell",
            {"system": "Linux", "machine": "AMD64"},
            "unsupported standalone target",
            False,
        ),
        (
            "powershell",
            {"system": "Windows", "machine": "ARM64"},
            "unsupported standalone target",
            False,
        ),
        (
            "shell",
            {"system": "Darwin", "machine": "x86_64"},
            "unsupported standalone target",
            False,
        ),
        (
            "shell",
            {
                "system": "Linux",
                "machine": "x86_64",
                "ldd_output": "musl libc (x86_64) Version 1.2.5",
            },
            "could not be proven",
            False,
        ),
        (
            "shell",
            {"system": "Linux", "machine": "x86_64", "ldd_output": "ldd unknown"},
            "could not be proven",
            False,
        ),
        (
            "shell",
            {
                "system": "Linux",
                "machine": "x86_64",
                "ldd_output": "ldd (GNU libc) 2.38",
            },
            "2.39 or newer",
            False,
        ),
        (
            "powershell",
            {"system": "Windows", "machine": "ARM64"},
            "unsupported standalone target",
            True,
        ),
        (
            "shell",
            {
                "system": "Linux",
                "machine": "x86_64",
                "ldd_output": "ldd (GNU libc) 2.38",
            },
            "2.39 or newer",
            True,
        ),
    ],
)
def test_unsupported_hosts_make_zero_requests(
    tmp_path: Path,
    runner: str,
    kwargs: dict[str, str],
    message: str,
    exact: bool,
) -> None:
    require_native_runner(runner)
    with fixture_server({}) as (server, base_url):
        arguments = ["-Version", VERSION] if runner == "powershell" and exact else []
        if runner == "shell" and exact:
            arguments = ["--version", VERSION]
        result = (
            run_powershell(tmp_path, base_url, arguments, **kwargs)
            if runner == "powershell"
            else run_shell(tmp_path, base_url, arguments, **kwargs)
        )
    assert result.returncode != 0
    assert message in result.stderr
    assert server.requests == []


@pytest.mark.parametrize(
    "banner",
    [
        "ldd (Ubuntu GLIBC 2.39-0ubuntu8) 2.39",
        "ldd (Debian GLIBC 2.41-12+deb13u1) 2.41",
        "ldd (GNU libc) 2.100",
    ],
)
@pytest.mark.skipif(platform.system() != "Linux", reason="Linux bootstrap contract")
def test_linux_glibc_banners_at_and_above_floor_are_accepted(
    tmp_path: Path, banner: str
) -> None:
    with fixture_server(fixture_routes()) as (_server, base_url):
        result = run_shell(
            tmp_path,
            base_url,
            ["--version", VERSION],
            system="Linux",
            machine="x86_64",
            ldd_output=banner,
        )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("runner", ["powershell", "shell"])
@pytest.mark.parametrize("failure", ["artifact", "checksum", "handoff"])
def test_artifact_checksum_and_handoff_failures_cleanup(
    tmp_path: Path, runner: str, failure: str
) -> None:
    require_native_runner(runner)
    routes = fixture_routes()
    arguments = (
        ["-Version", VERSION] if runner == "powershell" else ["--version", VERSION]
    )
    status = 0
    if failure == "artifact":
        target = "windows-x86_64" if runner == "powershell" else shell_target()
        document = json.loads((FIXTURES / f"proxy-{VERSION}.json").read_bytes())
        suffix = document["artifacts"][target]["name"]
        routes.pop(f"/v{VERSION}/{suffix}")
    elif failure == "checksum":
        document = json.loads((FIXTURES / f"proxy-{VERSION}.json").read_bytes())
        target = "windows-x86_64" if runner == "powershell" else shell_target()
        document["artifacts"][target]["sha256"] = "f" * 64
        routes[f"/v{VERSION}/proxy-{VERSION}.json"] = (
            200,
            json.dumps(document).encode(),
        )
    else:
        status = 37

    with fixture_server(routes) as (_server, base_url):
        result = (
            run_powershell(tmp_path, base_url, arguments, status=status)
            if runner == "powershell"
            else run_shell(tmp_path, base_url, arguments, status=status)
        )
    assert result.returncode == (37 if failure == "handoff" else 1)
    if failure == "handoff":
        assert "handoff stdout" in result.stdout
        assert "handoff stderr" in result.stderr
    temp_name = "bootstrap temp" if runner == "powershell" else "shell-temp"
    assert not list((tmp_path / temp_name).iterdir())


@pytest.mark.parametrize("runner", ["powershell", "shell"])
@pytest.mark.parametrize("failure", ["version", "target"])
def test_manifest_version_and_selected_target_are_required(
    tmp_path: Path, runner: str, failure: str
) -> None:
    require_native_runner(runner)
    document = json.loads((FIXTURES / f"proxy-{VERSION}.json").read_bytes())
    if failure == "version":
        document["version"] = "9.9.9"
        expected = "version does not match"
    else:
        document["artifacts"].pop(
            "windows-x86_64" if runner == "powershell" else shell_target()
        )
        expected = "has no artifact"
    routes = fixture_routes()
    routes[f"/v{VERSION}/proxy-{VERSION}.json"] = (200, json.dumps(document).encode())
    with fixture_server(routes) as (_server, base_url):
        result = (
            run_powershell(tmp_path, base_url, ["-Version", VERSION])
            if runner == "powershell"
            else run_shell(tmp_path, base_url, ["--version", VERSION])
        )
    assert result.returncode == 1
    assert expected in result.stderr


@pytest.mark.parametrize(
    ("runner", "arguments"),
    [
        ("powershell", ["-Version", "1.02.3"]),
        ("powershell", ["-InstallRoot", "relative"]),
        ("powershell", ["-Stable"]),
        ("shell", ["--version", "1.02.3"]),
        ("shell", ["--install-root", "relative"]),
        ("shell", ["--stable"]),
    ],
)
def test_malformed_input_fails_before_fetch(
    tmp_path: Path,
    runner: str,
    arguments: list[str],
) -> None:
    require_native_runner(runner)
    with fixture_server({}) as (server, base_url):
        result = (
            run_powershell(tmp_path, base_url, arguments)
            if runner == "powershell"
            else run_shell(tmp_path, base_url, arguments)
        )
    assert result.returncode != 0
    assert server.requests == []


@pytest.mark.skipif(not BASH.is_file(), reason="Bash is unavailable")
@pytest.mark.skipif(os.name == "nt", reason="POSIX bootstrap contract")
def test_piped_posix_bootstrap_hands_off_without_consuming_user_input(
    tmp_path: Path,
) -> None:
    handoff = tmp_path / "piped-posix-handoff.txt"
    temp_root = tmp_path / "piped-shell-temp"
    temp_root.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "_FORGE_PROXY_BOOTSTRAP_TESTING": "1",
            "_FORGE_PROXY_BOOTSTRAP_SYSTEM": platform.system(),
            "_FORGE_PROXY_BOOTSTRAP_MACHINE": platform.machine(),
            "_FORGE_PROXY_BOOTSTRAP_LDD_OUTPUT": "ldd (GNU libc) 2.39",
            "_FORGE_PROXY_BOOTSTRAP_TEMP_ROOT": bash_path(temp_root),
            "FORGE_BOOTSTRAP_HANDOFF_LOG": bash_path(handoff),
            "FORGE_BOOTSTRAP_HANDOFF_STATUS": "0",
        }
    )
    routes = fixture_routes()
    routes["/install.sh"] = (200, INSTALL_SH.read_bytes())
    routes["/pointer"] = (200, f"{VERSION}\n".encode("ascii"))
    with fixture_server(routes) as (_server, base_url):
        env["_FORGE_PROXY_BOOTSTRAP_POINTER_URL"] = f"{base_url}/pointer"
        env["_FORGE_PROXY_BOOTSTRAP_RELEASE_BASE_URL"] = base_url
        result = subprocess.run(
            [str(BASH), "-c", f"curl -fsSL {base_url}/install.sh | sh"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=30,
        )
    assert result.returncode == 0, result.stderr
    assert "--no-init" not in handoff_lines(handoff)
