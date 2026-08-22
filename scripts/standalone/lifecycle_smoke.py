"""Exercise a selected frozen artifact through the Proxy release lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from scripts.standalone.release import artifact_name, validate_manifest


ROOT = Path(__file__).resolve().parents[2]
STABLE_POINTER_URL = (
    "https://raw.githubusercontent.com/antoinezambelli/forge/"
    "main/installer/proxy-stable.txt"
)
RELEASE_BASE_URL = "https://github.com/antoinezambelli/forge/releases/download"
Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class ReleaseArtifact:
    path: Path
    version: str
    sha256: str
    target: str

    @property
    def name(self) -> str:
        return artifact_name(self.target)


class LocalReleaseServer(ThreadingHTTPServer):
    routes: dict[str, tuple[int, bytes]]


class LocalReleaseHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        server = self.server
        assert isinstance(server, LocalReleaseServer)
        status, payload = server.routes.get(self.path, (404, b"missing"))
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def local_release_server(
    routes: dict[str, tuple[int, bytes]],
) -> Iterator[str]:
    server = LocalReleaseServer(("127.0.0.1", 0), LocalReleaseHandler)
    server.routes = routes
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(executable: Path, arguments: list[str]) -> list[str]:
    if os.name == "nt" and executable.suffix.lower() in {".cmd", ".bat"}:
        return ["cmd", "/d", "/c", str(executable), *arguments]
    return [str(executable), *arguments]


def named_command(arguments: list[str]) -> list[str]:
    if os.name == "nt":
        return ["cmd", "/d", "/c", "forge-proxy", *arguments]
    return ["forge-proxy", *arguments]


def run_process(
    arguments: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    runner: Runner = subprocess.run,
    expected_error: str | None = None,
) -> dict[str, Any]:
    result = runner(
        arguments,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    record: dict[str, Any] = {
        "command": [Path(arguments[0]).name, *arguments[1:]],
        "status": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if expected_error is None:
        if result.returncode != 0:
            raise RuntimeError(
                f"lifecycle step failed ({' '.join(record['command'])}): "
                f"{result.stderr or result.stdout}"
            )
    else:
        if result.returncode == 0:
            raise RuntimeError(
                f"lifecycle step unexpectedly succeeded: {' '.join(record['command'])}"
            )
        if expected_error.lower() not in (result.stderr + result.stdout).lower():
            raise RuntimeError(
                f"lifecycle failure did not report {expected_error!r}: "
                f"{result.stderr or result.stdout}"
            )
        record["expected_failure"] = expected_error
    return record


def run_step(
    executable: Path,
    arguments: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    runner: Runner = subprocess.run,
    expected_error: str | None = None,
) -> dict[str, Any]:
    return run_process(
        command(executable, arguments),
        cwd=cwd,
        env=env,
        runner=runner,
        expected_error=expected_error,
    )


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if root.exists():
        for path in sorted(root.rglob("*")):
            digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
            if path.is_file():
                digest.update(path.read_bytes())
    return digest.hexdigest()


def version_key(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid release version: {version}")
    if any(len(part) > 1 and part.startswith("0") for part in parts):
        raise ValueError(f"invalid release version: {version}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def next_patch_version(version: str) -> str:
    major, minor, patch = version_key(version)
    return f"{major}.{minor}.{patch + 1}"


def _read_url(url: str, *, missing_ok: bool = False) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if missing_ok and exc.code == 404:
            return None
        raise RuntimeError(f"download unavailable: {url}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"download unavailable: {url}") from exc


def resolve_published_baseline(
    target: str,
    destination: Path,
    *,
    pointer_url: str = STABLE_POINTER_URL,
    release_base_url: str = RELEASE_BASE_URL,
    reader: Callable[..., bytes | None] = _read_url,
) -> ReleaseArtifact | None:
    raw_pointer = reader(pointer_url, missing_ok=True)
    if raw_pointer is None:
        return None
    try:
        pointer = raw_pointer.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("stable pointer is not ASCII") from exc
    if pointer.endswith("\n"):
        pointer = pointer[:-1]
    version_key(pointer)
    if "\n" in pointer or "\r" in pointer:
        raise ValueError("stable pointer must contain one bare X.Y.Z line")

    base = release_base_url.rstrip("/")
    manifest_url = f"{base}/v{pointer}/proxy-{pointer}.json"
    raw_manifest = reader(manifest_url)
    if raw_manifest is None:
        raise RuntimeError(f"download unavailable: {manifest_url}")
    try:
        document = json.loads(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("published Proxy manifest is not valid JSON") from exc
    manifest = validate_manifest(document, pointer)
    entry = manifest["artifacts"][target]
    artifact_url = f"{base}/v{pointer}/{entry['name']}"
    payload = reader(artifact_url)
    if payload is None:
        raise RuntimeError(f"download unavailable: {artifact_url}")
    if len(payload) != entry["size"]:
        raise ValueError("published baseline size does not match its manifest")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / entry["name"]
    path.write_bytes(payload)
    if artifact_sha256(path) != entry["sha256"]:
        raise ValueError("published baseline checksum does not match its manifest")
    if os.name != "nt":
        path.chmod(0o755)
    return ReleaseArtifact(path, pointer, entry["sha256"], target)


def retrievable_published_baseline(
    target: str,
    destination: Path,
    *,
    pointer_url: str = STABLE_POINTER_URL,
    release_base_url: str = RELEASE_BASE_URL,
    reader: Callable[..., bytes | None] = _read_url,
) -> ReleaseArtifact | None:
    try:
        return resolve_published_baseline(
            target,
            destination,
            pointer_url=pointer_url,
            release_base_url=release_base_url,
            reader=reader,
        )
    except Exception:
        return None


def release_manifest(artifact: ReleaseArtifact, *, sha256: str | None = None) -> bytes:
    document = {
        "version": artifact.version,
        "artifacts": {
            artifact.target: {
                "name": artifact.name,
                "sha256": sha256 or artifact.sha256,
                "size": artifact.path.stat().st_size,
            }
        },
    }
    return (json.dumps(document, sort_keys=True) + "\n").encode()


def set_release_routes(
    routes: dict[str, tuple[int, bytes]],
    artifact: ReleaseArtifact,
    *,
    sha256: str | None = None,
) -> None:
    routes[f"/v{artifact.version}/proxy-{artifact.version}.json"] = (
        200,
        release_manifest(artifact, sha256=sha256),
    )
    routes[f"/v{artifact.version}/{artifact.name}"] = (
        200,
        artifact.path.read_bytes(),
    )


def isolated_environment(root: Path, path_file: Path) -> dict[str, str]:
    user = root / "user"
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(user),
            "USERPROFILE": str(user),
            "APPDATA": str(user / "AppData" / "Roaming"),
            "LOCALAPPDATA": str(user / "AppData" / "Local"),
            "XDG_CONFIG_HOME": str(user / ".config"),
            "FORGE_PROXY_PATH_FILE": str(path_file),
        }
    )
    return env


def release_test_environment(
    env: dict[str, str], base_url: str, temporary: Path
) -> dict[str, str]:
    env = dict(env)
    env.update(
        {
            "_FORGE_PROXY_BOOTSTRAP_TESTING": "1",
            "_FORGE_PROXY_BOOTSTRAP_SYSTEM": platform.system(),
            "_FORGE_PROXY_BOOTSTRAP_MACHINE": platform.machine(),
            "_FORGE_PROXY_BOOTSTRAP_POINTER_URL": f"{base_url}/pointer",
            "_FORGE_PROXY_BOOTSTRAP_RELEASE_BASE_URL": base_url,
            "_FORGE_PROXY_BOOTSTRAP_TEMP_ROOT": str(temporary),
            "_FORGE_PROXY_INSTALLER_TESTING": "1",
            "_FORGE_PROXY_INSTALLER_POINTER_URL": f"{base_url}/pointer",
            "_FORGE_PROXY_INSTALLER_RELEASE_BASE_URL": base_url,
        }
    )
    if platform.system() == "Linux":
        result = subprocess.run(
            ["ldd", "--version"], capture_output=True, text=True, check=False
        )
        env["_FORGE_PROXY_BOOTSTRAP_LDD_OUTPUT"] = result.stdout + result.stderr
    return env


def bootstrap_arguments(artifact: ReleaseArtifact, install_root: Path) -> list[str]:
    if artifact.target == "windows-x86_64":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell is None:
            raise RuntimeError("PowerShell is required for the Windows bootstrap gate")
        return [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "install.ps1"),
            "-Version",
            artifact.version,
            "-NoInit",
            "-InstallRoot",
            str(install_root),
        ]
    return [
        "sh",
        str(ROOT / "install.sh"),
        "--version",
        artifact.version,
        "--no-init",
        "--install-root",
        str(install_root),
    ]


def shim_path(install_root: Path, target: str) -> Path:
    name = "forge-proxy.cmd" if target == "windows-x86_64" else "forge-proxy"
    return install_root / "bin" / name


def install_arguments(artifact: ReleaseArtifact, install_root: Path) -> list[str]:
    return [
        "install-artifact",
        "--version",
        artifact.version,
        "--sha256",
        artifact.sha256,
        "--no-init",
        "--install-root",
        str(install_root),
    ]


def slot_path(install_root: Path, artifact: ReleaseArtifact) -> Path:
    executable = (
        "forge-proxy.exe" if artifact.target == "windows-x86_64" else "forge-proxy"
    )
    return install_root / "versions" / artifact.version / executable


def command_environment(env: dict[str, str], command_dir: Path) -> dict[str, str]:
    resolved = dict(env)
    inherited = resolved.get("PATH", "")
    resolved["PATH"] = (
        f"{command_dir}{os.pathsep}{inherited}" if inherited else str(command_dir)
    )
    return resolved


def foreign_path_command(directory: Path, target: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    name = "forge-proxy.exe" if target == "windows-x86_64" else "forge-proxy"
    path = directory / name
    path.write_bytes(b"foreign forge-proxy PATH fixture\n")
    if target != "windows-x86_64":
        path.chmod(0o755)
    return path


def replace_installed_command(path: Path, target: str) -> bytes:
    if path.exists() or path.is_symlink():
        path.unlink()
    content = b"foreign replacement command\n"
    path.write_bytes(content)
    if target != "windows-x86_64":
        path.chmod(0o755)
    return content


def wait_for_removal(path: Path, *, seconds: float = 20) -> None:
    deadline = time.monotonic() + seconds
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    if path.exists():
        raise RuntimeError(f"owned installation state remained after uninstall: {path}")


def path_snapshot(path: Path) -> tuple[str, bytes | str]:
    if path.is_symlink():
        return ("symlink", os.readlink(path))
    return ("file", path.read_bytes())


def active_snapshot(
    install_root: Path, target: str
) -> tuple[bytes, tuple[str, bytes | str], str]:
    state_path = install_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    active = ReleaseArtifact(
        install_root,
        str(state["current_version"]),
        "",
        target,
    )
    return (
        state_path.read_bytes(),
        path_snapshot(shim_path(install_root, target)),
        artifact_sha256(slot_path(install_root, active)),
    )


def assert_active(
    artifact: ReleaseArtifact,
    install_root: Path,
    isolation: Path,
    env: dict[str, str],
    steps: list[dict[str, Any]],
) -> None:
    state = json.loads((install_root / "state.json").read_text(encoding="utf-8"))
    shim = shim_path(install_root, artifact.target)
    if state["current_version"] != artifact.version:
        raise RuntimeError("installed state reports the wrong active version")
    if (
        Path(state["command_dir"]).resolve() != shim.parent.resolve()
        or not state["path_integration"]
    ):
        raise RuntimeError("owned shim/PATH state was not recorded")
    slot = slot_path(install_root, artifact)
    if artifact_sha256(slot) != artifact.sha256:
        raise RuntimeError("active slot does not contain the selected bytes")
    steps.append(run_step(shim, ["--version"], cwd=isolation, env=env))
    if steps[-1]["stdout"].strip() != artifact.version:
        raise RuntimeError("installed command reported the wrong version")
    steps.append(run_step(shim, ["check"], cwd=isolation, env=env))


def profile_snapshot(user: Path) -> tuple[Path, bytes]:
    profiles = list(user.rglob("*.toml"))
    if len(profiles) != 1:
        raise RuntimeError("lifecycle gate expected exactly one initialized profile")
    return profiles[0], profiles[0].read_bytes()


def assert_profile(snapshot: tuple[Path, bytes]) -> None:
    path, content = snapshot
    if not path.is_file() or path.read_bytes() != content:
        raise RuntimeError("install lifecycle changed or removed the managed profile")


def assert_failed_update_preserved(
    before: tuple[bytes, tuple[str, bytes | str], str],
    install_root: Path,
    active: ReleaseArtifact,
    isolation: Path,
    env: dict[str, str],
    steps: list[dict[str, Any]],
) -> None:
    if active_snapshot(install_root, active.target) != before:
        raise RuntimeError("failed update changed the active installation")
    assert_active(active, install_root, isolation, env, steps)


def candidate_ownership_prelude(
    candidate: ReleaseArtifact,
    isolation: Path,
    steps: list[dict[str, Any]],
) -> dict[str, bool]:
    fixture = isolation / "candidate-ownership"
    user = fixture / "user"
    user.mkdir(parents=True)
    install_root = fixture / "install root"
    path_file = fixture / "user-path.txt"
    path_file.write_text("existing-path", encoding="utf-8")
    env = isolated_environment(fixture, path_file)
    if candidate.target != "windows-x86_64":
        env["SHELL"] = "/bin/bash"

    foreign_dir = fixture / "foreign-command"
    foreign = foreign_path_command(foreign_dir, candidate.target)
    foreign_before = foreign.read_bytes()
    conflict_env = command_environment(env, foreign_dir)
    install_args = install_arguments(candidate, install_root)
    steps.append(
        run_step(
            candidate.path,
            install_args,
            cwd=fixture,
            env=conflict_env,
            expected_error="unowned forge-proxy command",
        )
    )
    if foreign.read_bytes() != foreign_before:
        raise RuntimeError("collision refusal changed the foreign PATH command")
    if install_root.exists():
        raise RuntimeError("collision refusal published standalone installation state")
    if path_file.read_text(encoding="utf-8") != "existing-path":
        raise RuntimeError("collision refusal changed isolated PATH state")

    foreign.unlink()
    foreign_dir.rmdir()
    steps.append(run_step(candidate.path, install_args, cwd=fixture, env=env))
    shim = shim_path(install_root, candidate.target)
    steps.append(
        run_step(
            shim,
            [
                "init",
                "--non-interactive",
                "--force",
                "--backend-url",
                "http://127.0.0.1:1",
            ],
            cwd=fixture,
            env=env,
        )
    )
    resolved_env = command_environment(env, shim.parent)
    steps.append(
        run_process(
            named_command(["--version"]),
            cwd=fixture,
            env=resolved_env,
        )
    )
    if steps[-1]["stdout"].strip() != candidate.version:
        raise RuntimeError("bare forge-proxy resolved to the wrong installation")
    steps.append(run_process(named_command(["check"]), cwd=fixture, env=resolved_env))

    replacement = replace_installed_command(shim, candidate.target)
    steps.append(
        run_step(
            candidate.path,
            install_args,
            cwd=fixture,
            env=resolved_env,
            expected_error="unowned forge-proxy command",
        )
    )
    if shim.read_bytes() != replacement:
        raise RuntimeError("reinstall refusal changed the replacement command")

    state = install_root / "state.json"
    marker = install_root / "ownership.txt"
    uninstaller_name = (
        "uninstall.cmd" if candidate.target == "windows-x86_64" else "uninstall.sh"
    )
    uninstaller = install_root / uninstaller_name
    versions = install_root / "versions"
    staging = install_root / ".staging"
    profile = profile_snapshot(user)
    steps.append(
        run_step(
            slot_path(install_root, candidate),
            ["uninstall"],
            cwd=fixture,
            env=resolved_env,
        )
    )
    for owned_path in (state, marker, uninstaller, versions, staging):
        wait_for_removal(owned_path)
    if not shim.is_file() or shim.read_bytes() != replacement:
        raise RuntimeError("uninstall removed or changed the replacement command")
    if path_file.read_text(encoding="utf-8") != "existing-path":
        raise RuntimeError("ownership uninstall did not restore isolated PATH state")
    assert_profile(profile)

    shim.unlink()
    for directory in (shim.parent, install_root):
        try:
            directory.rmdir()
        except OSError:
            pass
    return {
        "collision_rejected": True,
        "foreign_command_preserved": True,
        "bare_command_resolved": True,
        "replacement_reinstall_rejected": True,
        "replacement_survived_uninstall": True,
    }


def run_lifecycle(
    artifact: Path,
    version: str,
    digest: str,
    target: str,
    *,
    pointer_url: str = STABLE_POINTER_URL,
    release_base_url: str = RELEASE_BASE_URL,
) -> dict[str, Any]:
    artifact = artifact.resolve()
    version_key(version)
    if artifact_sha256(artifact) != digest:
        raise ValueError("selected artifact digest changed before lifecycle smoke")
    candidate = ReleaseArtifact(artifact, version, digest, target)

    with tempfile.TemporaryDirectory(prefix="forge-lifecycle-") as raw_root:
        isolation = Path(raw_root)
        user = isolation / "user"
        user.mkdir()
        sentinel = user / "unchanged.txt"
        sentinel.write_text("real-user-state-sentinel\n", encoding="utf-8")
        before_user = tree_digest(user)
        install_root = isolation / "install root"
        path_file = isolation / "user-path.txt"
        path_file.write_text("existing-path", encoding="utf-8")
        bootstrap_temp = isolation / "bootstrap-temp"
        bootstrap_temp.mkdir()
        env = isolated_environment(isolation, path_file)
        steps: list[dict[str, Any]] = []

        steps.append(run_step(candidate.path, ["--version"], cwd=isolation, env=env))
        if steps[-1]["stdout"].strip() != candidate.version:
            raise RuntimeError("selected artifact reported the wrong version")
        steps.append(run_step(candidate.path, ["--help"], cwd=isolation, env=env))
        steps.append(
            run_step(
                candidate.path,
                ["_installer-self-check", "--expected-version", candidate.version],
                cwd=isolation,
                env=env,
            )
        )
        ownership = candidate_ownership_prelude(candidate, isolation, steps)

        baseline = retrievable_published_baseline(
            target,
            isolation / "baseline",
            pointer_url=pointer_url,
            release_base_url=release_base_url,
        )
        if baseline is not None and version_key(baseline.version) >= version_key(
            version
        ):
            raise RuntimeError(
                "published stable Proxy baseline must be older than the candidate"
            )

        routes: dict[str, tuple[int, bytes]] = {}
        set_release_routes(routes, candidate)
        if baseline is not None:
            set_release_routes(routes, baseline)

        with local_release_server(routes) as base_url:
            env = release_test_environment(env, base_url, bootstrap_temp)
            initial = baseline or candidate
            steps.append(
                run_process(
                    bootstrap_arguments(initial, install_root),
                    cwd=isolation,
                    env=env,
                )
            )
            shim = shim_path(install_root, target)
            steps.append(
                run_step(
                    shim,
                    [
                        "init",
                        "--non-interactive",
                        "--force",
                        "--backend-url",
                        "http://127.0.0.1:1",
                    ],
                    cwd=isolation,
                    env=env,
                )
            )
            assert_active(initial, install_root, isolation, env, steps)
            profile = profile_snapshot(user)

            if baseline is not None:
                steps.append(
                    run_step(
                        shim,
                        ["update", "--version", candidate.version],
                        cwd=isolation,
                        env=env,
                    )
                )
                assert_active(candidate, install_root, isolation, env, steps)
                assert_profile(profile)

            install_args = install_arguments(candidate, install_root)
            steps.append(run_step(candidate.path, install_args, cwd=isolation, env=env))
            assert_active(candidate, install_root, isolation, env, steps)
            assert_profile(profile)

            failure_version = next_patch_version(candidate.version)
            failed_candidate = ReleaseArtifact(
                candidate.path,
                failure_version,
                candidate.sha256,
                candidate.target,
            )

            routes["/pointer"] = (200, f"{failure_version}\n".encode())
            routes.pop(f"/v{failure_version}/proxy-{failure_version}.json", None)
            protected = active_snapshot(install_root, target)
            steps.append(
                run_step(
                    shim,
                    ["update"],
                    cwd=isolation,
                    env=env,
                    expected_error="download unavailable",
                )
            )
            assert_failed_update_preserved(
                protected, install_root, candidate, isolation, env, steps
            )

            set_release_routes(routes, failed_candidate, sha256="f" * 64)
            protected = active_snapshot(install_root, target)
            steps.append(
                run_step(
                    shim,
                    ["update", "--version", failure_version],
                    cwd=isolation,
                    env=env,
                    expected_error="checksum mismatch",
                )
            )
            assert_failed_update_preserved(
                protected, install_root, candidate, isolation, env, steps
            )

            set_release_routes(routes, failed_candidate)
            protected = active_snapshot(install_root, target)
            steps.append(
                run_step(
                    shim,
                    ["update", "--version", failure_version],
                    cwd=isolation,
                    env=env,
                    expected_error="does not match requested version",
                )
            )
            assert_failed_update_preserved(
                protected, install_root, candidate, isolation, env, steps
            )

            if baseline is not None:
                steps.append(
                    run_process(
                        bootstrap_arguments(baseline, install_root),
                        cwd=isolation,
                        env=env,
                    )
                )
                assert_active(baseline, install_root, isolation, env, steps)
                assert_profile(profile)
                final_active = baseline
                baseline_status = "forward-update-and-exact-recovery-exercised"
            else:
                final_active = candidate
                baseline_status = (
                    "cross-version checks skipped: no retrievable older Proxy artifact; "
                    "candidate lifecycle and update failure paths exercised"
                )

            steps.append(
                run_step(
                    slot_path(install_root, final_active),
                    ["uninstall"],
                    cwd=isolation,
                    env=env,
                )
            )
            wait_for_removal(install_root)
            assert_profile(profile)

        if path_file.read_text(encoding="utf-8") != "existing-path":
            raise RuntimeError("uninstall did not restore isolated PATH state")
        if (
            not sentinel.is_file()
            or sentinel.read_text(encoding="utf-8") != "real-user-state-sentinel\n"
        ):
            raise RuntimeError("isolated user sentinel changed")
        if list(bootstrap_temp.iterdir()):
            raise RuntimeError("bootstrap left temporary files behind")
        return {
            "version": version,
            "sha256": digest,
            "artifact": artifact.name,
            "target": target,
            "baseline": baseline.version if baseline is not None else None,
            "baseline_status": baseline_status,
            "steps": steps,
            "command_ownership": ownership,
            "owned_state_removed": True,
            "path_state_restored": True,
            "profile_preserved": True,
            "real_user_state_untouched": True,
            "isolated_user_before": before_user,
            "isolated_user_after": tree_digest(user),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument(
        "--target",
        required=True,
        choices=("windows-x86_64", "linux-x86_64-gnu", "macos-arm64"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_lifecycle(args.artifact, args.version, args.sha256, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
