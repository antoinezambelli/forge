"""Cross-process smoke for a frozen Forge Proxy executable."""

from __future__ import annotations

import argparse
import json
import locale
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def reserve_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class MockBackend:
    def __init__(self, protocol: str) -> None:
        self.protocol = protocol
        self.requests: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *args: object) -> None:
                del args

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                body = json.loads(self.rfile.read(length))
                owner.requests.append({"path": self.path, "body": body})
                if owner.protocol == "anthropic":
                    response = {
                        "id": "msg_packaged", "type": "message",
                        "role": "assistant", "model": "claude-packaged",
                        "content": [{"type": "text", "text": "anthropic-ok"}],
                        "stop_reason": "end_turn", "stop_sequence": None,
                        "usage": {"input_tokens": 3, "output_tokens": 2},
                    }
                else:
                    response = {
                        "id": "chatcmpl-packaged", "object": "chat.completion",
                        "model": "mock-model",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "openai-ok"},
                            "finish_reason": "stop",
                        }],
                        "usage": {
                            "prompt_tokens": 3, "completion_tokens": 2,
                            "total_tokens": 5,
                        },
                    }
                payload = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    def __enter__(self) -> "MockBackend":
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def request_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        payload = response.read()
        return response.status, json.loads(payload) if payload else {}


def wait_for_health(port: int, process: subprocess.Popen[str]) -> float:
    started = time.monotonic()
    deadline = started + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"packaged proxy exited before health: {stdout}\n{stderr}"
            )
        try:
            status, body = request_json(
                "GET", f"http://127.0.0.1:{port}/forge/health"
            )
            if status == 200 and body == {"status": "ok"}:
                return time.monotonic() - started
        except (OSError, urllib.error.URLError, TimeoutError):
            time.sleep(0.05)
    raise TimeoutError("packaged proxy did not become healthy within 30 seconds")


def port_closed(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return False
    except OSError:
        return True


def start_proxy(executable: Path, args: list[str], cwd: Path) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONIOENCODING", None)
    env.pop("PYTHONUTF8", None)
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(
        [str(executable), *args],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **kwargs,
    )


def graceful_stop(process: subprocess.Popen[str], port: int) -> tuple[float, bool]:
    started = time.monotonic()
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.communicate(timeout=20)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("packaged proxy did not exit after graceful signal") from exc
    elapsed = time.monotonic() - started
    return elapsed, process.returncode == 0 and port_closed(port)


def extraction_snapshot() -> set[Path]:
    root = Path(tempfile.gettempdir())
    return {path.resolve() for path in root.glob("_MEI*") if path.is_dir()}


def cli_check(executable: Path, option: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONIOENCODING", None)
    env.pop("PYTHONUTF8", None)
    getencoding = getattr(locale, "getencoding", None)
    encoding = (
        getencoding() if getencoding is not None else locale.getpreferredencoding(False)
    )
    return subprocess.run(
        [str(executable), option], cwd=cwd, env=env,
        capture_output=True, text=True, encoding=encoding,
        check=False, timeout=30,
    )


def run_smoke(executable: Path, form: str, expected_version: str) -> dict[str, Any]:
    executable = executable.resolve()
    with tempfile.TemporaryDirectory(prefix="forge-packaged-smoke-") as raw_cwd:
        cwd = Path(raw_cwd)
        version = cli_check(executable, "--version", cwd)
        help_result = cli_check(executable, "--help", cwd)
        exact_version = version.returncode == 0 and version.stdout.strip() == expected_version
        help_ok = help_result.returncode == 0 and "usage:" in help_result.stdout.lower()

        before_extract = extraction_snapshot()
        with MockBackend("openai") as backend:
            port = reserve_port()
            process = start_proxy(executable, [
                "--backend-url", backend.url, "--model", "mock-model",
                "--port", str(port),
            ], cwd)
            cold_start = wait_for_health(port, process)
            during_extract = extraction_snapshot() - before_extract
            status, body = request_json(
                "POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                {"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]},
            )
            openai_ok = (
                status == 200
                and body["choices"][0]["message"]["content"] == "openai-ok"
                and backend.requests[-1]["path"].endswith("/v1/chat/completions")
                and backend.requests[-1]["body"]["messages"][0]["role"] == "user"
            )
            shutdown_seconds, openai_shutdown = graceful_stop(process, port)

        cleanup = all(not path.exists() for path in during_extract)
        if form == "onedir":
            extraction = {
                "kind": "directory",
                "observed_path": str(executable.parent),
                "cleanup": None,
            }
        else:
            extraction = {
                "kind": "temporary-onefile",
                "observed_path": (
                    str(next(iter(during_extract))) if during_extract else None
                ),
                "cleanup": cleanup and bool(during_extract),
            }

        with MockBackend("anthropic") as backend:
            port = reserve_port()
            process = start_proxy(executable, [
                "--backend-url", backend.url, "--backend", "anthropic",
                "--model", "claude-packaged", "--backend-api-key", "packaged-key",
                "--port", str(port),
            ], cwd)
            wait_for_health(port, process)
            status, body = request_json(
                "POST", f"http://127.0.0.1:{port}/v1/messages",
                {
                    "model": "claude-packaged", "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            anthropic_ok = (
                status == 200
                and body["content"][0]["text"] == "anthropic-ok"
                and backend.requests[-1]["path"].endswith("/v1/messages")
                and backend.requests[-1]["body"]["messages"][0]["role"] == "user"
            )
            _, anthropic_shutdown = graceful_stop(process, port)

    return {
        "runtime_identity": {"version": version.stdout.strip()},
        "cold_start_seconds": round(cold_start, 6),
        "shutdown_seconds": round(shutdown_seconds, 6),
        "extraction": extraction,
        "smoke": {
            "version": exact_version,
            "help": help_ok,
            "health": True,
            "openai": openai_ok,
            "anthropic": anthropic_ok,
            "graceful_shutdown": openai_shutdown and anthropic_shutdown,
            "listener_closed": openai_shutdown and anthropic_shutdown,
            "process_exited": openai_shutdown and anthropic_shutdown,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--form", choices=("onedir", "onefile"), required=True)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.executable, args.form, args.expected_version), indent=2))


if __name__ == "__main__":
    main()
