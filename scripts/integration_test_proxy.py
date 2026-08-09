"""Integration tests for the proxy against real local-model backends.

Up to six phases run sequentially:

1. External mode — script launches ``llama-server`` via subprocess, the
   proxy points at it via ``backend_url``. Matches what users do per the
   BACKEND_SETUP docs.
2. Managed mode — the proxy owns the llama-server via ServerManager.
   Matches ``python -m forge.proxy --backend llamaserver --gguf X``.
3. Generic OpenAI-compatible mode — script launches ``llama-server`` as a
   representative OpenAI-shaped downstream while Forge selects the explicit
   unmanaged ``openai`` profile.
4. Optional external vLLM — ``--vllm-url`` points Forge at a user-managed
   vLLM server. The script owns only the proxy in this phase.
5. External Ollama — the proxy uses Ollama's OpenAI-compatible surface.
6. Managed Ollama — Forge attaches through its native Ollama adapter.

The same Forge-local, metadata, and inference checks run in every enabled phase:

- Forge liveness, initial/current usage, closed namespace, and CORS
- Forwarded metadata fidelity against the backend itself
- Current-context session attribution, replacement, and subagent non-replacement

- OpenAI text completion (regression coverage)
- OpenAI unversioned chat-completions alias
- Anthropic text completion, no tools (Path 2 text round trip)
- Anthropic tool call, non-streaming (Path 2 tool injection + emit)
- Anthropic tool call, streaming (Path 2 SSE event sequence on the wire)
- Anthropic multi-turn tool-result round trip (Path 2)

Every phase also verifies the shutdown behavior it owns. The live assertions
cover transport and Forge-owned state only; model output choices are never
graded.

An Anthropic-shaped downstream is not exercised live here. See the
``anthropic-downstream`` scenario in ``smoke_test_proxy.py`` for deterministic
wire-shape coverage with a mocked backend.

Usage:
    python scripts/integration_test_proxy.py --gguf PATH
        [--server-flags "..."] [--mode {native,prompt}]
        [--skip-external] [--skip-managed] [--skip-llama] [--skip-ollama]
        [--vllm-url URL]

``--gguf`` is required when any llama-server-backed phase is enabled.
Paths may use native Windows or Linux syntax. Ollama phases select an already
installed Gemma-4 E4B Q4 model, falling back to Ministral-3 8B Instruct Q4.

A proxy log is written to scripts/integration_test_proxy.log alongside
this script — inspect it on failure for forge-side detail.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx


LLAMA_SERVER_BIN = "llama-server"
OLLAMA_BIN = "ollama"
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL_PREFERENCE = (
    "gemma4:e4b-it-q4_K_M",
    "ministral-3:8b-instruct-2512-q4_K_M",
)

# Distinct port pairs per phase so a stale process from one phase doesn't
# poison the other.
EXTERNAL_BACKEND_PORT = 18086
EXTERNAL_PROXY_PORT = 18087
MANAGED_BACKEND_PORT = 18088
MANAGED_PROXY_PORT = 18089
# External vLLM is user-managed (we only own the proxy port here).
VLLM_PROXY_PORT = 18091
EXTERNAL_OLLAMA_PROXY_PORT = 18092
MANAGED_OLLAMA_PROXY_PORT = 18093
GENERIC_OPENAI_BACKEND_PORT = 18094
GENERIC_OPENAI_PROXY_PORT = 18095

LOG_FILE = Path(__file__).parent / "integration_test_proxy.log"

# Reasoning models can spend tens of seconds thinking before emitting tool
# calls. Cold first inference is the slowest; subsequent calls are faster.
REQUEST_TIMEOUT = 240.0


# ── Logging ───────────────────────────────────────────────────────────

def _setup_logging() -> None:
    """Pipe forge logs to a file so failure post-mortem has detail."""
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    handler = logging.FileHandler(LOG_FILE)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    ))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


# ── Real-backend helpers ──────────────────────────────────────────────

def _spawn_llama_server(
    gguf: Path, port: int, mode: str = "native", extra_flags: list[str] | None = None,
) -> subprocess.Popen:
    """Launch llama-server with forge's canonical flags (matches ServerManager)."""
    cmd = [
        LLAMA_SERVER_BIN,
        "-m", str(gguf),
        "-ngl", "999",
        "--port", str(port),
    ]
    # Native FC needs the chat template's tool-calling (--jinja). Prompt mode
    # injects the tool surface into the prompt and parses text, so it omits
    # --jinja — matching ServerManager's mode-conditional behavior.
    if mode == "native":
        cmd.append("--jinja")
    if extra_flags:
        cmd.extend(extra_flags)
    print(f"[external] launching: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def _wait_llama_ready(port: int, timeout: float = 180.0) -> None:
    """Poll /props until llama-server responds; matches ServerManager's check."""
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/props"
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
                pass
            await asyncio.sleep(1.0)
    raise RuntimeError(f"llama-server on :{port} did not become healthy in {timeout}s")


def _select_ollama_model() -> str | None:
    """Select the first approved installed Ollama model in preference order."""
    try:
        result = subprocess.run(
            [OLLAMA_BIN, "list"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print(f"[ERROR] {OLLAMA_BIN!r} was not found on PATH")
        return None

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        print(f"[ERROR] ollama list failed ({result.returncode}): {detail}")
        return None

    available: dict[str, str] = {}
    for line in result.stdout.splitlines()[1:]:
        fields = line.split()
        if fields:
            available[fields[0].casefold()] = fields[0]

    for candidate in OLLAMA_MODEL_PREFERENCE:
        selected = available.get(candidate.casefold())
        if selected is not None:
            print(f"Ollama model: {selected}")
            return selected

    expected = " or ".join(OLLAMA_MODEL_PREFERENCE)
    installed = ", ".join(available.values()) or "none"
    print(
        f"[ERROR] No supported Ollama integration model is installed; "
        f"expected {expected}. Installed: {installed}"
    )
    return None


async def _stop_ollama_model(model: str) -> None:
    """Explicitly unload an externally exercised Ollama model from VRAM."""
    process = await asyncio.create_subprocess_exec(
        OLLAMA_BIN,
        "stop",
        model,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        detail = (stderr or stdout).decode(errors="replace").strip()
        raise RuntimeError(
            f"ollama stop {model!r} failed ({process.returncode}): {detail}"
        )
    print(f"[ollama] unloaded {model}")


def _metadata_mount_root(backend_url: str) -> str:
    """Return the public metadata mount implied by a configured backend URL."""
    parsed = urlsplit(backend_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit(parsed._replace(path=path, query="", fragment=""))


async def _discover_vllm_model(backend_url: str) -> str:
    """Read the first served identity for the optional real-vLLM phase."""
    catalog_url = f"{_metadata_mount_root(backend_url)}/v1/models"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(catalog_url)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data") if isinstance(payload, dict) else None
    first = rows[0] if isinstance(rows, list) and rows else None
    model = first.get("id") if isinstance(first, dict) else None
    if not isinstance(model, str) or not model:
        raise RuntimeError(f"vLLM catalog has no served model: {payload!r}")
    return model


async def _wait_tcp_closed(port: int, timeout: float = 10.0) -> None:
    """Wait until a locally owned listening port stops accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _reader, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            return
        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.1)
    raise AssertionError(f"owned port :{port} still accepts connections")


async def _assert_ollama_model_unloaded(model: str, timeout: float = 10.0) -> None:
    """Verify that an integration-owned Ollama model no longer occupies VRAM."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process = await asyncio.create_subprocess_exec(
            OLLAMA_BIN,
            "ps",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            detail = (stderr or stdout).decode(errors="replace").strip()
            raise RuntimeError(f"ollama ps failed ({process.returncode}): {detail}")
        loaded = {
            line.split()[0].casefold()
            for line in stdout.decode(errors="replace").splitlines()[1:]
            if line.split()
        }
        if model.casefold() not in loaded:
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"Ollama model {model!r} remains loaded")


async def _shutdown_result(
    ports: tuple[int, ...],
    *,
    process: subprocess.Popen | None = None,
    ollama_model: str | None = None,
) -> tuple[str, str, str]:
    """Verify phase-owned resources after cleanup and return one result row."""
    started = time.monotonic()
    try:
        for port in ports:
            await _wait_tcp_closed(port)
        if process is not None:
            assert process.poll() is not None, "owned backend process is still running"
        if ollama_model is not None:
            await _assert_ollama_model_unloaded(ollama_model)
    except AssertionError as exc:
        return "L1 Owned lifecycle shutdown", "FAIL", str(exc)[:200]
    except Exception as exc:
        return (
            "L1 Owned lifecycle shutdown",
            "ERROR",
            f"{type(exc).__name__}: {exc}"[:200],
        )
    return (
        "L1 Owned lifecycle shutdown",
        "PASS",
        f"{time.monotonic() - started:.1f}s",
    )


# ── Test case definitions ────────────────────────────────────────────

GET_WEATHER_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}

GET_WEATHER_TOOL_ANTHROPIC = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}

INTEGRATION_SESSION_ID = "forge-integration-session"
LITELLM_SESSION_ID = "forge-integration-litellm-session"
METADATA_TARGETS = (
    "/health?forge_integration=1",
    "/v1/health?forge_integration=1",
    "/v1/models?forge_integration=1",
    "/models?forge_integration=1",
    "/props?forge_integration=1",
)


async def _run_test_forge_local_endpoints(proxy_base: str, _model: str) -> None:
    """Forge-owned routes are local, bounded, and empty before inference."""
    print("  -- F1 Forge-local health, usage, namespace, and CORS")
    async with httpx.AsyncClient(timeout=10.0) as client:
        health = await client.get(f"{proxy_base}/forge/health")
        usage = await client.get(f"{proxy_base}/forge/usage")
        unknown = await client.get(f"{proxy_base}/forge/unknown")
        options = await client.options(f"{proxy_base}/forge/usage")

    assert health.status_code == 200, f"F1 health status={health.status_code}"
    assert health.json() == {"status": "ok"}, f"F1 health body={health.text}"
    assert usage.status_code == 204 and usage.content == b"", (
        f"F1 initial usage status={usage.status_code} body={usage.text!r}"
    )
    assert unknown.status_code == 404, f"F1 unknown status={unknown.status_code}"
    assert options.status_code == 204, f"F1 OPTIONS status={options.status_code}"
    assert options.headers.get("access-control-allow-origin") == "*"


def _normalized_vllm_catalog(content: bytes) -> dict | None:
    """Remove fields that vLLM regenerates for every /v1/models request."""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("object") != "list":
        return None
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        return None
    if not all(isinstance(row, dict) for row in rows):
        return None
    if not any(row.get("owned_by") == "vllm" for row in rows):
        return None

    for row in rows:
        row.pop("created", None)
        permissions = row.get("permission")
        if not isinstance(permissions, list):
            continue
        for permission in permissions:
            if isinstance(permission, dict):
                permission.pop("id", None)
                permission.pop("created", None)
    return payload


async def _run_test_metadata_forwarding(
    proxy_base: str,
    backend_mount_root: str,
) -> None:
    """Compare the public metadata courier with the real backend response."""
    print("  -- F2 Backend metadata forwarding fidelity")
    async with httpx.AsyncClient(timeout=30.0) as client:
        for target in METADATA_TARGETS:
            direct = await client.get(f"{backend_mount_root}{target}")
            proxied = await client.get(f"{proxy_base}{target}")
            assert proxied.status_code == direct.status_code, (
                f"F2 {target} status proxy={proxied.status_code} "
                f"backend={direct.status_code}"
            )
            bodies_match = proxied.content == direct.content
            if not bodies_match and urlsplit(target).path == "/v1/models":
                direct_catalog = _normalized_vllm_catalog(direct.content)
                proxied_catalog = _normalized_vllm_catalog(proxied.content)
                bodies_match = (
                    direct_catalog is not None
                    and proxied_catalog is not None
                    and proxied_catalog == direct_catalog
                )
                if bodies_match:
                    print("     /v1/models: ignored volatile vLLM permission metadata")
            assert bodies_match, (
                f"F2 {target} body differs from backend"
            )
            assert (
                proxied.headers.get("content-type")
                == direct.headers.get("content-type")
            ), f"F2 {target} content-type differs from backend"
            assert proxied.headers.get("access-control-allow-origin") == "*"
            print(f"     {target}: {proxied.status_code} {len(proxied.content)} bytes")


async def _read_forge_usage(
    proxy_base: str,
    expected_session_id: str | None = None,
) -> httpx.Response:
    """Poll briefly because reporting finalization follows response delivery."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{proxy_base}/forge/usage")
        for _ in range(100):
            if response.status_code == 200:
                if expected_session_id is None:
                    return response
                payload = response.json()
                session = payload.get("session")
                if (
                    isinstance(session, dict)
                    and session.get("id") == expected_session_id
                ):
                    return response
            await asyncio.sleep(0.01)
            response = await client.get(f"{proxy_base}/forge/usage")
        return response


async def _run_test_forge_usage(
    proxy_base: str,
    model: str,
    expected_source: str | None,
) -> None:
    """Publish and read the one Forge-local current-context snapshot."""
    print("  -- F3 Forge-local current-context usage")
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        inference = await client.post(
            f"{proxy_base}/v1/chat/completions",
            headers={"X-Claude-Code-Session-Id": INTEGRATION_SESSION_ID},
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": "Reply with exactly the single word: OK",
                }],
                "stream": False,
            },
        )
    assert inference.status_code == 200, (
        f"F3 inference status={inference.status_code} body={inference.text[:300]}"
    )
    effective_model = inference.json().get("model")
    assert effective_model == model, (
        f"F3 effective model={effective_model!r}, expected {model!r}"
    )

    usage_response = await _read_forge_usage(proxy_base, INTEGRATION_SESSION_ID)
    if expected_source is None:
        assert usage_response.status_code == 204 and usage_response.content == b"", (
            f"F3 expected unavailable usage, got {usage_response.status_code} "
            f"{usage_response.text[:300]!r}"
        )
        print("     usage unavailable as expected (no trustworthy denominator)")
        return

    assert usage_response.status_code == 200, (
        f"F3 usage status={usage_response.status_code} body={usage_response.text[:300]}"
    )
    usage = usage_response.json()
    assert isinstance(usage.get("current_usage_tokens"), int)
    assert usage["current_usage_tokens"] > 0
    assert isinstance(usage.get("context_window_tokens"), int)
    assert usage["context_window_tokens"] > usage["current_usage_tokens"]
    assert 0 < usage.get("usage_percent", 0) < 100
    assert usage.get("model") == effective_model
    assert usage.get("context_window_source") == expected_source
    assert usage.get("session") == {
        "id": INTEGRATION_SESSION_ID,
        "source": "claude_code",
    }
    assert isinstance(usage.get("observed_at"), str) and usage["observed_at"].endswith("Z")
    print(
        f"     usage={usage['current_usage_tokens']}/"
        f"{usage['context_window_tokens']} source={expected_source} "
        f"model={effective_model}"
    )

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        subagent = await client.post(
            f"{proxy_base}/v1/chat/completions",
            headers={"X-Claude-Code-Agent-Id": "integration-subagent"},
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": "Reply with exactly the single word: OK",
                }],
                "stream": False,
            },
        )
    assert subagent.status_code == 200, (
        f"F3 subagent status={subagent.status_code} body={subagent.text[:300]}"
    )
    assert subagent.json().get("model") == model
    retained = await _read_forge_usage(proxy_base, INTEGRATION_SESSION_ID)
    assert retained.status_code == 200 and retained.json() == usage, (
        "F3 subagent request replaced the top-level usage snapshot"
    )

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        litellm = await client.post(
            f"{proxy_base}/v1/chat/completions",
            json={
                "model": model,
                "litellm_session_id": LITELLM_SESSION_ID,
                "messages": [{
                    "role": "user",
                    "content": (
                        "This is a second eligible integration request. "
                        "Return any valid assistant response."
                    ),
                }],
                "stream": False,
            },
        )
    assert litellm.status_code == 200, (
        f"F3 LiteLLM inference status={litellm.status_code} "
        f"body={litellm.text[:300]}"
    )
    assert litellm.json().get("model") == model
    replaced_response = await _read_forge_usage(proxy_base, LITELLM_SESSION_ID)
    assert replaced_response.status_code == 200, (
        f"F3 replacement usage status={replaced_response.status_code} "
        f"body={replaced_response.text[:300]}"
    )
    replaced = replaced_response.json()
    assert replaced.get("session") == {
        "id": LITELLM_SESSION_ID,
        "source": "litellm",
    }
    assert replaced.get("model") == model
    assert replaced.get("context_window_source") == expected_source
    assert replaced.get("observed_at") != usage.get("observed_at")
    print("     latest eligible snapshot replaced by LiteLLM session")


async def _run_test_openai_text(proxy_base: str, model: str) -> None:
    """Test 1: OpenAI inbound, text only (regression coverage)."""
    print("  -- T1 OpenAI text completion (regression)")
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(
            f"{proxy_base}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with exactly the single word: OK"}],
                "stream": False,
            },
        )
    assert r.status_code == 200, f"T1 status={r.status_code} body={r.text[:300]}"
    data = r.json()
    assert "choices" in data, f"T1 missing 'choices': {data}"
    assert data.get("model") == model, (
        f"T1 effective model={data.get('model')!r}, expected {model!r}"
    )
    msg = data["choices"][0]["message"]
    print(f"     content={msg.get('content', '')[:80]!r}")
    print(f"     usage={data.get('usage')}")
    assert msg["role"] == "assistant"


async def _run_test_openai_alias(proxy_base: str, model: str) -> None:
    """Test 2: unversioned llama.cpp chat-completions alias."""
    print("  -- T2 OpenAI /chat/completions alias")
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            f"{proxy_base}/chat/completions",
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": "Return any valid assistant response.",
                }],
                "stream": False,
            },
        )
    assert response.status_code == 200, (
        f"T2 status={response.status_code} body={response.text[:300]}"
    )
    data = response.json()
    assert data.get("model") == model, (
        f"T2 effective model={data.get('model')!r}, expected {model!r}"
    )
    choices = data.get("choices")
    assert isinstance(choices, list) and choices, f"T2 invalid choices: {choices!r}"
    assert choices[0].get("message", {}).get("role") == "assistant"


async def _run_test_anthropic_text(proxy_base: str, model: str) -> None:
    """Test 3: Anthropic inbound, text only — Path 2 round trip."""
    print("  -- T3 Anthropic text completion (Path 2, no tools)")
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(
            f"{proxy_base}/v1/messages",
            json={
                "model": model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Reply with exactly the single word: OK"}],
                "stream": False,
            },
        )
    assert r.status_code == 200, f"T3 status={r.status_code} body={r.text[:300]}"
    data = r.json()
    assert data.get("type") == "message", f"T3 wrong type: {data}"
    assert data["role"] == "assistant"
    assert data["id"].startswith("msg_"), f"T3 bad id: {data['id']}"
    assert data.get("model") == model, (
        f"T3 effective model={data.get('model')!r}, expected {model!r}"
    )
    text_blocks = [b for b in data["content"] if b.get("type") == "text"]
    assert text_blocks, f"T3 no text blocks: {data['content']}"
    print(f"     text={text_blocks[0]['text'][:80]!r}")
    print(f"     stop_reason={data.get('stop_reason')}")
    print(f"     usage={data.get('usage')}")


async def _run_test_anthropic_tool_nonstream(proxy_base: str, model: str) -> None:
    """Test 4: Anthropic inbound with tools, non-streaming — Path 2."""
    print("  -- T4 Anthropic tool call, non-streaming (Path 2)")
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(
            f"{proxy_base}/v1/messages",
            json={
                "model": model,
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Use the get_weather tool to check the weather in "
                            "Paris. Call the tool, do not answer in text."
                        ),
                    },
                ],
                "tools": [GET_WEATHER_TOOL_ANTHROPIC],
                "stream": False,
            },
        )
    assert r.status_code == 200, f"T4 status={r.status_code} body={r.text[:300]}"
    data = r.json()
    assert data.get("type") == "message", f"T4 wrong type: {data}"
    assert data.get("model") == model, (
        f"T4 effective model={data.get('model')!r}, expected {model!r}"
    )
    tool_uses = [b for b in data["content"] if b.get("type") == "tool_use"]
    text_blocks = [b for b in data["content"] if b.get("type") == "text"]
    print(f"     content blocks: tool_use={len(tool_uses)} text={len(text_blocks)}")
    assert tool_uses or text_blocks, f"T4 no supported content blocks: {data['content']}"
    if not tool_uses:
        assert data.get("stop_reason") == "end_turn", (
            f"T4 text response stop_reason={data.get('stop_reason')}"
        )
        return
    block = tool_uses[0]
    assert block["name"] == "get_weather", f"T4 wrong tool: {block['name']}"
    assert block["id"].startswith("toolu_"), f"T4 bad toolu id: {block['id']}"
    assert isinstance(block.get("input"), dict), f"T4 input not dict: {block.get('input')}"
    print(f"     tool_use: name={block['name']} id={block['id']} input={block['input']}")
    print(f"     usage={data.get('usage')}")
    assert data.get("stop_reason") == "tool_use", f"T4 stop_reason={data.get('stop_reason')}"
    assert data.get("usage", {}).get("input_tokens", 0) > 0, (
        f"T4 expected non-zero input_tokens from real backend, got {data.get('usage')}"
    )


async def _run_test_anthropic_tool_stream(proxy_base: str, model: str) -> None:
    """Test 5: Anthropic inbound with tools, streaming — Path 2 SSE on the wire."""
    print("  -- T5 Anthropic tool call, streaming (Path 2 SSE)")
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(
            f"{proxy_base}/v1/messages",
            json={
                "model": model,
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Use the get_weather tool to check the weather in "
                            "Paris. Call the tool, do not answer in text."
                        ),
                    },
                ],
                "tools": [GET_WEATHER_TOOL_ANTHROPIC],
                "stream": True,
            },
        )
    assert r.status_code == 200, f"T5 status={r.status_code}"
    sse_text = r.text
    assert "[DONE]" not in sse_text, "T5 Anthropic SSE must NOT emit [DONE]"
    event_lines = [line for line in sse_text.splitlines() if line.startswith("event: ")]
    event_types = [line.removeprefix("event: ").strip() for line in event_lines]
    print(f"     events: {event_types}")
    assert event_types, f"T5 no event: lines, body={sse_text[:300]!r}"
    assert event_types[0] == "message_start", f"T5 first event={event_types[0]}"
    assert event_types[-1] == "message_stop", f"T5 last event={event_types[-1]}"
    data_lines = [
        json.loads(line.removeprefix("data: "))
        for line in sse_text.splitlines()
        if line.startswith("data: ")
    ]
    message_start = next(
        (event for event in data_lines if event.get("type") == "message_start"),
        None,
    )
    assert isinstance(message_start, dict), "T5 missing message_start data"
    assert message_start.get("message", {}).get("model") == model, (
        f"T5 effective model={message_start.get('message', {}).get('model')!r}, "
        f"expected {model!r}"
    )


async def _run_test_anthropic_tool_multiturn(proxy_base: str, model: str) -> None:
    """Test 6: Anthropic tool history and result survive a real round trip."""
    print("  -- T6 Anthropic multi-turn tool_result (Path 2 round trip)")
    user_msg = {
        "role": "user",
        "content": (
            "What's the weather in Paris? Use the get_weather tool, then "
            "tell me the result."
        ),
    }
    assistant_msg = {
        "role": "assistant",
        "content": [{
            "type": "tool_use",
            "id": "toolu_forge_integration",
            "name": "get_weather",
            "input": {"city": "Paris"},
        }],
    }
    tool_result_msg = {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "toolu_forge_integration",
            "content": "Paris: 18°C, sunny, light wind from the west.",
        }],
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            f"{proxy_base}/v1/messages",
            json={
                "model": model, "max_tokens": 512,
                "messages": [user_msg, assistant_msg, tool_result_msg],
                "tools": [GET_WEATHER_TOOL_ANTHROPIC],
                "stream": False,
            },
        )
    assert response.status_code == 200, (
        f"T6 status={response.status_code} {response.text[:200]}"
    )
    data = response.json()
    assert data.get("type") == "message" and data.get("role") == "assistant"
    assert data.get("model") == model, (
        f"T6 effective model={data.get('model')!r}, expected {model!r}"
    )
    blocks = data.get("content")
    assert isinstance(blocks, list) and blocks, f"T6 invalid content: {blocks!r}"
    block_types = [block.get("type") for block in blocks if isinstance(block, dict)]
    assert len(block_types) == len(blocks)
    assert all(block_type in {"text", "tool_use"} for block_type in block_types)
    assert isinstance(data.get("stop_reason"), str)
    usage = data.get("usage")
    assert isinstance(usage, dict) and usage.get("input_tokens", 0) > 0
    print(f"     blocks={block_types} stop_reason={data['stop_reason']} usage={usage}")


TESTS = [
    ("T1 OpenAI text", _run_test_openai_text),
    ("T2 OpenAI alias", _run_test_openai_alias),
    ("T3 Anthropic text", _run_test_anthropic_text),
    ("T4 Anthropic tool non-stream", _run_test_anthropic_tool_nonstream),
    ("T5 Anthropic tool stream", _run_test_anthropic_tool_stream),
    ("T6 Anthropic tool_result round trip", _run_test_anthropic_tool_multiturn),
]


async def _run_all_tests(
    proxy_base: str,
    backend_mount_root: str,
    model: str,
    expected_usage_source: str | None,
) -> list[tuple[str, str, str]]:
    """Run the full battery against a proxy. Returns [(name, status, detail)]."""
    results: list[tuple[str, str, str]] = []
    cases = [
        (
            "F1 Forge-local routes",
            _run_test_forge_local_endpoints,
            (proxy_base, model),
        ),
        (
            "F2 Metadata forwarding",
            _run_test_metadata_forwarding,
            (proxy_base, backend_mount_root),
        ),
        (
            "F3 Forge-local usage",
            _run_test_forge_usage,
            (proxy_base, model, expected_usage_source),
        ),
        *[(name, fn, (proxy_base, model)) for name, fn in TESTS],
    ]
    for name, fn, args in cases:
        try:
            t0 = time.monotonic()
            await fn(*args)
            results.append((name, "PASS", f"{time.monotonic() - t0:.1f}s"))
        except AssertionError as exc:
            results.append((name, "FAIL", str(exc)[:200]))
            print(f"     [FAIL] {exc}")
        except Exception as exc:
            results.append((name, "ERROR", f"{type(exc).__name__}: {exc}"[:200]))
            print(f"     [ERROR] {type(exc).__name__}: {exc}")
    return results


# ── Phase 1: External mode ───────────────────────────────────────────

async def phase_external(
    gguf: Path, mode: str = "native", extra_flags: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    print(f"\n===== Phase 1: external mode (fc={mode}) =====")
    print(f"      llama-server on :{EXTERNAL_BACKEND_PORT}, proxy on :{EXTERNAL_PROXY_PORT}")

    backend_root = f"http://127.0.0.1:{EXTERNAL_BACKEND_PORT}"
    llama_proc = _spawn_llama_server(gguf, EXTERNAL_BACKEND_PORT, mode, extra_flags)
    try:
        await _wait_llama_ready(EXTERNAL_BACKEND_PORT)
        print("[external] llama-server ready")

        from forge.proxy import ProxyServer
        proxy = ProxyServer(
            backend_url=backend_root,
            backend="llamaserver",
            model=gguf.stem,
            port=EXTERNAL_PROXY_PORT,
            backend_capability=mode,
        )
        proxy.start()
        print(f"[external] proxy ready at {proxy.url}")
        try:
            results = await _run_all_tests(
                proxy.url, backend_root, gguf.stem, "backend_metadata",
            )
        finally:
            proxy.stop()
    finally:
        llama_proc.terminate()
        try:
            llama_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            llama_proc.kill()
            llama_proc.wait(timeout=10)
        print("[external] llama-server stopped")
    results.append(await _shutdown_result(
        (EXTERNAL_PROXY_PORT, EXTERNAL_BACKEND_PORT),
        process=llama_proc,
    ))
    return results


# ── Phase 2: Managed mode ────────────────────────────────────────────

async def phase_managed(
    gguf: Path, mode: str = "native", extra_flags: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    print(f"\n===== Phase 2: managed mode (fc={mode}) =====")
    print(f"      forge owns llama-server on :{MANAGED_BACKEND_PORT}, proxy on :{MANAGED_PROXY_PORT}")

    from forge.proxy import ProxyServer
    from forge.server import BudgetMode

    proxy = ProxyServer(
        backend="llamaserver",
        gguf=str(gguf),
        backend_port=MANAGED_BACKEND_PORT,
        port=MANAGED_PROXY_PORT,
        budget_mode=BudgetMode.BACKEND,
        backend_capability=mode,
        extra_flags=extra_flags,
    )
    proxy.start()
    print(f"[managed] proxy ready at {proxy.url}")
    try:
        results = await _run_all_tests(
            proxy.url,
            f"http://127.0.0.1:{MANAGED_BACKEND_PORT}",
            gguf.stem,
            "managed_backend",
        )
    finally:
        proxy.stop()
        print("[managed] proxy + managed llama-server stopped")
    results.append(await _shutdown_result(
        (MANAGED_PROXY_PORT, MANAGED_BACKEND_PORT),
    ))
    return results


# ── Phase 3: Generic OpenAI-compatible profile ──────────────────────

async def phase_generic_openai(
    gguf: Path, mode: str = "native", extra_flags: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Exercise the generic unmanaged OpenAI profile against llama-server."""
    print(f"\n===== Phase 3: generic OpenAI profile (fc={mode}) =====")
    print(
        f"      llama-server on :{GENERIC_OPENAI_BACKEND_PORT}, "
        f"proxy on :{GENERIC_OPENAI_PROXY_PORT}"
    )

    backend_root = f"http://127.0.0.1:{GENERIC_OPENAI_BACKEND_PORT}"
    llama_proc = _spawn_llama_server(
        gguf, GENERIC_OPENAI_BACKEND_PORT, mode, extra_flags,
    )
    try:
        await _wait_llama_ready(GENERIC_OPENAI_BACKEND_PORT)
        print("[openai] llama-server ready")

        from forge.proxy import ProxyServer
        proxy = ProxyServer(
            backend_url=backend_root,
            backend="openai",
            model=gguf.stem,
            port=GENERIC_OPENAI_PROXY_PORT,
            backend_capability=mode,
        )
        proxy.start()
        print(f"[openai] proxy ready at {proxy.url}")
        try:
            # The generic profile intentionally interprets no backend-specific
            # context metadata, so reporting remains unavailable without an
            # operator-supplied denominator.
            results = await _run_all_tests(
                proxy.url, backend_root, gguf.stem, None,
            )
        finally:
            proxy.stop()
    finally:
        llama_proc.terminate()
        try:
            llama_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            llama_proc.kill()
            llama_proc.wait(timeout=10)
        print("[openai] llama-server stopped")
    results.append(await _shutdown_result(
        (GENERIC_OPENAI_PROXY_PORT, GENERIC_OPENAI_BACKEND_PORT),
        process=llama_proc,
    ))
    return results


# ── Phase 4: External vLLM (opt-in) ──────────────────────────────────

async def phase_external_vllm(vllm_url: str) -> list[tuple[str, str, str]]:
    """Run the live battery against a user-managed vLLM server.

    External mode only — vLLM is not spawned/torn down here. The same
    protocol-translation tests apply (the proxy layer is backend-agnostic);
    this exercises VLLMClient + served-model-name discovery against a real
    vLLM server. Start vLLM with ``--enable-auto-tool-choice
    --tool-call-parser <name>`` (and ``--reasoning-parser`` for thinking
    models) so the tool tests (T4–T6) have a native tool surface.
    """
    print("\n===== Phase 4: external vLLM (fc=native) =====")
    print(f"      user-managed vLLM at {vllm_url}, proxy on :{VLLM_PROXY_PORT}")
    backend_root = _metadata_mount_root(vllm_url)
    model = await _discover_vllm_model(vllm_url)
    print(f"      served model={model}")

    from forge.proxy import ProxyServer
    proxy = ProxyServer(
        backend_url=vllm_url,
        backend="vllm",
        port=VLLM_PROXY_PORT,
        backend_capability="native",
    )
    proxy.start()
    print(f"[vllm] proxy ready at {proxy.url}")
    try:
        results = await _run_all_tests(
            proxy.url, backend_root, model, "backend_metadata",
        )
    finally:
        proxy.stop()
        print("[vllm] proxy stopped (vLLM server left running — user-managed)")
    results.append(await _shutdown_result((VLLM_PROXY_PORT,)))
    return results


# ── Phase 5: External Ollama ─────────────────────────────────

async def phase_external_ollama(model: str) -> list[tuple[str, str, str]]:
    """Exercise Ollama's OpenAI-compatible surface as an unmanaged backend."""
    print("\n===== Phase 5: external Ollama (OpenAI-compatible) =====")
    print(
        f"      Ollama at {OLLAMA_URL}, model={model}, "
        f"proxy on :{EXTERNAL_OLLAMA_PROXY_PORT}"
    )

    from forge.proxy import ProxyServer
    proxy = ProxyServer(
        backend_url=OLLAMA_URL,
        backend="ollama",
        model=model,
        port=EXTERNAL_OLLAMA_PROXY_PORT,
        backend_capability="native",
    )
    proxy.start()
    print(f"[ollama-external] proxy ready at {proxy.url}")
    try:
        # External Ollama has no trustworthy context-window source unless the
        # operator supplies one, so the post-inference /forge/usage state is 204.
        results = await _run_all_tests(proxy.url, OLLAMA_URL, model, None)
    finally:
        proxy.stop()
        await _stop_ollama_model(model)
        print("[ollama-external] proxy stopped")
    results.append(await _shutdown_result(
        (EXTERNAL_OLLAMA_PROXY_PORT,),
        ollama_model=model,
    ))
    return results


# ── Phase 6: Managed Ollama ──────────────────────────────────

async def phase_managed_ollama(model: str) -> list[tuple[str, str, str]]:
    """Exercise Forge's native attached-daemon Ollama profile."""
    print("\n===== Phase 6: managed Ollama (native adapter) =====")
    print(f"      model={model}, proxy on :{MANAGED_OLLAMA_PROXY_PORT}")

    from forge.proxy import ProxyServer
    proxy = ProxyServer(
        backend="ollama",
        model=model,
        port=MANAGED_OLLAMA_PROXY_PORT,
        backend_capability="native",
    )
    proxy.start()
    print(f"[ollama-managed] proxy ready at {proxy.url}")
    try:
        results = await _run_all_tests(
            proxy.url, OLLAMA_URL, model, "managed_backend",
        )
    finally:
        # Managed Proxy shutdown owns ``ollama stop`` for the attached model.
        proxy.stop()
        print("[ollama-managed] proxy stopped and model unloaded")
    results.append(await _shutdown_result(
        (MANAGED_OLLAMA_PROXY_PORT,),
        ollama_model=model,
    ))
    return results


# ── Entry point ──────────────────────────────────────────────────────

def _print_summary(phase: str, results: list[tuple[str, str, str]]) -> None:
    print(f"\n  [{phase} summary]")
    for name, status, detail in results:
        print(f"     {status:5s}  {name:34s}  {detail}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gguf",
        type=Path,
        default=None,
        help=(
            "GGUF model path for llama-server phases. Required unless those "
            "phases are skipped; quote paths containing spaces."
        ),
    )
    parser.add_argument(
        "--server-flags", default=None,
        help="Extra llama-server flags as a single string, e.g. "
             "'--no-mmap -fa 1 --cache-type-k q8_0 --cache-type-v q8_0 -c 32768'. "
             "Threaded into external spawn and managed ServerManager.",
    )
    parser.add_argument(
        "--mode", choices=["native", "prompt"], default="native",
        help="Function-calling mode for the proxy + backend (default: native). "
             "'prompt' exercises forge's prompt-injection FC path.",
    )
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help=(
            "Skip external llama-server, generic OpenAI, and external "
            "Ollama phases."
        ),
    )
    parser.add_argument(
        "--skip-managed",
        action="store_true",
        help="Skip managed llama-server and managed Ollama phases.",
    )
    parser.add_argument(
        "--skip-llama",
        action="store_true",
        help=(
            "Skip specialized and generic llama-server-backed phases; "
            "--gguf is then unnecessary."
        ),
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip Ollama model discovery and both Ollama phases.",
    )
    parser.add_argument(
        "--vllm-url", default=None,
        help="Run an extra external-mode phase against a user-managed vLLM "
             "server at this URL (e.g. http://localhost:8000). Start vLLM with "
             "--enable-auto-tool-choice --tool-call-parser <name> for the tool "
             "tests. Skipped if not provided. The --gguf flag is ignored for "
             "this phase.",
    )
    args = parser.parse_args()

    needs_gguf = (
        not args.skip_llama
        and (not args.skip_external or not args.skip_managed)
    )
    gguf = args.gguf.expanduser() if args.gguf is not None else None
    if needs_gguf and gguf is None:
        parser.error(
            "--gguf PATH is required unless llama-server phases are skipped"
        )
    if needs_gguf and not gguf.is_file():
        parser.error(f"GGUF is not a readable file: {gguf}")

    extra_flags = args.server_flags.split() if args.server_flags else None

    _setup_logging()
    if gguf is not None:
        print(f"GGUF: {gguf}")
    print(f"FC mode: {args.mode}")
    if extra_flags:
        print(f"Extra server flags: {extra_flags}")
    print(f"Forge proxy log: {LOG_FILE}")

    summaries: list[tuple[str, list[tuple[str, str, str]]]] = []

    if not args.skip_llama and not args.skip_external:
        assert gguf is not None
        ext = await phase_external(gguf, args.mode, extra_flags)
        _print_summary("external", ext)
        summaries.append(("external", ext))

    if not args.skip_llama and not args.skip_managed:
        assert gguf is not None
        man = await phase_managed(gguf, args.mode, extra_flags)
        _print_summary("managed", man)
        summaries.append(("managed", man))

    if not args.skip_llama and not args.skip_external:
        assert gguf is not None
        generic = await phase_generic_openai(gguf, args.mode, extra_flags)
        _print_summary("openai-generic", generic)
        summaries.append(("openai-generic", generic))

    if args.vllm_url:
        vll = await phase_external_vllm(args.vllm_url)
        _print_summary("vllm-external", vll)
        summaries.append(("vllm-external", vll))

    run_ollama = (
        not args.skip_ollama
        and (not args.skip_external or not args.skip_managed)
    )
    if run_ollama:
        ollama_model = _select_ollama_model()
        if ollama_model is None:
            missing = [(
                "Supported Ollama model",
                "ERROR",
                "Neither Gemma-4 E4B Q4 nor Ministral-3 8B Instruct Q4 is installed",
            )]
            _print_summary("ollama-discovery", missing)
            summaries.append(("ollama-discovery", missing))
        else:
            if not args.skip_external:
                ollama_external = await phase_external_ollama(ollama_model)
                _print_summary("ollama-external", ollama_external)
                summaries.append(("ollama-external", ollama_external))
            if not args.skip_managed:
                ollama_managed = await phase_managed_ollama(ollama_model)
                _print_summary("ollama-managed", ollama_managed)
                summaries.append(("ollama-managed", ollama_managed))

    print("\n===== Final =====")
    any_fail = False
    for phase, results in summaries:
        passed = sum(1 for _, s, _ in results if s == "PASS")
        total = len(results)
        if any(s != "PASS" for _, s, _ in results):
            any_fail = True
        print(f"  {phase}: {passed}/{total} passed")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
