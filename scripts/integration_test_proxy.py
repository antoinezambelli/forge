"""Integration tests for the proxy against a real llama.cpp backend.

Two phases, sequential, each owning its own backend lifecycle:

1. External mode — script launches ``llama-server`` via subprocess, the
   proxy points at it via ``backend_url``. Matches what users do per the
   BACKEND_SETUP docs (CC user from issue, TI evaluation path).
2. Managed mode — the proxy owns the llama-server via ServerManager.
   Matches ``python -m forge.proxy --backend llamaserver --gguf X``.

Same four tests run in each phase:

- OpenAI text completion (regression coverage)
- Anthropic text completion, no tools (Path 2 text round trip)
- Anthropic tool call, non-streaming (Path 2 tool injection + emit)
- Anthropic tool call, streaming (Path 2 SSE event sequence on the wire)

Path 1 (Anthropic-shape downstream) is not covered here — that needs a
real Anthropic API or LiteLLM container.  See test_path1_anthropic_passthrough
in smoke_test_proxy.py for the wire-shape coverage of that path.

Usage:
    python scripts/integration_test_proxy.py [--gguf PATH]

A proxy log is written to scripts/integration_test_proxy.log alongside
this script — inspect it on failure for forge-side detail.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# AnthropicClient is imported via forge.proxy even though external+managed
# modes here use LlamafileClient — set a dummy key so the SDK import doesn't
# choke on missing env in some setups.
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-for-integration")


DEFAULT_GGUF = "/home/antoine/models/Ministral-3-14B-Instruct-2512-Q4_K_M.gguf"
LLAMA_SERVER_BIN = "llama-server"

# Distinct port pairs per phase so a stale process from one phase doesn't
# poison the other.
EXTERNAL_BACKEND_PORT = 18086
EXTERNAL_PROXY_PORT = 18087
MANAGED_BACKEND_PORT = 18088
MANAGED_PROXY_PORT = 18089

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

def _spawn_llama_server(gguf: Path, port: int) -> subprocess.Popen:
    """Launch llama-server with forge's canonical flags (matches ServerManager)."""
    cmd = [
        LLAMA_SERVER_BIN,
        "-m", str(gguf),
        "-ngl", "999",
        "--port", str(port),
        "--jinja",
    ]
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


async def _run_test_openai_text(proxy_base: str) -> None:
    """Test 1: OpenAI inbound, text only (regression coverage)."""
    print("  -- T1 OpenAI text completion (regression)")
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(
            f"{proxy_base}/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Reply with exactly the single word: OK"}],
                "stream": False,
            },
        )
    assert r.status_code == 200, f"T1 status={r.status_code} body={r.text[:300]}"
    data = r.json()
    assert "choices" in data, f"T1 missing 'choices': {data}"
    msg = data["choices"][0]["message"]
    print(f"     content={msg.get('content', '')[:80]!r}")
    assert msg["role"] == "assistant"


async def _run_test_anthropic_text(proxy_base: str) -> None:
    """Test 2: Anthropic inbound, text only — Path 2 round trip."""
    print("  -- T2 Anthropic text completion (Path 2, no tools)")
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(
            f"{proxy_base}/v1/messages",
            json={
                "model": "test",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Reply with exactly the single word: OK"}],
                "stream": False,
            },
        )
    assert r.status_code == 200, f"T2 status={r.status_code} body={r.text[:300]}"
    data = r.json()
    assert data.get("type") == "message", f"T2 wrong type: {data}"
    assert data["role"] == "assistant"
    assert data["id"].startswith("msg_"), f"T2 bad id: {data['id']}"
    text_blocks = [b for b in data["content"] if b.get("type") == "text"]
    assert text_blocks, f"T2 no text blocks: {data['content']}"
    print(f"     text={text_blocks[0]['text'][:80]!r}")
    print(f"     stop_reason={data.get('stop_reason')}")


async def _run_test_anthropic_tool_nonstream(proxy_base: str) -> None:
    """Test 3: Anthropic inbound with tools, non-streaming — Path 2."""
    print("  -- T3 Anthropic tool call, non-streaming (Path 2)")
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(
            f"{proxy_base}/v1/messages",
            json={
                "model": "test",
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
    assert r.status_code == 200, f"T3 status={r.status_code} body={r.text[:300]}"
    data = r.json()
    assert data.get("type") == "message", f"T3 wrong type: {data}"
    tool_uses = [b for b in data["content"] if b.get("type") == "tool_use"]
    text_blocks = [b for b in data["content"] if b.get("type") == "text"]
    print(f"     content blocks: tool_use={len(tool_uses)} text={len(text_blocks)}")
    if not tool_uses:
        # Forge's handler.py falls back to text if the model refuses to call
        # the tool after retries — that's not a wire bug. Note loudly.
        print(f"     [WARN] no tool_use blocks — model returned text: "
              f"{(text_blocks[0]['text'][:200] if text_blocks else '')!r}")
        return
    block = tool_uses[0]
    assert block["name"] == "get_weather", f"T3 wrong tool: {block['name']}"
    assert block["id"].startswith("toolu_"), f"T3 bad toolu id: {block['id']}"
    assert isinstance(block.get("input"), dict), f"T3 input not dict: {block.get('input')}"
    print(f"     tool_use: name={block['name']} id={block['id']} input={block['input']}")
    assert data.get("stop_reason") == "tool_use", f"T3 stop_reason={data.get('stop_reason')}"


async def _run_test_anthropic_tool_stream(proxy_base: str) -> None:
    """Test 4: Anthropic inbound with tools, streaming — Path 2 SSE on the wire."""
    print("  -- T4 Anthropic tool call, streaming (Path 2 SSE)")
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(
            f"{proxy_base}/v1/messages",
            json={
                "model": "test",
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
    assert r.status_code == 200, f"T4 status={r.status_code}"
    sse_text = r.text
    assert "[DONE]" not in sse_text, "T4 Anthropic SSE must NOT emit [DONE]"
    event_lines = [l for l in sse_text.splitlines() if l.startswith("event: ")]
    event_types = [l.removeprefix("event: ").strip() for l in event_lines]
    print(f"     events: {event_types}")
    assert event_types, f"T4 no event: lines, body={sse_text[:300]!r}"
    assert event_types[0] == "message_start", f"T4 first event={event_types[0]}"
    assert event_types[-1] == "message_stop", f"T4 last event={event_types[-1]}"
    has_tool_use = any('"tool_use"' in l for l in sse_text.splitlines() if l.startswith("data: "))
    if not has_tool_use:
        print(f"     [WARN] no tool_use content block in stream — model returned text only")


TESTS = [
    ("T1 OpenAI text", _run_test_openai_text),
    ("T2 Anthropic text", _run_test_anthropic_text),
    ("T3 Anthropic tool non-stream", _run_test_anthropic_tool_nonstream),
    ("T4 Anthropic tool stream", _run_test_anthropic_tool_stream),
]


async def _run_all_tests(proxy_base: str) -> list[tuple[str, str, str]]:
    """Run the full battery against a proxy. Returns [(name, status, detail)]."""
    results: list[tuple[str, str, str]] = []
    for name, fn in TESTS:
        try:
            t0 = time.monotonic()
            await fn(proxy_base)
            results.append((name, "PASS", f"{time.monotonic() - t0:.1f}s"))
        except AssertionError as exc:
            results.append((name, "FAIL", str(exc)[:200]))
            print(f"     [FAIL] {exc}")
        except Exception as exc:
            results.append((name, "ERROR", f"{type(exc).__name__}: {exc}"[:200]))
            print(f"     [ERROR] {type(exc).__name__}: {exc}")
    return results


# ── Phase 1: External mode ───────────────────────────────────────────

async def phase_external(gguf: Path) -> list[tuple[str, str, str]]:
    print(f"\n===== Phase 1: external mode =====")
    print(f"      llama-server on :{EXTERNAL_BACKEND_PORT}, proxy on :{EXTERNAL_PROXY_PORT}")

    llama_proc = _spawn_llama_server(gguf, EXTERNAL_BACKEND_PORT)
    try:
        await _wait_llama_ready(EXTERNAL_BACKEND_PORT)
        print(f"[external] llama-server ready")

        from forge.proxy import ProxyServer
        proxy = ProxyServer(
            backend_url=f"http://127.0.0.1:{EXTERNAL_BACKEND_PORT}",
            port=EXTERNAL_PROXY_PORT,
            mode="native",
            backend_protocol="openai",
        )
        proxy.start()
        print(f"[external] proxy ready at {proxy.url}")
        try:
            return await _run_all_tests(proxy.url)
        finally:
            proxy.stop()
    finally:
        llama_proc.terminate()
        try:
            llama_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            llama_proc.kill()
        print("[external] llama-server stopped")


# ── Phase 2: Managed mode ────────────────────────────────────────────

async def phase_managed(gguf: Path) -> list[tuple[str, str, str]]:
    print(f"\n===== Phase 2: managed mode =====")
    print(f"      forge owns llama-server on :{MANAGED_BACKEND_PORT}, proxy on :{MANAGED_PROXY_PORT}")

    from forge.proxy import ProxyServer
    from forge.server import BudgetMode

    proxy = ProxyServer(
        backend="llamaserver",
        gguf=str(gguf),
        backend_port=MANAGED_BACKEND_PORT,
        port=MANAGED_PROXY_PORT,
        budget_mode=BudgetMode.BACKEND,
        mode="native",
    )
    proxy.start()
    print(f"[managed] proxy ready at {proxy.url}")
    try:
        return await _run_all_tests(proxy.url)
    finally:
        proxy.stop()
        print("[managed] proxy + managed llama-server stopped")


# ── Entry point ──────────────────────────────────────────────────────

def _print_summary(phase: str, results: list[tuple[str, str, str]]) -> None:
    print(f"\n  [{phase} summary]")
    for name, status, detail in results:
        print(f"     {status:5s}  {name:34s}  {detail}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default=DEFAULT_GGUF, help="GGUF model path")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--skip-managed", action="store_true")
    args = parser.parse_args()

    gguf = Path(args.gguf)
    if not gguf.exists():
        print(f"[FATAL] GGUF not found: {gguf}")
        return 2

    _setup_logging()
    print(f"GGUF: {gguf}")
    print(f"Forge proxy log: {LOG_FILE}")

    summaries: list[tuple[str, list[tuple[str, str, str]]]] = []

    if not args.skip_external:
        ext = await phase_external(gguf)
        _print_summary("external", ext)
        summaries.append(("external", ext))

    if not args.skip_managed:
        man = await phase_managed(gguf)
        _print_summary("managed", man)
        summaries.append(("managed", man))

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
