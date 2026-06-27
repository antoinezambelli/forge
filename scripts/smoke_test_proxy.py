"""Smoke tests for the proxy.

Three scenarios, each with its own mock backend + proxy on distinct ports:

1. ``test_openai()`` — OpenAI inbound → OpenAI backend.  Original smoke
   coverage; proves the existing OpenAI path didn't regress.

2. ``test_path2_anthropic_to_openai()`` — Anthropic inbound (``/v1/messages``)
   → OpenAI backend.  Exercises Anthropic→OpenAI translation on inbound,
   OpenAI→Anthropic emission on outbound, and the Anthropic SSE wire
   format (``event:`` lines, no ``[DONE]`` terminator).

3. ``test_path1_anthropic_passthrough()`` — Anthropic inbound → Anthropic
   backend.  Mock records the request body it received from forge's
   AnthropicClient and the test asserts ``cache_control`` survived
   end-to-end (the ADR-015 passthrough invariant).

Usage: python scripts/smoke_test_proxy.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import httpx

# The Anthropic SDK requires an API key at construction time even when
# ``base_url`` retargets it at a local mock.  Set a dummy value before
# any forge import so AnthropicClient doesn't trip on missing env.
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-for-smoke")


# ── HTTP helpers ──────────────────────────────────────────────────────

async def _read_request(
    reader: asyncio.StreamReader,
) -> tuple[str, dict[str, str], bytes]:
    """Read an HTTP request: returns (path, headers, body)."""
    request_line = await reader.readline()
    path = request_line.decode().split(" ")[1] if request_line else ""
    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        decoded = line.decode().strip()
        if not decoded:
            break
        if ":" in decoded:
            k, v = decoded.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    content_length = int(headers.get("content-length", "0"))
    body = b""
    if content_length > 0:
        body = await reader.readexactly(content_length)
    return path, headers, body


def _write_json_response(body: str) -> bytes:
    return _write_response(200, "OK", body)


def _write_response(status: int, reason: str, body: str) -> bytes:
    # Connection: close — these mocks serve one request per connection and close
    # it, so tell the client not to pool it. The deferred-discovery flow makes
    # two backend hits (/props then /v1/chat/completions); without this, httpx
    # may reuse the just-closed keep-alive connection and the second hit flakes.
    return (
        f"HTTP/1.1 {status} {reason}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
        f"{body}"
    ).encode()


# Credential a gated backend (test 4) requires — mirrors `llama-server --api-key`.
GATED_KEY = "TESTKEY"


# ── Mock backends ─────────────────────────────────────────────────────

async def openai_mock_backend(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
) -> None:
    """OpenAI-shape mock — /props for context length + /v1/chat/completions."""
    path, _headers, _body = await _read_request(reader)

    if "/props" in path:
        body = json.dumps({"default_generation_settings": {"n_ctx": 8192}})
    else:
        body = json.dumps({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": "mock",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_mock1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    writer.write(_write_json_response(body))
    await writer.drain()
    writer.close()
    await writer.wait_closed()


async def gated_openai_mock_backend(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
) -> None:
    """OpenAI-shape mock that GATES on ``Authorization: Bearer TESTKEY``.

    Mirrors ``llama-server --api-key``: a missing/wrong credential gets 401 on
    *every* path — crucially including the ``/props`` context-length probe. This
    lets the deferred-discovery path (finding #2) be exercised without a real
    gated backend: the proxy can't probe at startup (it has no credential then),
    so it must defer to the first request and use that request's inbound key.
    """
    path, headers, _body = await _read_request(reader)

    if headers.get("authorization") != f"Bearer {GATED_KEY}":
        body = json.dumps({"error": {"message": "missing/invalid api key", "type": "auth"}})
        writer.write(_write_response(401, "Unauthorized", body))
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        return

    if "/props" in path:
        body = json.dumps({"default_generation_settings": {"n_ctx": 8192}})
    else:
        body = json.dumps({
            "id": "chatcmpl-gated",
            "object": "chat.completion",
            "model": "mock",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "OK"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        })

    writer.write(_write_json_response(body))
    await writer.drain()
    writer.close()
    await writer.wait_closed()


# Captured request bodies from the Anthropic mock, used by path-1 assertions.
_anthropic_mock_seen: list[dict[str, Any]] = []


async def anthropic_mock_backend(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
) -> None:
    """Anthropic-shape mock — records inbound bodies, returns a tool_use message.

    The Anthropic SDK (used by forge's AnthropicClient in path 1) calls
    ``POST {base_url}/v1/messages``.  Response must look like an Anthropic
    Message object or the SDK will fail to parse.
    """
    path, _headers, body_bytes = await _read_request(reader)

    if body_bytes:
        try:
            _anthropic_mock_seen.append(json.loads(body_bytes))
        except json.JSONDecodeError:
            _anthropic_mock_seen.append({"_raw": body_bytes.decode(errors="replace")})

    if "/v1/messages" in path:
        body = json.dumps({
            "id": "msg_anthropic_mock_0001",
            "type": "message",
            "role": "assistant",
            "model": "claude-mock",
            "content": [{
                "type": "tool_use",
                "id": "toolu_mock_0001",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 6},
        })
    else:
        body = json.dumps({"error": {"type": "not_found", "message": path}})

    writer.write(_write_json_response(body))
    await writer.drain()
    writer.close()
    await writer.wait_closed()


# ── Proxy lifecycle helper ────────────────────────────────────────────

def _start_proxy(**kwargs: Any) -> Any:
    """Construct + start a ProxyServer, return it (blocks until ready)."""
    from forge.proxy import ProxyServer
    proxy = ProxyServer(**kwargs)
    proxy.start()
    return proxy


# ── Test 1: OpenAI end-to-end (regression coverage) ──────────────────

async def test_openai() -> None:
    print("\n=== test_openai (OpenAI inbound → OpenAI backend) ===")
    mock = await asyncio.start_server(openai_mock_backend, "127.0.0.1", 18080)
    proxy = _start_proxy(
        backend_url="http://127.0.0.1:18080",
        port=18081,
        budget_tokens=8192,
    )
    print("[setup] mock=:18080 proxy=:18081")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            health = await client.get("http://127.0.0.1:18081/health")
            assert health.status_code == 200, f"health: {health.status_code}"

            # /v1/models reports the client's real identity, not a hardcoded
            # stub. External llama.cpp mode with no --model → the "default"
            # placeholder (forge genuinely doesn't know the served name).
            models = await client.get("http://127.0.0.1:18081/v1/models")
            assert models.status_code == 200, f"models: {models.status_code}"
            mdata = models.json()
            assert mdata["object"] == "list", mdata
            assert mdata["data"][0]["id"] == "default", mdata["data"]
            assert mdata["data"][0]["id"] != "forge", "regressed to hardcoded stub"
            print("[ok] /v1/models reports client identity (default placeholder)")

            resp = await client.post(
                "http://127.0.0.1:18081/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [
                        {"role": "system", "content": "You are a weather assistant."},
                        {"role": "user", "content": "What's the weather in Paris?"},
                    ],
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }],
                    "stream": False,
                },
            )
            assert resp.status_code == 200, resp.status_code
            data = resp.json()
            choice = data["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            tc = choice["message"]["tool_calls"][0]
            assert tc["function"]["name"] == "get_weather"
            args = json.loads(tc["function"]["arguments"])
            assert args["city"] == "Paris"
            print("[ok] non-streaming tool call")

            resp_stream = await client.post(
                "http://127.0.0.1:18081/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }],
                    "stream": True,
                },
            )
            body = resp_stream.text
            assert any("[DONE]" in l for l in body.split("\n")), "openai SSE missing [DONE]"
            print("[ok] streaming with [DONE] terminator")

    finally:
        proxy.stop()
        mock.close()
        await mock.wait_closed()


# ── Test 1b: Gated passthrough → deferred discovery (finding #2) ─────

async def test_gated_passthrough_deferred_discovery() -> None:
    print("\n=== test_gated_passthrough_deferred_discovery (deferred probe vs gated backend) ===")
    mock = await asyncio.start_server(gated_openai_mock_backend, "127.0.0.1", 18090)
    # Pure passthrough: no --backend-api-key, no --budget-tokens → discovery is
    # deferred to the first request. Against a GATED backend this used to crash
    # at startup (the unauthenticated /props probe); now the proxy must just start.
    proxy = _start_proxy(backend_url="http://127.0.0.1:18090", port=18091)
    print("[setup] gated mock=:18090 proxy=:18091 (no backend_api_key, no budget_tokens)")
    print("[ok] proxy started against a gated backend (deferred — no startup probe/crash)")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Negative: no inbound credential → the deferred /props probe is
            # unauthenticated → backend 401 → clean 401, discovery NOT latched.
            r0 = await client.post(
                "http://127.0.0.1:18091/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
            assert r0.status_code == 401, (
                f"no-cred: expected 401, got {r0.status_code} {r0.text[:200]}"
            )
            print("[ok] no-credential first request → 401 (probe rejected, not latched)")

            # Positive: same request WITH the credential → the deferred probe
            # authenticates with the inbound key, context is discovered, request
            # completes. Also proves no-latch-on-failure: the prior 401 didn't
            # poison discovery — this retry succeeds.
            r1 = await client.post(
                "http://127.0.0.1:18091/v1/chat/completions",
                headers={"Authorization": f"Bearer {GATED_KEY}"},
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
            assert r1.status_code == 200, (
                f"with-cred: expected 200, got {r1.status_code} {r1.text[:200]}"
            )
            assert r1.json()["choices"][0]["message"]["content"] == "OK"
            print("[ok] credentialed request → deferred discovery succeeded, 200 "
                  "(retry after the 401 works → no-latch-on-failure)")

    finally:
        proxy.stop()
        mock.close()
        await mock.wait_closed()


# ── Test 2: Path 2 (Anthropic inbound → OpenAI backend) ──────────────

async def test_path2_anthropic_to_openai() -> None:
    print("\n=== test_path2_anthropic_to_openai (Anthropic inbound → OpenAI backend) ===")
    mock = await asyncio.start_server(openai_mock_backend, "127.0.0.1", 18082)
    proxy = _start_proxy(
        backend_url="http://127.0.0.1:18082",
        port=18083,
        budget_tokens=8192,
        backend_protocol="openai",  # default but be explicit
    )
    print("[setup] mock=:18082 proxy=:18083 backend_protocol=openai")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # --- Non-streaming Anthropic request ---
            resp = await client.post(
                "http://127.0.0.1:18083/v1/messages",
                json={
                    "model": "test-anthropic",
                    "max_tokens": 1024,
                    "system": "You are a weather assistant.",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "What's the weather in Paris?",
                                    "cache_control": {"type": "ephemeral"},
                                },
                            ],
                        },
                    ],
                    "tools": [{
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }],
                    "tool_choice": {"type": "auto"},
                    "stream": False,
                },
            )
            assert resp.status_code == 200, f"path-2 non-stream: {resp.status_code} {resp.text}"
            data = resp.json()
            print(f"[response] {json.dumps(data, indent=2)}")

            # Anthropic-shape assertions
            assert data["type"] == "message", f"type: {data.get('type')}"
            assert data["role"] == "assistant"
            assert data["id"].startswith("msg_"), f"id: {data['id']}"
            assert data["stop_reason"] == "tool_use", f"stop_reason: {data['stop_reason']}"
            tool_use_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
            assert len(tool_use_blocks) == 1, f"tool_use blocks: {len(tool_use_blocks)}"
            block = tool_use_blocks[0]
            assert block["name"] == "get_weather"
            assert block["input"] == {"city": "Paris"}
            assert block["id"].startswith("toolu_"), f"toolu id: {block['id']}"
            print("[ok] non-streaming: Anthropic-shape response, tool_use block correct")

            # --- Streaming Anthropic request ---
            resp_stream = await client.post(
                "http://127.0.0.1:18083/v1/messages",
                json={
                    "model": "test-anthropic",
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "user", "content": "Weather in Paris?"},
                    ],
                    "tools": [{
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }],
                    "stream": True,
                },
            )
            assert resp_stream.status_code == 200, f"path-2 stream: {resp_stream.status_code}"
            sse_text = resp_stream.text
            print(f"[stream raw] {sse_text!r}")

            assert "[DONE]" not in sse_text, "Anthropic SSE must NOT emit [DONE]"
            event_lines = [l for l in sse_text.splitlines() if l.startswith("event: ")]
            data_lines = [l for l in sse_text.splitlines() if l.startswith("data: ")]
            event_types = [l.removeprefix("event: ").strip() for l in event_lines]
            assert event_types[0] == "message_start", f"first event: {event_types[0]}"
            assert event_types[-1] == "message_stop", f"last event: {event_types[-1]}"
            assert "message_delta" in event_types, "missing message_delta"
            assert any("tool_use" in l for l in data_lines), "no tool_use block in stream"
            print(f"[ok] streaming: {len(event_types)} events, sequence {event_types[0]}…{event_types[-1]}, no [DONE]")

    finally:
        proxy.stop()
        mock.close()
        await mock.wait_closed()


# ── Test 3: Path 1 (Anthropic inbound → Anthropic backend) ───────────

async def test_path1_anthropic_passthrough() -> None:
    print("\n=== test_path1_anthropic_passthrough (Anthropic inbound → Anthropic backend) ===")
    _anthropic_mock_seen.clear()
    mock = await asyncio.start_server(anthropic_mock_backend, "127.0.0.1", 18084)
    proxy = _start_proxy(
        backend_url="http://127.0.0.1:18084",
        port=18085,
        budget_tokens=8192,
        backend_protocol="anthropic",
        model="claude-mock",
    )
    print("[setup] mock=:18084 proxy=:18085 backend_protocol=anthropic")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # /v1/models surfaces the configured model identity (not "forge").
            # This proxy was constructed with model="claude-mock".
            models = await client.get("http://127.0.0.1:18085/v1/models")
            assert models.status_code == 200, f"models: {models.status_code}"
            mdata = models.json()
            assert mdata["data"][0]["id"] == "claude-mock", mdata["data"]
            print("[ok] /v1/models reports configured model 'claude-mock'")

            cache_marker = {"type": "ephemeral"}
            resp = await client.post(
                "http://127.0.0.1:18085/v1/messages",
                # v0.8.0 one-credential rule: this proxy is pure passthrough (no
                # --backend-api-key), so the request must carry a credential —
                # forge relocates this single inbound x-api-key to the Anthropic
                # backend's native slot. Without it forge fails loud (401), which
                # is correct: you can't reach an Anthropic backend with no key.
                headers={"x-api-key": "smoke-anthropic-key"},
                json={
                    "model": "claude-mock",
                    "max_tokens": 1024,
                    "system": [
                        {
                            "type": "text",
                            "text": "You are a weather assistant.",
                            "cache_control": cache_marker,
                        },
                    ],
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Pretend this is a giant cached preamble.",
                                    "cache_control": cache_marker,
                                },
                                {"type": "text", "text": "What's the weather in Paris?"},
                            ],
                        },
                    ],
                    "tools": [{
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }],
                    "stream": False,
                },
            )
            assert resp.status_code == 200, f"path-1 non-stream: {resp.status_code} {resp.text}"
            data = resp.json()
            print(f"[response] {json.dumps(data, indent=2)}")

            # Anthropic-shape on the response leg
            assert data["type"] == "message"
            assert data["role"] == "assistant"
            assert data["id"].startswith("msg_")
            tool_use_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
            assert len(tool_use_blocks) == 1, f"tool_use blocks: {len(tool_use_blocks)}"
            assert tool_use_blocks[0]["name"] == "get_weather"
            assert tool_use_blocks[0]["input"] == {"city": "Paris"}
            print("[ok] response shape: Anthropic Message with tool_use block")

            # The passthrough invariant — mock backend must have seen cache_control.
            assert _anthropic_mock_seen, "mock backend never received a request"
            seen_body = _anthropic_mock_seen[-1]
            print(f"[mock saw] keys={sorted(seen_body.keys())}")
            seen_str = json.dumps(seen_body)
            assert "cache_control" in seen_str, (
                "cache_control did NOT survive passthrough — "
                f"mock saw body: {seen_str[:400]}"
            )
            assert '"ephemeral"' in seen_str, "cache_control value mangled"
            print("[ok] cache_control survived end-to-end through forge proxy")

    finally:
        proxy.stop()
        mock.close()
        await mock.wait_closed()


# ── Entry point ───────────────────────────────────────────────────────

async def main() -> None:
    await test_openai()
    await test_gated_passthrough_deferred_discovery()
    await test_path2_anthropic_to_openai()
    await test_path1_anthropic_passthrough()
    print("\n[PASS] All proxy smoke tests passed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as exc:
        print(f"\n[FAIL] {exc}")
        sys.exit(1)
