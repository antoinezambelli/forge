"""Collected deterministic acceptance smoke for the Forge 0.9.0 Proxy contract.

This is deliberately a directly runnable product experiment rather than a
pytest gate.  Every scenario owns a real ``ProxyServer`` and deterministic
loopback backend, records the downstream wire, and cleans up before the next
scenario.  The registry runs to completion and exits nonzero only after the
full per-scenario summary has been printed.

Usage: ``python scripts/smoke_test_proxy.py``
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import json
import logging
import os
import socket
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

import httpx

# Anthropic's SDK requires a construction-time key even when retargeted to a
# loopback mock.  Individual requests still characterize Forge's auth policy.
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-for-smoke")
# Expected negative-path observations are asserted from HTTP responses and
# journals; production logger tracebacks would drown the collected summary.
logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Shared deterministic HTTP backend and lifecycle helpers


_SEQUENCE = itertools.count(1)
_SEQUENCE_LOCK = threading.Lock()


def _next_sequence() -> int:
    with _SEQUENCE_LOCK:
        return next(_SEQUENCE)


@dataclass(frozen=True)
class BackendRequest:
    sequence: int
    method: str
    target: str
    path: str
    query: str
    headers: dict[str, str]
    raw_body: bytes
    json_body: Any
    monotonic: float


@dataclass(frozen=True)
class MockResponse:
    status: int = 200
    body: Any = field(default_factory=dict)
    content_type: str = "application/json"
    headers: dict[str, str] = field(default_factory=dict)


Behavior = Callable[[BackendRequest, int], MockResponse | Awaitable[MockResponse]]


def _reason(status: int) -> str:
    return {
        200: "OK",
        204: "No Content",
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        500: "Internal Server Error",
        502: "Bad Gateway",
    }.get(status, "Response")


class ProgrammableBackend:
    """Dependency-free HTTP/1.1 backend with an ordered request journal."""

    def __init__(self, behavior: Behavior) -> None:
        self.behavior = behavior
        self.journal: list[BackendRequest] = []
        self._server: asyncio.AbstractServer | None = None
        self._changed = asyncio.Event()

    @property
    def port(self) -> int:
        assert self._server is not None and self._server.sockets
        return int(self._server.sockets[0].getsockname()[1])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def wait_for_requests(self, count: int, timeout: float = 10.0) -> None:
        async def wait() -> None:
            while len(self.journal) < count:
                self._changed.clear()
                if len(self.journal) < count:
                    await self._changed.wait()

        await asyncio.wait_for(wait(), timeout)

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request_line = await reader.readline()
            parts = request_line.decode("latin-1").strip().split(" ")
            method = parts[0] if len(parts) >= 1 else ""
            target = parts[1] if len(parts) >= 2 else ""
            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if line in (b"", b"\r\n", b"\n"):
                    break
                decoded = line.decode("latin-1").strip()
                if ":" in decoded:
                    key, value = decoded.split(":", 1)
                    headers[key.strip().lower()] = value.strip()
            length = int(headers.get("content-length", "0"))
            raw_body = await reader.readexactly(length) if length else b""
            try:
                json_body = json.loads(raw_body) if raw_body else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                json_body = None
            split = urlsplit(target)
            request = BackendRequest(
                sequence=_next_sequence(), method=method, target=target,
                path=split.path, query=split.query, headers=headers,
                raw_body=raw_body, json_body=json_body, monotonic=time.monotonic(),
            )
            self.journal.append(request)
            self._changed.set()
            response = self.behavior(request, len(self.journal) - 1)
            if inspect.isawaitable(response):
                response = await response
            assert isinstance(response, MockResponse)
        except Exception as exc:  # make fixture bugs visible at the HTTP boundary
            response = MockResponse(
                status=500,
                body={"fixture_error": f"{type(exc).__name__}: {exc}"},
            )

        if isinstance(response.body, bytes):
            body = response.body
        elif isinstance(response.body, str):
            body = response.body.encode("utf-8")
        else:
            body = json.dumps(response.body).encode("utf-8")
        headers = {
            "Content-Type": response.content_type,
            "Content-Length": str(len(body)),
            "Connection": "close",
            **response.headers,
        }
        head = [f"HTTP/1.1 {response.status} {_reason(response.status)}"]
        head.extend(f"{key}: {value}" for key, value in headers.items())
        writer.write(("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()


def _openai_response(
    *, text: str | None = "OK", tool: str | None = None,
    args: Any = None, model: str = "backend-model", reasoning: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    finish = "stop"
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool is not None:
        message["content"] = None
        message["tool_calls"] = [{
            "id": "call_mock_1", "type": "function",
            "function": {
                "name": tool,
                "arguments": args if isinstance(args, str) else json.dumps(args or {}),
            },
        }]
        finish = "tool_calls"
    return {
        "id": "chatcmpl-smoke", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _anthropic_response(*, model: str = "claude-wire") -> dict[str, Any]:
    return {
        "id": "msg_smoke_1", "type": "message", "role": "assistant",
        "model": model,
        "content": [{
            "type": "tool_use", "id": "toolu_smoke_1", "name": "weather",
            "input": {"city": "Paris"},
        }],
        "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 6},
    }


def _anthropic_text_response(
    text: str, *, model: str = "claude-wire",
) -> dict[str, Any]:
    return {
        "id": "msg_smoke_text", "type": "message", "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 8, "output_tokens": 4},
    }


def _models_response(model: str = "served-model", budget: int = 32768) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": model, "object": "model", "max_model_len": budget}],
    }


def _tool(name: str = "weather") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name, "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"], "additionalProperties": False,
            },
        },
    }


def _chat_body(
    *, model: str = "caller-model", tools: bool = False, stream: bool = False,
    content: str = "weather in Paris", **extra: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }
    if tools:
        body["tools"] = [_tool()]
    body.update(extra)
    return body


def _reserve_loopback_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


async def _port_accepting(port: int) -> bool:
    def connect() -> bool:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            return False

    return await asyncio.to_thread(connect)


async def _lifecycle_call(
    label: str, proxy: Any, call: Callable[[], None], timeout: float,
) -> None:
    """Run a blocking facade call with diagnostic watchdog and explicit joins."""

    done = threading.Event()
    outcome: list[BaseException] = []

    def invoke() -> None:
        try:
            call()
        except BaseException as exc:
            outcome.append(exc)
        finally:
            done.set()

    caller = threading.Thread(target=invoke, name=f"smoke-{label}", daemon=True)
    caller.start()
    completed = await asyncio.to_thread(done.wait, timeout)
    if not completed:
        # A watchdog is diagnostic, not cancellation.  Verify and join every
        # lifecycle/listener object before reporting the timeout.
        if getattr(proxy, "_started", False):
            try:
                proxy.stop()
            except Exception:
                pass
        lifecycle = getattr(proxy, "_thread", None)
        if lifecycle is not None:
            lifecycle.join(timeout=5)
        caller.join(timeout=5)
        port = int(getattr(proxy, "_port"))
        listener = await _port_accepting(port)
        raise AssertionError(
            f"{label} watchdog expired; caller_alive={caller.is_alive()} "
            f"lifecycle_alive={bool(lifecycle and lifecycle.is_alive())} "
            f"listener={listener}"
        )
    caller.join(timeout=1)
    assert not caller.is_alive(), f"{label} caller thread did not join"
    if outcome:
        raise outcome[0]


async def _start_proxy(**kwargs: Any) -> Any:
    from forge.proxy import ProxyServer

    last_error: BaseException | None = None
    for _ in range(3):
        port = _reserve_loopback_port()
        proxy = ProxyServer(port=port, **kwargs)
        try:
            await _lifecycle_call("start", proxy, proxy.start, 20.0)
            assert await _port_accepting(port), "start returned without a listener"
            return proxy
        except OSError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


async def _stop_proxy(proxy: Any) -> None:
    port = int(proxy._port)
    if getattr(proxy, "_started", False):
        await _lifecycle_call("stop", proxy, proxy.stop, 20.0)
    lifecycle = getattr(proxy, "_thread", None)
    if lifecycle is not None:
        lifecycle.join(timeout=5)
        assert not lifecycle.is_alive(), "Proxy lifecycle thread survived cleanup"
    assert not await _port_accepting(port), f"Proxy listener {port} survived cleanup"


@asynccontextmanager
async def _external_proxy(
    behavior: Behavior, *, mount: str = "", **proxy_kwargs: Any,
) -> Any:
    backend = ProgrammableBackend(behavior)
    await backend.start()
    proxy = None
    try:
        proxy = await _start_proxy(
            backend_url=f"{backend.url}{mount}", **proxy_kwargs,
        )
        yield backend, proxy
    finally:
        if proxy is not None:
            await _stop_proxy(proxy)
        await backend.stop()


def _journal(backend: ProgrammableBackend) -> str:
    return " | ".join(
        f"#{r.sequence} {r.method} {r.target} auth="
        f"{r.headers.get('authorization') or r.headers.get('x-api-key') or '-'}"
        for r in backend.journal
    ) or "<empty>"


async def _request(
    proxy: Any, method: str, path: str, *, headers: dict[str, str] | None = None,
    body: Any = None, timeout: float = 10.0,
) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.request(
            method, f"{proxy.url}{path}", headers=headers, json=body,
        )


# ---------------------------------------------------------------------------
# HTTP characterization scenarios


async def routes_and_aliases() -> None:
    def behavior(req: BackendRequest, _index: int) -> MockResponse:
        if req.path == "/health":
            return MockResponse(body={"backend": "ready"})
        if req.path == "/v1/health":
            return MockResponse(body={"backend": "v1-ready"})
        if req.path in {"/v1/models", "/models"}:
            return MockResponse(body=_models_response("catalog-model", 16384))
        if req.path == "/props":
            return MockResponse(body={"default_generation_settings": {"n_ctx": 16384}})
        return MockResponse(body=_openai_response(text="route-ok"))

    async with _external_proxy(
        behavior, budget_tokens=8192, model="configured-fallback",
    ) as (backend, proxy):
        health = await _request(proxy, "GET", "/health?detail=1")
        assert health.status_code == 200 and health.json() == {"backend": "ready"}

        models = await _request(proxy, "GET", "/v1/models?tenant=a%2Fb")
        assert models.status_code == 200
        assert models.json() == _models_response("catalog-model", 16384)

        forwarded_gets = [
            ("/v1/health?x=1", {"backend": "v1-ready"}),
            ("/models", _models_response("catalog-model", 16384)),
            ("/props?raw=1", {"default_generation_settings": {"n_ctx": 16384}}),
        ]
        for path, expected in forwarded_gets:
            response = await _request(proxy, "GET", path)
            assert response.status_code == 200 and response.json() == expected

        forge_health = await _request(proxy, "GET", "/forge/health")
        assert forge_health.status_code == 200
        assert forge_health.json() == {"status": "ok"}
        context = await _request(proxy, "GET", "/forge/usage")
        assert context.status_code == 204 and context.content == b""

        missing_gets = ["/forge/unknown", "/unknown"]
        for path in missing_gets:
            response = await _request(proxy, "GET", path)
            assert response.status_code == 404, f"{path}: {response.status_code}"

        management_routes = [
            ("POST", "/props?slot=3"),
            ("POST", "/models"),
            ("DELETE", "/models"),
            ("POST", "/models/load"),
            ("POST", "/models/unload"),
            ("GET", "/models/sse"),
        ]
        expected_404 = {
            "error": {"message": "Not found", "type": "proxy_error"},
        }
        for method, path in management_routes:
            response = await _request(
                proxy, method, path,
                body={"model": "management-smoke"} if method != "GET" else None,
            )
            assert response.status_code == 404, f"{method} {path}: {response.status_code}"
            assert response.json() == expected_404, f"{method} {path}: {response.text}"
        assert [request.target for request in backend.journal] == [
            "/health?detail=1", "/v1/models?tenant=a%2Fb", "/v1/health?x=1",
            "/models", "/props?raw=1",
        ], _journal(backend)

        options = await _request(proxy, "OPTIONS", "/arbitrary/path?x=1")
        assert options.status_code == 204
        assert options.headers["access-control-allow-origin"] == "*"
        assert "x-api-key" in options.headers["access-control-allow-headers"].lower()

        fallback_body = _chat_body()
        fallback_body.pop("model")
        routed_requests = (
            ("/v1/chat/completions", _chat_body(model="caller-wins"), "caller-wins"),
            ("/chat/completions?alias=1", fallback_body, "configured-fallback"),
        )
        for alias, body, expected_model in routed_requests:
            response = await _request(proxy, "POST", alias, body=body)
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == "route-ok"
            assert response.json()["model"] == expected_model
            assert backend.journal[-1].json_body["model"] == expected_model
        assert [r.target for r in backend.journal][-2:] == [
            "/v1/chat/completions", "/v1/chat/completions",
        ], _journal(backend)


async def metadata_forwarding() -> None:
    raw = b"\x00forge-metadata\xff"

    def behavior(req: BackendRequest, _index: int) -> MockResponse:
        assert req.method == "GET" and req.target == "/models?tenant=a%2Fb"
        return MockResponse(
            status=403,
            body=raw,
            content_type="application/octet-stream",
        )

    async with _external_proxy(
        behavior, budget_tokens=8192,
    ) as (backend, proxy):
        response = await _request(
            proxy, "GET", "/models?tenant=a%2Fb",
            headers={"x-api-key": "METADATA-KEY"},
        )
        assert response.status_code == 403 and response.content == raw
        assert response.headers["content-type"] == "application/octet-stream"
        assert response.headers["content-length"] == str(len(raw))
        assert response.headers["access-control-allow-origin"] == "*"
        assert response.headers["connection"] == "close"
        assert backend.journal[0].headers.get("authorization") == (
            "Bearer METADATA-KEY"
        )

    # A backend transport failure is a bounded 502 on an approved path.
    dead_port = _reserve_loopback_port()
    proxy = await _start_proxy(
        backend_url=f"http://127.0.0.1:{dead_port}", budget_tokens=8192,
    )
    try:
        response = await _request(proxy, "GET", "/props")
        assert response.status_code == 502
        assert response.json()["error"]["message"] == "Backend request failed"
    finally:
        await _stop_proxy(proxy)


async def current_context_reporting() -> None:
    def behavior(req: BackendRequest, _index: int) -> MockResponse:
        body = req.json_body or {}
        model = body.get("model")
        if model == "failed-model":
            return MockResponse(status=500, body={"error": "inference failed"})
        response = _openai_response(text="context-ok", model="backend-model")
        if model == "unavailable-model":
            response.pop("usage")
        return MockResponse(body=response)

    async def context(proxy: Any) -> httpx.Response:
        for _ in range(100):
            response = await _request(proxy, "GET", "/forge/usage")
            if response.status_code == 200:
                return response
            await asyncio.sleep(0.01)
        return response

    async with _external_proxy(
        behavior, budget_tokens=100,
    ) as (backend, proxy):
        initial = await _request(proxy, "GET", "/forge/usage")
        assert initial.status_code == 204 and initial.content == b""

        published = await _request(
            proxy, "POST", "/v1/chat/completions",
            headers={"X-Claude-Code-Session-Id": "claude-session"},
            body=_chat_body(
                model="published-model", litellm_session_id="lite-session",
            ),
        )
        assert published.status_code == 200
        snapshot = (await context(proxy)).json()
        assert snapshot["current_usage_tokens"] == 10
        assert snapshot["context_window_tokens"] == 100
        assert snapshot["usage_percent"] == 10.0
        assert snapshot["model"] == "published-model"
        assert snapshot["context_window_source"] == "operator_config"
        assert snapshot["session"] == {
            "id": "claude-session", "source": "claude_code",
        }
        assert backend.journal[-1].json_body["litellm_session_id"] == "lite-session"

        failed = await _request(
            proxy, "POST", "/v1/chat/completions",
            body=_chat_body(model="failed-model"),
        )
        assert failed.status_code == 502
        assert (await context(proxy)).json()["model"] == "published-model"

        subagent = await _request(
            proxy, "POST", "/v1/chat/completions",
            headers={"X-Claude-Code-Agent-Id": "agent"},
            body=_chat_body(model="subagent-model"),
        )
        assert subagent.status_code == 200
        assert (await context(proxy)).json()["model"] == "published-model"

        unavailable = await _request(
            proxy, "POST", "/v1/chat/completions",
            body=_chat_body(model="unavailable-model"),
        )
        assert unavailable.status_code == 200
        for _ in range(100):
            cleared = await _request(proxy, "GET", "/forge/usage")
            if cleared.status_code == 204:
                break
            await asyncio.sleep(0.01)
        assert cleared.status_code == 204 and cleared.content == b""

    # A new Proxy process starts with no snapshot.
    async with _external_proxy(
        behavior, budget_tokens=100,
    ) as (_backend, restarted):
        empty = await _request(restarted, "GET", "/forge/usage")
        assert empty.status_code == 204 and empty.content == b""


async def generic_fidelity_and_streaming() -> None:
    def behavior(req: BackendRequest, _index: int) -> MockResponse:
        body = req.json_body or {}
        if body.get("tools"):
            return MockResponse(body=_openai_response(tool="weather", args={"city": "Paris"}))
        return MockResponse(body=_openai_response(text="plain-ok"))

    async with _external_proxy(behavior, budget_tokens=8192) as (backend, proxy):
        plain = _chat_body(
            model="caller-plain", max_tokens=77, stop=["END"],
            litellm_session_id="session-generic", vendor_extension={"keep": True},
        )
        response = await _request(proxy, "POST", "/v1/chat/completions", body=plain)
        assert response.status_code == 200
        assert response.json()["model"] == "caller-plain"
        first = backend.journal[-1].json_body
        assert first["model"] == "caller-plain"
        assert first["max_tokens"] == 77 and first["stop"] == ["END"]
        assert first["litellm_session_id"] == "session-generic"
        assert first["vendor_extension"] == {"keep": True}
        assert first["messages"] == plain["messages"]

        raw_tool = _tool()
        raw_tool["function"]["x-vendor-schema"] = {"verbatim": True}
        tool_body = _chat_body(
            model="caller-tools", tools=True, stream=False,
            tool_choice={"type": "function", "function": {"name": "weather"}},
        )
        tool_body["tools"] = [raw_tool]
        tool_body["messages"][0]["name"] = "named-caller"
        tool_response = await _request(proxy, "POST", "/v1/chat/completions", body=tool_body)
        assert tool_response.status_code == 200
        assert tool_response.json()["model"] == "caller-tools"
        sent = backend.journal[-1].json_body
        assert sent["messages"] == tool_body["messages"]
        assert sent["tools"] == tool_body["tools"]
        assert sent["tool_choice"] == tool_body["tool_choice"]

        stream_body = dict(tool_body, model="caller-stream", stream=True)
        streamed = await _request(proxy, "POST", "/v1/chat/completions", body=stream_body)
        assert streamed.status_code == 200
        assert streamed.headers["content-type"].startswith("text/event-stream")
        assert "[DONE]" in streamed.text
        assert '"tool_calls"' in streamed.text
        # Forge buffers inference; the downstream call itself remains non-SSE.
        assert backend.journal[-1].json_body.get("stream") is not True


async def tool_free_guardrail_bypass() -> None:
    rescue_shaped = json.dumps({
        "tool": "weather", "args": {"city": "Paris"},
    })

    def behavior(req: BackendRequest, _index: int) -> MockResponse:
        assert "tools" not in req.json_body
        return MockResponse(body=_openai_response(text=rescue_shaped))

    async with _external_proxy(
        behavior, budget_tokens=128, max_retries=5,
        max_tool_errors=5, rescue_enabled=True, inject_respond_tool=True,
    ) as (backend, proxy):
        response = await _request(
            proxy, "POST", "/v1/chat/completions",
            body=_chat_body(model="tool-free-bypass", content="X" * 800),
        )
        assert response.status_code == 200
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] == rescue_shaped
        assert len(backend.journal) == 1, _journal(backend)
        wire = backend.journal[0].json_body
        assert wire["model"] == "tool-free-bypass"
        assert "tools" not in wire


async def specialized_external_adapters() -> None:
    async def run_case(backend_name: str) -> None:
        def behavior(_req: BackendRequest, _index: int) -> MockResponse:
            return MockResponse(body=_openai_response(text=f"{backend_name}-ok"))

        async with _external_proxy(
            behavior, backend=backend_name, budget_tokens=8192,
            model=f"configured-{backend_name}",
        ) as (backend, proxy):
            body = _chat_body(
                model=f"caller-{backend_name}",
                max_tokens=41, vendor_extension={"adapter": backend_name},
            )
            response = await _request(
                proxy, "POST", "/v1/chat/completions", body=body,
            )
            assert response.status_code == 200
            assert response.json()["model"] == f"caller-{backend_name}"
            assert len(backend.journal) == 1, _journal(backend)
            request = backend.journal[0]
            assert request.path == "/v1/chat/completions"
            assert request.path != "/api/chat"
            assert request.json_body["model"] == f"caller-{backend_name}"
            assert request.json_body["messages"] == body["messages"]
            assert request.json_body["max_tokens"] == 41
            assert request.json_body["vendor_extension"] == {"adapter": backend_name}
            # All three specialized external names currently select Forge's
            # llama/OpenAI-compatible adapter.  In particular external Ollama
            # does not select OllamaClient's /api/chat wire.
            assert request.json_body["cache_prompt"] is True

    for backend_name in ("llamaserver", "llamafile", "ollama"):
        await run_case(backend_name)


async def retry_compaction_and_exhaustion() -> None:
    counts: dict[str, int] = {}

    def behavior(req: BackendRequest, _index: int) -> MockResponse:
        model = (req.json_body or {}).get("model", "")
        counts[model] = counts.get(model, 0) + 1
        if model == "retry-model" and counts[model] == 1:
            return MockResponse(body=_openai_response(text="please do it yourself"))
        if model == "exhaust-model":
            return MockResponse(body=_openai_response(text="still plain text"))
        if model == "tool-error-model":
            return MockResponse(body=_openai_response(tool="weather", args="[]"))
        return MockResponse(body=_openai_response(tool="weather", args={"city": "Paris"}))

    async with _external_proxy(
        behavior, budget_tokens=200, max_retries=5, max_tool_errors=1,
        reasoning_replay="full",
    ) as (backend, proxy):
        retry = _chat_body(model="retry-model", tools=True)
        retry["messages"][0]["name"] = "raw-first-attempt"
        result = await _request(proxy, "POST", "/v1/chat/completions", body=retry)
        assert result.status_code == 200
        retry_wires = [r.json_body for r in backend.journal if r.json_body.get("model") == "retry-model"]
        assert len(retry_wires) == 2
        assert retry_wires[0]["messages"] == retry["messages"]
        assert len(retry_wires[1]["messages"]) > len(retry_wires[0]["messages"])
        assert "name" not in retry_wires[1]["messages"][0]

        before = len(backend.journal)
        exhausted = await _request(
            proxy, "POST", "/v1/chat/completions",
            body=_chat_body(model="exhaust-model", tools=True),
        )
        assert exhausted.status_code == 200
        assert exhausted.json()["choices"][0]["message"]["content"] == "still plain text"
        assert len(backend.journal) - before == 6  # initial + max_retries

        before = len(backend.journal)
        tool_error = await _request(
            proxy, "POST", "/v1/chat/completions",
            body=_chat_body(model="tool-error-model", tools=True),
        )
        assert tool_error.status_code == 200
        assert len(backend.journal) - before == 2  # max_tool_errors, not max_retries

        short = _chat_body(model="compact-short", tools=True, content="short")
        short["messages"][0]["name"] = "wire-marker"
        await _request(proxy, "POST", "/v1/chat/completions", body=short)
        short_wire = backend.journal[-1].json_body
        long = _chat_body(model="compact-long", tools=True, content="X" * 900)
        long["messages"][0]["name"] = "wire-marker"
        await _request(proxy, "POST", "/v1/chat/completions", body=long)
        long_wire = backend.journal[-1].json_body
        assert short_wire["messages"][0]["name"] == "wire-marker"
        assert long_wire["messages"][0]["name"] == "wire-marker"
        assert long_wire["messages"][0]["content"] == "X" * 900
        assert long_wire["messages"] == long["messages"]


async def guardrail_modes() -> None:
    rescue_text = json.dumps({"tool": "weather", "args": {"city": "Paris"}})

    def rescue_behavior(_req: BackendRequest, _index: int) -> MockResponse:
        return MockResponse(body=_openai_response(text=rescue_text))

    async with _external_proxy(
        rescue_behavior, budget_tokens=8192, max_retries=0,
    ) as (_backend, proxy):
        rescued = await _request(
            proxy, "POST", "/v1/chat/completions",
            body=_chat_body(tools=True),
        )
        assert rescued.json()["choices"][0]["finish_reason"] == "tool_calls"

    async with _external_proxy(
        rescue_behavior, budget_tokens=8192, max_retries=0, rescue_enabled=False,
    ) as (_backend, proxy):
        unrescued = await _request(
            proxy, "POST", "/v1/chat/completions",
            body=_chat_body(tools=True),
        )
        assert unrescued.json()["choices"][0]["message"]["content"] == rescue_text

    def prompt_behavior(_req: BackendRequest, _index: int) -> MockResponse:
        return MockResponse(body=_openai_response(text=rescue_text))

    async with _external_proxy(
        prompt_behavior, budget_tokens=8192, backend_capability="prompt",
    ) as (backend, proxy):
        prompt = await _request(
            proxy, "POST", "/v1/chat/completions",
            body=_chat_body(tools=True),
        )
        assert prompt.status_code == 200
        wire = backend.journal[-1].json_body
        assert "tools" not in wire
        assert "weather" in json.dumps(wire["messages"])

    def respond_behavior(req: BackendRequest, _index: int) -> MockResponse:
        names = [t["function"]["name"] for t in req.json_body.get("tools", [])]
        assert names == ["weather", "respond"]
        return MockResponse(body=_openai_response(tool="respond", args={"message": "final text"}))

    async with _external_proxy(
        respond_behavior, budget_tokens=8192, inject_respond_tool=True,
    ) as (_backend, proxy):
        response = await _request(
            proxy, "POST", "/v1/chat/completions",
            body=_chat_body(tools=True),
        )
        assert response.json()["choices"][0]["message"]["content"] == "final text"

    def reasoning_behavior(_req: BackendRequest, _index: int) -> MockResponse:
        return MockResponse(body=_openai_response(tool="weather", args={"city": "Paris"}))

    async with _external_proxy(
        reasoning_behavior, budget_tokens=8192, reasoning_replay="keep-last",
    ) as (backend, proxy):
        body = _chat_body(tools=True)
        body["messages"] = [
            {"role": "assistant", "content": None, "reasoning_content": "old", "tool_calls": []},
            {"role": "assistant", "content": None, "reasoning_content": "latest", "tool_calls": []},
        ]
        await _request(proxy, "POST", "/v1/chat/completions", body=body)
        sent = backend.journal[-1].json_body["messages"]
        assert "reasoning_content" not in sent[0]
        assert sent[1]["reasoning_content"] == "latest"


async def anthropic_inbound_conversion() -> None:
    def behavior(_req: BackendRequest, _index: int) -> MockResponse:
        return MockResponse(body=_openai_response(tool="weather", args={"city": "Paris"}))

    async with _external_proxy(behavior, budget_tokens=8192) as (backend, proxy):
        body = {
            "model": "anthropic-caller", "max_tokens": 321,
            "system": "Be concise.",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{
                "name": "weather", "description": "Get weather",
                "input_schema": _tool()["function"]["parameters"],
            }],
            "tool_choice": {"type": "any"}, "stop_sequences": ["DONE"],
            "metadata": {"user_id": "drop-me"},
            "litellm_session_id": "drop-on-conversion", "stream": False,
        }
        response = await _request(proxy, "POST", "/v1/messages", body=body)
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message" and data["model"] == "anthropic-caller"
        assert data["stop_reason"] == "tool_use"
        wire = backend.journal[-1].json_body
        assert wire["model"] == "anthropic-caller"
        assert wire["max_tokens"] == 321 and wire["stop"] == ["DONE"]
        assert wire["tool_choice"] == "required"
        assert "metadata" not in wire
        assert wire["litellm_session_id"] == "drop-on-conversion"
        assert wire["tools"][0]["function"]["name"] == "weather"

        streamed = await _request(
            proxy, "POST", "/v1/messages", body=dict(body, stream=True),
        )
        assert streamed.status_code == 200
        assert "[DONE]" not in streamed.text
        events = [
            line.removeprefix("event: ") for line in streamed.text.splitlines()
            if line.startswith("event: ")
        ]
        assert events[0] == "message_start" and events[-1] == "message_stop"
        assert "message_delta" in events and "tool_use" in streamed.text


async def anthropic_downstream() -> None:
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:
        raise AssertionError(
            "Anthropic downstream characterization requires the installed anthropic SDK"
        ) from exc

    def behavior(req: BackendRequest, _index: int) -> MockResponse:
        assert req.path == "/v1/messages"
        return MockResponse(body=_anthropic_response())

    async with _external_proxy(
        behavior, backend="anthropic", model="claude-pin", budget_tokens=8192,
    ) as (backend, proxy):
        cache = {"type": "ephemeral"}
        body = {
            "model": "caller-ignored", "max_tokens": 256,
            "system": [{"type": "text", "text": "cached system", "cache_control": cache}],
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Weather?", "cache_control": cache}],
            }],
            "tools": [{
                "name": "weather", "description": "Get weather",
                "input_schema": _tool()["function"]["parameters"],
            }],
            "stream": False,
        }
        response = await _request(
            proxy, "POST", "/v1/messages",
            headers={"x-api-key": "forwarded-anthropic"}, body=body,
        )
        assert response.status_code == 200 and response.json()["model"] == "claude-pin"
        first = backend.journal[-1]
        assert first.headers.get("x-api-key") == "forwarded-anthropic"
        assert first.json_body["model"] == "claude-pin"
        assert "cache_control" in json.dumps(first.json_body)
        assert "litellm_session_id" not in first.json_body

        # Forge observes the LiteLLM session without consuming it; the
        # applicable Anthropic path forwards it unchanged.
        before = len(backend.journal)
        session_response = await _request(
            proxy, "POST", "/v1/messages",
            headers={"x-api-key": "forwarded-anthropic"},
            body=dict(body, litellm_session_id="anthropic-rejected"),
        )
        assert session_response.status_code == 200, session_response.text
        assert len(backend.journal) == before + 1
        assert backend.journal[-1].json_body["litellm_session_id"] == (
            "anthropic-rejected"
        )

        anthropic_stream = await _request(
            proxy, "POST", "/v1/messages",
            headers={"x-api-key": "forwarded-anthropic"},
            body=dict(body, stream=True),
        )
        assert anthropic_stream.status_code == 200
        assert "[DONE]" not in anthropic_stream.text
        anthropic_events = [
            line.removeprefix("event: ")
            for line in anthropic_stream.text.splitlines()
            if line.startswith("event: ")
        ]
        assert anthropic_events[0] == "message_start"
        assert anthropic_events[-1] == "message_stop"

        openai_body = _chat_body(model="openai-alias", tools=True, stream=True)
        streamed = await _request(
            proxy, "POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer relocated"}, body=openai_body,
        )
        assert streamed.status_code == 200 and "[DONE]" in streamed.text
        assert backend.journal[-1].headers.get("x-api-key") == "relocated"
        assert backend.journal[-1].json_body["model"] == "claude-pin"


async def anthropic_unpinned_concurrency_retry() -> None:
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:
        raise AssertionError(
            "Anthropic downstream characterization requires the installed anthropic SDK"
        ) from exc

    concurrent_barrier = asyncio.Barrier(2)
    retry_attempts = 0

    async def behavior(req: BackendRequest, _index: int) -> MockResponse:
        nonlocal retry_attempts
        assert req.path == "/v1/messages"
        model = req.json_body["model"]
        if model in {"route-alpha", "route-beta"}:
            await asyncio.wait_for(concurrent_barrier.wait(), 10.0)
            return MockResponse(body=_anthropic_response(model=model))
        assert model == "route-retry"
        retry_attempts += 1
        if retry_attempts == 1:
            return MockResponse(body=_anthropic_text_response("use a tool", model=model))
        return MockResponse(body=_anthropic_response(model=model))

    async with _external_proxy(
        behavior, backend="anthropic", budget_tokens=8192,
        max_retries=1,
    ) as (backend, proxy):
        def routed_body(model: str, marker: str) -> dict[str, Any]:
            cache = {"type": "ephemeral"}
            return {
                "model": model, "max_tokens": 128,
                "system": [{
                    "type": "text", "text": f"system-{marker}",
                    "cache_control": cache,
                }],
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text", "text": f"request-{marker}",
                        "cache_control": cache,
                    }],
                }],
                "tools": [{
                    "name": "weather", "description": "Get weather",
                    "input_schema": _tool()["function"]["parameters"],
                }],
            }

        alpha_body = routed_body("route-alpha", "alpha")
        beta_body = routed_body("route-beta", "beta")
        alpha, beta = await asyncio.gather(
            _request(
                proxy, "POST", "/v1/messages",
                headers={"x-api-key": "key-alpha"}, body=alpha_body,
            ),
            _request(
                proxy, "POST", "/v1/messages",
                headers={"x-api-key": "key-beta"}, body=beta_body,
            ),
        )
        assert alpha.status_code == 200 and alpha.json()["model"] == "route-alpha"
        assert beta.status_code == 200 and beta.json()["model"] == "route-beta"
        concurrent = {
            request.json_body["model"]: request for request in backend.journal
        }
        assert set(concurrent) == {"route-alpha", "route-beta"}
        for model, marker, key in (
            ("route-alpha", "alpha", "key-alpha"),
            ("route-beta", "beta", "key-beta"),
        ):
            request = concurrent[model]
            serialized = json.dumps(request.json_body)
            assert f"system-{marker}" in serialized
            assert f"request-{marker}" in serialized
            other = "beta" if marker == "alpha" else "alpha"
            assert f"system-{other}" not in serialized
            assert f"request-{other}" not in serialized
            assert request.headers.get("x-api-key") == key

        retry_body = routed_body("route-retry", "retry")
        before = len(backend.journal)
        retried = await _request(
            proxy, "POST", "/v1/messages",
            headers={"x-api-key": "key-retry"}, body=retry_body,
        )
        assert retried.status_code == 200
        assert retried.json()["model"] == "route-retry"
        retry_wires = backend.journal[before:]
        assert len(retry_wires) == 2, _journal(backend)
        first_body = retry_wires[0].json_body
        rebuilt_body = retry_wires[1].json_body
        assert first_body["model"] == rebuilt_body["model"] == "route-retry"
        assert "cache_control" in json.dumps(first_body)
        assert "cache_control" not in json.dumps(rebuilt_body)
        assert len(rebuilt_body["messages"]) > len(first_body["messages"])
        assert "use a tool" in json.dumps(rebuilt_body["messages"])
        assert all(
            request.headers.get("x-api-key") == "key-retry"
            for request in retry_wires
        )


async def credentials_and_conflicts() -> None:
    def behavior(_req: BackendRequest, _index: int) -> MockResponse:
        return MockResponse(body=_openai_response(text="auth-ok"))

    async with _external_proxy(
        behavior, budget_tokens=8192, backend_api_key="STATIC",
    ) as (backend, proxy):
        ok = await _request(proxy, "POST", "/v1/chat/completions", body=_chat_body())
        assert ok.status_code == 200
        assert backend.journal[-1].headers.get("authorization") == "Bearer STATIC"
        before = len(backend.journal)
        conflict = await _request(
            proxy, "POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer INBOUND"}, body=_chat_body(),
        )
        assert conflict.status_code == 400 and len(backend.journal) == before
        assert "INBOUND" not in conflict.text

    async with _external_proxy(behavior, budget_tokens=8192) as (backend, proxy):
        forwarded = await _request(
            proxy, "POST", "/v1/chat/completions",
            headers={"x-api-key": "FORWARDED"}, body=_chat_body(),
        )
        assert forwarded.status_code == 200
        # OpenAI inbound -> OpenAI backend is same-protocol, so the single
        # credential stays in the caller's chosen slot verbatim.
        assert backend.journal[-1].headers.get("x-api-key") == "FORWARDED"
        before = len(backend.journal)
        conflict = await _request(
            proxy, "POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer ONE", "x-api-key": "TWO"},
            body=_chat_body(),
        )
        assert conflict.status_code == 400 and len(backend.journal) == before
        assert "ONE" not in conflict.text and "TWO" not in conflict.text


async def generic_deferred_discovery() -> None:
    key = "DEFERRED"

    def behavior(req: BackendRequest, _index: int) -> MockResponse:
        if req.headers.get("authorization") != f"Bearer {key}":
            return MockResponse(status=401, body={"error": "credential required"})
        if req.path.endswith("/props"):
            return MockResponse(body={"default_generation_settings": {"n_ctx": 16384}})
        return MockResponse(body=_openai_response(text="deferred-ok"))

    async with _external_proxy(behavior) as (backend, proxy):
        assert backend.journal == [], "passthrough startup unexpectedly probed"
        failed = await _request(proxy, "POST", "/v1/chat/completions", body=_chat_body())
        assert failed.status_code == 401
        assert [r.path for r in backend.journal] == ["/v1/chat/completions"]
        good = await _request(
            proxy, "POST", "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, body=_chat_body(),
        )
        assert good.status_code == 200
        assert [r.path for r in backend.journal] == [
            "/v1/chat/completions", "/v1/chat/completions",
        ], _journal(backend)
        again = await _request(
            proxy, "POST", "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, body=_chat_body(),
        )
        assert again.status_code == 200
        assert [r.path for r in backend.journal][-1] == "/v1/chat/completions"
        assert not any(r.path.endswith("/props") for r in backend.journal)


async def vllm_deferred_inference() -> None:
    key = "VLLMKEY"

    def behavior(req: BackendRequest, _index: int) -> MockResponse:
        if req.headers.get("authorization") != f"Bearer {key}":
            return MockResponse(status=401, body={"error": "credential required"})
        if req.path == "/v1/models":
            return MockResponse(body=_models_response("served-vllm", 65536))
        return MockResponse(body=_openai_response(tool="weather", args={"city": "Paris"}))

    async with _external_proxy(behavior, backend="vllm") as (backend, proxy):
        assert backend.journal == []
        failed = await _request(
            proxy, "POST", "/v1/chat/completions",
            body=_chat_body(model="alias", tools=True),
        )
        assert failed.status_code == 401
        body = _chat_body(
            model="alias", tools=True,
            tool_choice={"type": "function", "function": {"name": "weather"}},
            litellm_session_id="drop-vllm", vendor_extension="drop-vllm",
        )
        body["tools"][0]["function"]["x-vendor"] = True
        good = await _request(
            proxy, "POST", "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, body=body,
        )
        assert good.status_code == 200 and good.json()["model"] == "served-vllm"
        assert [r.path for r in backend.journal] == [
            "/v1/models", "/v1/models", "/v1/chat/completions",
        ], _journal(backend)
        wire = backend.journal[-1].json_body
        assert wire["model"] == "served-vllm"
        assert wire["tool_choice"] == body["tool_choice"]
        assert wire["litellm_session_id"] == "drop-vllm"
        assert wire["vendor_extension"] == "drop-vllm"
        assert wire["tools"] == body["tools"]


async def vllm_models_discovery_retry() -> None:
    attempts = 0

    def behavior(req: BackendRequest, _index: int) -> MockResponse:
        nonlocal attempts
        if req.path == "/health":
            return MockResponse(body={"ready": True})
        assert req.path == "/v1/models"
        attempts += 1
        if attempts == 1:
            return MockResponse(status=502, body={"error": "temporary"})
        return MockResponse(body=_models_response("models-triggered", 49152))

    async with _external_proxy(behavior, backend="vllm") as (backend, proxy):
        assert backend.journal == [], "GET /v1/models discovery ran during startup"
        first = await _request(
            proxy, "GET", "/v1/models",
            headers={"Authorization": "Bearer MODELKEY"},
        )
        assert first.status_code == 502
        assert [r.path for r in backend.journal] == ["/v1/models"]
        assert not any(r.path.endswith("chat/completions") for r in backend.journal)
        health = await _request(proxy, "GET", "/health")
        assert health.status_code == 200
        assert health.json() == {"ready": True}
        assert [r.path for r in backend.journal] == ["/v1/models", "/health"]
        second = await _request(
            proxy, "GET", "/v1/models",
            headers={"Authorization": "Bearer MODELKEY"},
        )
        assert second.status_code == 200
        assert second.json()["data"][0]["id"] == "models-triggered"
        assert [r.path for r in backend.journal] == [
            "/v1/models", "/health", "/v1/models",
        ]
        assert all(
            r.headers.get("authorization") == "Bearer MODELKEY"
            for r in backend.journal if r.path == "/v1/models"
        )


async def vllm_discovery_matrix() -> None:
    async def run_case(
        *, budget: int | None, model: str | None, static: str | None,
        expected_start_probes: int, expected_wire_model: str,
    ) -> None:
        def behavior(req: BackendRequest, _index: int) -> MockResponse:
            if req.path == "/v1/models":
                return MockResponse(body=_models_response("served-matrix", 32768))
            return MockResponse(body=_openai_response(text="matrix-ok"))

        kwargs: dict[str, Any] = {"backend": "vllm"}
        if budget is not None:
            kwargs["budget_tokens"] = budget
        if model is not None:
            kwargs["model"] = model
        if static is not None:
            kwargs["backend_api_key"] = static
        async with _external_proxy(behavior, **kwargs) as (backend, proxy):
            assert len(backend.journal) == expected_start_probes, _journal(backend)
            response = await _request(
                proxy, "POST", "/v1/chat/completions",
                headers=None if static else {"Authorization": "Bearer FORWARD"},
                body=_chat_body(model="caller-matrix"),
            )
            assert response.status_code == 200
            wire = [r for r in backend.journal if r.path.endswith("chat/completions")][-1]
            assert wire.json_body["model"] == expected_wire_model
            if static:
                assert all(
                    r.headers.get("authorization") == f"Bearer {static}"
                    for r in backend.journal
                )

    # A static credential does not change lazy timing: both implicit combined
    # discovery and explicit-budget identity-only discovery wait for inference.
    await run_case(
        budget=None, model=None, static="STATIC-IMPLICIT",
        expected_start_probes=0, expected_wire_model="served-matrix",
    )
    await run_case(
        budget=8192, model=None, static="STATIC-EXPLICIT",
        expected_start_probes=0, expected_wire_model="served-matrix",
    )
    # Pin + explicit budget settles both roles and suppresses all discovery.
    await run_case(
        budget=8192, model="pin-model", static="STATIC-PINNED",
        expected_start_probes=0, expected_wire_model="pin-model",
    )

    # A forwarded credential with a pin but no budget defers a pinned-entry
    # budget probe to inference and never overwrites the pin.
    def pinned_behavior(req: BackendRequest, _index: int) -> MockResponse:
        if req.path == "/v1/models":
            return MockResponse(body={
                "object": "list",
                "data": [{"id": "pin-model", "object": "model", "max_model_len": 24576}],
            })
        return MockResponse(body=_openai_response(text="pinned-ok"))

    async with _external_proxy(
        pinned_behavior, backend="vllm", model="pin-model",
    ) as (backend, proxy):
        assert backend.journal == []
        response = await _request(
            proxy, "POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer FORWARD-PIN"},
            body=_chat_body(model="caller"),
        )
        assert response.status_code == 200
        await backend.wait_for_requests(2)
        assert [r.path for r in backend.journal] == [
            "/v1/chat/completions", "/v1/models",
        ], _journal(backend)
        assert backend.journal[0].json_body["model"] == "pin-model"

    # Forwarded credential + explicit budget still defers unpinned identity.
    # Identity is independent from the reporting window, so an ID without
    # max_model_len is enough for inference and the success latch is retained.
    explicit_attempts = 0

    def forwarded_explicit_behavior(
        req: BackendRequest, _index: int,
    ) -> MockResponse:
        nonlocal explicit_attempts
        if req.path == "/v1/models":
            explicit_attempts += 1
            if explicit_attempts == 1:
                return MockResponse(body={
                    "object": "list",
                    "data": [{"id": "served-no-window", "object": "model"}],
                })
            return MockResponse(body=_models_response("served-explicit", 73728))
        return MockResponse(body=_openai_response(text="forwarded-explicit-ok"))

    async with _external_proxy(
        forwarded_explicit_behavior, backend="vllm", budget_tokens=8192,
    ) as (backend, proxy):
        assert backend.journal == []
        first = await _request(
            proxy, "POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer FORWARD-EXPLICIT"},
            body=_chat_body(model="caller-explicit"),
        )
        assert first.status_code == 200
        assert [r.path for r in backend.journal] == [
            "/v1/models", "/v1/chat/completions",
        ]
        assert backend.journal[-1].json_body["model"] == "served-no-window"
        second = await _request(
            proxy, "POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer FORWARD-EXPLICIT"},
            body=_chat_body(model="caller-explicit"),
        )
        assert second.status_code == 200
        assert [r.path for r in backend.journal] == [
            "/v1/models", "/v1/chat/completions", "/v1/chat/completions",
        ], _journal(backend)
        assert all(
            request.headers.get("authorization") == "Bearer FORWARD-EXPLICIT"
            for request in backend.journal
        )
        assert backend.journal[-1].json_body["model"] == "served-no-window"
        assert second.json()["model"] == "served-no-window"

    # Static credentials no longer change discovery timing. Public catalog
    # forwarding is honest and side-effect-free; first-request identity
    # failure blocks dispatch and remains retryable.
    def static_explicit_failure_behavior(
        req: BackendRequest, _index: int,
    ) -> MockResponse:
        if req.path == "/v1/models":
            return MockResponse(status=502, body={"error": "catalog unavailable"})
        return MockResponse(body=_openai_response(text="static-soft-failure-ok"))

    async with _external_proxy(
        static_explicit_failure_behavior, backend="vllm", budget_tokens=8192,
        backend_api_key="STATIC-EXPLICIT-FAIL",
    ) as (backend, proxy):
        assert backend.journal == []
        health = await _request(proxy, "GET", "/forge/health")
        assert health.status_code == 200
        models = await _request(proxy, "GET", "/v1/models")
        assert models.status_code == 502
        assert [r.path for r in backend.journal] == ["/v1/models"]
        response = await _request(
            proxy, "POST", "/v1/chat/completions",
            body=_chat_body(model="caller-static-soft"),
        )
        assert response.status_code == 502
        assert [r.path for r in backend.journal] == [
            "/v1/models", "/v1/models",
        ]
        assert all(
            request.headers.get("authorization") == "Bearer STATIC-EXPLICIT-FAIL"
            for request in backend.journal
        )


async def eager_static_failure_timing() -> None:
    def generic_behavior(req: BackendRequest, _index: int) -> MockResponse:
        assert req.path == "/v1/chat/completions"
        return MockResponse(body=_openai_response(text="metadata-optional"))

    async with _external_proxy(
        generic_behavior, backend_api_key="EAGER-STATIC",
    ) as (backend, proxy):
        assert backend.journal == []
        response = await _request(
            proxy, "POST", "/v1/chat/completions", body=_chat_body(),
        )
        assert response.status_code == 200
        assert [request.path for request in backend.journal] == [
            "/v1/chat/completions",
        ]

    def vllm_behavior(req: BackendRequest, _index: int) -> MockResponse:
        assert req.path == "/v1/models"
        return MockResponse(status=500, body={"error": "no identity"})

    async with _external_proxy(
        vllm_behavior, backend="vllm", backend_api_key="EAGER-STATIC",
    ) as (backend, proxy):
        assert backend.journal == []
        response = await _request(
            proxy, "POST", "/v1/chat/completions", body=_chat_body(),
        )
        assert response.status_code == 502, response.text
        assert [request.path for request in backend.journal] == ["/v1/models"]
        assert backend.journal[0].headers.get("authorization") == (
            "Bearer EAGER-STATIC"
        )


async def url_normalization_and_explicit_port() -> None:
    async def run_case(mount: str, expected: list[str]) -> None:
        def behavior(req: BackendRequest, _index: int) -> MockResponse:
            return MockResponse(body=_openai_response(text="url-ok"))

        async with _external_proxy(
            behavior, mount=mount, budget_tokens=8192,
        ) as (backend, proxy):
            response = await _request(
                proxy, "POST", "/v1/chat/completions",
                headers={"Authorization": "Bearer URLKEY"}, body=_chat_body(),
            )
            assert response.status_code == 200
            assert [r.path for r in backend.journal] == expected, _journal(backend)

    await run_case("", ["/v1/chat/completions"])
    await run_case("/v1/", ["/v1/chat/completions"])
    await run_case("/prefix", ["/prefix/v1/chat/completions"])
    await run_case("/prefix/v1", ["/prefix/v1/chat/completions"])

    # An explicit unmanaged port replaces only the URL authority port and
    # preserves the normalized mount prefix.
    backend = ProgrammableBackend(
        lambda _req, _index: MockResponse(body=_openai_response(text="port-ok")),
    )
    await backend.start()
    proxy = None
    try:
        proxy = await _start_proxy(
            backend_url="http://127.0.0.1:1/prefix",
            backend_port=backend.port,
            budget_tokens=8192,
        )
        response = await _request(
            proxy, "POST", "/v1/chat/completions", body=_chat_body(),
        )
        assert response.status_code == 200
        assert [request.path for request in backend.journal] == [
            "/prefix/v1/chat/completions",
        ]
    finally:
        if proxy is not None:
            await _stop_proxy(proxy)
        await backend.stop()


class _RecordingManagedClient:
    api_format = "openai"
    model = "managed-recording"

    def __init__(self, first_barrier: threading.Barrier, release: threading.Event) -> None:
        self.first_barrier = first_barrier
        self.release = release
        self.calls = 0
        self.events: list[tuple[str, int, int]] = []
        self.last_usage: dict[int, Any] = {}

    async def send(self, *_args: Any, **_kwargs: Any) -> Any:
        from forge.core.workflow import TextResponse

        self.calls += 1
        call = self.calls
        self.events.append(("start", call, _next_sequence()))
        if call == 1:
            await asyncio.to_thread(self.first_barrier.wait, 10.0)
            released = await asyncio.to_thread(self.release.wait, 10.0)
            assert released
        self.events.append(("end", call, _next_sequence()))
        return TextResponse(content=f"managed-{call}")

    async def aclose(self) -> None:
        return None


class _StubManager:
    def __init__(self) -> None:
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


async def serialization_defaults() -> None:
    # Unmanaged defaults to concurrent.  A real backend barrier requires both
    # calls to arrive before either response can complete.
    barrier = asyncio.Barrier(2)
    unmanaged_events: list[tuple[str, int]] = []

    async def concurrent_behavior(_req: BackendRequest, index: int) -> MockResponse:
        unmanaged_events.append(("start", _next_sequence()))
        await asyncio.wait_for(barrier.wait(), 10.0)
        unmanaged_events.append(("end", _next_sequence()))
        return MockResponse(body=_openai_response(text=f"unmanaged-{index}"))

    async with _external_proxy(
        concurrent_behavior, budget_tokens=8192,
    ) as (_backend, proxy):
        results = await asyncio.gather(*(
            _request(
                proxy, "POST", "/v1/chat/completions",
                body=_chat_body(model=f"unmanaged-{index}"),
            )
            for index in range(2)
        ))
        assert all(r.status_code == 200 for r in results)
        starts = [seq for kind, seq in unmanaged_events if kind == "start"]
        ends = [seq for kind, seq in unmanaged_events if kind == "end"]
        assert len(starts) == 2 and len(ends) == 2 and max(starts) < min(ends)

    # Managed automatic serialization is exercised through a real public
    # ProxyServer while replacing only module-local managed dependencies.
    import forge.proxy.proxy as proxy_module
    from forge.server import _ManagedBackendSetup

    first_barrier = threading.Barrier(2)
    release = threading.Event()
    recording = _RecordingManagedClient(first_barrier, release)
    manager = _StubManager()
    setup_calls: list[dict[str, Any]] = []

    async def fake_setup_backend(**kwargs: Any) -> Any:
        setup_calls.append(kwargs)
        return _ManagedBackendSetup(manager, 8192)

    original_setup = proxy_module._setup_managed_backend
    original_ollama = proxy_module.OllamaClient
    proxy = None
    try:
        proxy_module._setup_managed_backend = fake_setup_backend
        proxy_module.OllamaClient = lambda **_kwargs: recording  # type: ignore[assignment]
        proxy = await _start_proxy(backend="ollama", model="managed-recording")

        tasks = [
            asyncio.create_task(_request(
                proxy, "POST", "/v1/chat/completions",
                body=_chat_body(model=f"managed-{index}"),
            ))
            for index in range(2)
        ]
        # The first backend call and the test rendezvous before release.  The
        # second request is already competing, but the managed queue cannot
        # enter client.send until the first call records its end.
        await asyncio.to_thread(first_barrier.wait, 10.0)
        release.set()
        results = await asyncio.gather(*tasks)
        assert all(r.status_code == 200 for r in results)
        assert [kind for kind, _call, _seq in recording.events] == [
            "start", "end", "start", "end",
        ]
        assert [call for _kind, call, _seq in recording.events] == [1, 1, 2, 2]
        assert len(setup_calls) == 1 and setup_calls[0]["backend"] == "ollama"
    finally:
        if proxy is not None:
            await _stop_proxy(proxy)
        proxy_module._setup_managed_backend = original_setup
        proxy_module.OllamaClient = original_ollama
    assert manager.stopped, "managed stub manager was not stopped"


# ---------------------------------------------------------------------------
# Closed compatibility-matrix coverage ledger


@dataclass(frozen=True)
class CoverageRow:
    key: str
    surface: str
    scenarios: tuple[str, ...] = ()
    owner: str | None = None
    note: str | None = None


def _s(
    key: str, surface: str, *scenarios: str, note: str | None = None,
) -> CoverageRow:
    return CoverageRow(key, surface, scenarios=scenarios, note=note)


def _o(key: str, surface: str, owner: str, note: str | None = None) -> CoverageRow:
    return CoverageRow(key, surface, owner=owner, note=note)


SCENARIOS: dict[str, Callable[[], Awaitable[None]]] = {
    "routes-and-aliases": routes_and_aliases,
    "metadata-forwarding": metadata_forwarding,
    "current-context-reporting": current_context_reporting,
    "generic-fidelity-streaming": generic_fidelity_and_streaming,
    "tool-free-guardrail-bypass": tool_free_guardrail_bypass,
    "specialized-external-adapters": specialized_external_adapters,
    "retry-compaction-exhaustion": retry_compaction_and_exhaustion,
    "guardrail-modes": guardrail_modes,
    "anthropic-inbound-conversion": anthropic_inbound_conversion,
    "anthropic-downstream": anthropic_downstream,
    "anthropic-unpinned-concurrency-retry": anthropic_unpinned_concurrency_retry,
    "credentials-conflicts": credentials_and_conflicts,
    "generic-deferred-discovery": generic_deferred_discovery,
    "vllm-deferred-inference": vllm_deferred_inference,
    "vllm-models-discovery-retry": vllm_models_discovery_retry,
    "vllm-discovery-matrix": vllm_discovery_matrix,
    "eager-static-failure-timing": eager_static_failure_timing,
    "url-normalization-explicit-port": url_normalization_and_explicit_port,
    "serialization-defaults": serialization_defaults,
}


# Coverage keys group the public Proxy contract into stable sections and rows.
# A smoke row names stable scenario ids. Rows unsuitable for deterministic HTTP
# smoke cite an existing test or integration owner. This closed inventory is
# validated at startup, so a stale id or unowned row fails the run clearly.
COVERAGE: tuple[CoverageRow, ...] = (
    # A. Primary Proxy CLI and configuration (A01-A35)
    _o("A01", "flat entry points and ProxyServer", "tests/unit/test_proxy_proxy.py; tests/unit/test_proxy_server.py"),
    _o("A02", "ordinary omitted configuration", "tests/unit/test_proxy_proxy.py; tests/unit/test_server.py"),
    _o("A03", "Python constructor default representation", "tests/unit/test_proxy_proxy.py (Python API)"),
    _s("A04", "managed/unmanaged selection", "vllm-discovery-matrix", "serialization-defaults"),
    _s("A05", "specialized backend names", "specialized-external-adapters"),
    _o("A06", "external openai/anthropic backend profiles", "tests/unit/test_proxy_proxy.py (constructor validation)"),
    _s("A07", "backend selector migration", "anthropic-downstream", "anthropic-inbound-conversion"),
    _o("A08", "valid managed identities", "tests/unit/test_proxy_proxy.py; tests/unit/test_server.py"),
    _o("A09", "profile-invalid identity fields", "tests/unit/test_proxy_proxy.py; tests/unit/test_server.py (malformed configuration)"),
    _s("A10", "unmanaged model fallback/pin", "routes-and-aliases", "generic-fidelity-streaming", "specialized-external-adapters", "vllm-discovery-matrix", "anthropic-downstream", "anthropic-unpinned-concurrency-retry"),
    _o("A11", "omitted backend_port", "tests/unit/test_proxy_proxy.py; tests/unit/test_server.py"),
    _o("A12", "explicit spawned backend port", "tests/unit/test_server.py; scripts/integration_test_proxy.py (real backend)"),
    _s("A13", "explicit unmanaged authority port", "url-normalization-explicit-port"),
    _o("A14", "managed context modes", "tests/unit/test_server.py"),
    _o("A15", "explicit unmanaged budget_mode rejected", "tests/unit/test_proxy_config.py::test_unmanaged_rejects_explicit_budget_mode_even_backend"),
    _o("A16", "positive managed manual budget", "tests/unit/test_server.py"),
    _o("A17", "managed invalid/manual budget timing", "tests/unit/test_server.py (malformed configuration)"),
    _s("A18", "unmanaged reporting denominator", "retry-compaction-exhaustion", "vllm-discovery-matrix", "current-context-reporting"),
    _o("A19", "programmatic extra_flags", "tests/unit/test_proxy_proxy.py; tests/unit/test_server.py"),
    _o("A20", "extra_flags rejected for Ollama/unmanaged", "tests/unit/test_proxy_config.py::test_nonspawned_extra_flags_rejected"),
    _o("A21", "terminal CLI extra_flags grammar", "tests/unit/test_proxy_cli.py; tests/unit/test_proxy_proxy.py"),
    _o("A22", "extra_flags profile conflicts", "tests/unit/test_server.py (malformed configuration)"),
    _o("A23", "Proxy host/listen port", "tests/unit/test_proxy_proxy.py; tests/unit/test_proxy_server.py"),
    _s("A24", "normal serialization selection", "serialization-defaults"),
    _o("A25", "contradictory serialize switches", "tests/unit/test_proxy_cli.py::test_serialization_switches_are_mutually_exclusive"),
    _s("A26", "valid retry/tool-error limits", "retry-compaction-exhaustion"),
    _o("A27", "negative retry/tool-error limits", "tests/unit/test_proxy_config.py::test_nonnegative_retry_controls"),
    _o("A28", "backend_timeout", "tests/unit/test_proxy_proxy.py; tests/unit/test_client_auth.py"),
    _s("A29", "rescue", "guardrail-modes", "tool-free-guardrail-bypass"),
    _s("A30", "backend capability", "guardrail-modes"),
    _s("A31", "synthetic respond tool", "guardrail-modes", "tool-free-guardrail-bypass"),
    _s("A32", "reasoning replay", "guardrail-modes"),
    _s("A33", "backend credential", "credentials-conflicts", "generic-deferred-discovery", "anthropic-downstream"),
    _o("A34", "verbose logging", "src/forge/proxy/__main__.py (current CLI/Python entrypoint; logging-only surface)"),
    _s("A35", "validation timing/error form", "eager-static-failure-timing"),

    # B. Proxy HTTP routes and externally visible runtime (B01-B18)
    _s("B01", "POST /v1/chat/completions", "generic-fidelity-streaming", "tool-free-guardrail-bypass", "specialized-external-adapters"),
    _s("B02", "POST /chat/completions", "routes-and-aliases"),
    _s("B03", "POST /v1/messages", "anthropic-inbound-conversion", "anthropic-downstream"),
    _s("B04", "global OPTIONS/CORS", "routes-and-aliases"),
    _s("B05", "forwarded backend readiness GET /health", "routes-and-aliases"),
    _s("B06", "forwarded GET /v1/health", "routes-and-aliases"),
    _s("B07", "Forge liveness GET /forge/health", "routes-and-aliases", "vllm-discovery-matrix"),
    _s("B08", "honest forwarded GET /v1/models", "routes-and-aliases", "vllm-models-discovery-retry"),
    _s("B09", "forwarded GET /models", "routes-and-aliases", "metadata-forwarding"),
    _s("B10", "forwarded GET /props", "routes-and-aliases", "metadata-forwarding"),
    _s("B11", "last-completed GET /forge/usage", "routes-and-aliases", "current-context-reporting"),
    _s("B12", "unknown /forge/*", "routes-and-aliases"),
    _s("B13", "management routes and /models/sse", "routes-and-aliases"),
    _s("B14", "unknown GETs", "routes-and-aliases"),
    _s("B15", "exact metadata path/query forwarding", "routes-and-aliases", "metadata-forwarding", "vllm-models-discovery-retry"),
    _s("B16", "forwarded response and transport fidelity", "metadata-forwarding"),
    _s("B17", "metadata authentication", "metadata-forwarding", "vllm-models-discovery-retry"),
    _o("B18", "Docker liveness target /forge/health", "Dockerfile"),

    # C. Lifecycle, identity, metadata, context, inference (C01-C35)
    _o("C01", "managed ownership", "tests/unit/test_server.py; scripts/integration_test_proxy.py (real backend)"),
    _o("C02", "managed shutdown", "tests/unit/test_server.py; scripts/integration_test_proxy.py (real backend)"),
    _s("C03", "unmanaged ownership", "routes-and-aliases", "url-normalization-explicit-port"),
    _o("C04", "managed context modes/Ollama tiers", "tests/unit/test_server.py"),
    _o("C05", "managed missing backend context degrades reporting", "tests/unit/test_proxy_proxy.py; tests/unit/test_server.py"),
    _s("C06", "side-effect-free unmanaged startup", "generic-deferred-discovery", "eager-static-failure-timing"),
    _s("C07", "unpinned vLLM static credential", "vllm-discovery-matrix", "eager-static-failure-timing"),
    _s("C08", "unpinned vLLM forwarded credential", "vllm-deferred-inference", "vllm-models-discovery-retry"),
    _s("C09", "pinned vLLM", "vllm-discovery-matrix"),
    _s("C10", "vLLM identity/context separation", "vllm-deferred-inference", "vllm-discovery-matrix"),
    _s("C11", "generic unmanaged missing metadata", "generic-deferred-discovery", "eager-static-failure-timing"),
    _s("C12", "mount-root normalization", "url-normalization-explicit-port"),
    _s("C13", "explicit unmanaged connection-port override", "url-normalization-explicit-port"),
    _s("C14", "generic clean first attempt", "generic-fidelity-streaming", "specialized-external-adapters"),
    _s("C15", "unconditional Proxy no-compaction", "retry-compaction-exhaustion"),
    _s("C16", "serialization/concurrency default", "serialization-defaults"),
    _s("C17", "tool-free inference", "generic-fidelity-streaming", "tool-free-guardrail-bypass"),
    _s("C18", "retry mutation/exhaustion", "retry-compaction-exhaustion", "anthropic-unpinned-concurrency-retry"),
    _s("C19", "buffered streaming", "generic-fidelity-streaming", "anthropic-inbound-conversion", "anthropic-downstream"),
    _s("C20", "generic model fallback", "routes-and-aliases", "generic-fidelity-streaming"),
    _s("C21", "vLLM passthrough/raw-tool fidelity", "vllm-deferred-inference"),
    _o("C22", "managed Ollama translation", "tests/unit/test_ollama_client.py; tests/unit/test_proxy_handler.py"),
    _s("C23", "Anthropic downstream", "anthropic-downstream", "anthropic-unpinned-concurrency-retry"),
    _s("C24", "Anthropic-to-OpenAI conversion", "anthropic-inbound-conversion"),
    _s("C25", "response model", "generic-fidelity-streaming", "vllm-deferred-inference", "anthropic-downstream", "anthropic-unpinned-concurrency-retry"),
    _s("C26", "session usage", "current-context-reporting"),
    _s(
        "C27", "session collision precedence", "current-context-reporting",
        note=(
            "Whitespace/non-string cases: tests/unit/test_proxy_handler.py::"
            "test_request_session_precedence_and_opacity"
        ),
    ),
    _s("C28", "session source", "current-context-reporting"),
    _s("C29", "litellm_session_id passthrough", "generic-fidelity-streaming", "anthropic-inbound-conversion", "anthropic-downstream", "vllm-deferred-inference", "current-context-reporting"),
    _s("C30", "top-level/subagent reporting eligibility", "current-context-reporting"),
    _s(
        "C31", "last-completed process-local snapshot", "routes-and-aliases",
        "current-context-reporting",
        note=(
            "Partial delivery: tests/unit/test_proxy_server.py::"
            "test_partial_buffered_sse_delivery_retains_prior_snapshot"
        ),
    ),
    _s("C32", "request-local snapshot numerator", "current-context-reporting"),
    _s("C33", "snapshot denominator/provenance", "current-context-reporting"),
    _o("C34", "same-model reuse and switch-back refresh", "tests/unit/test_proxy_server.py::test_backend_window_reuses_only_current_exact_model_and_requeries_switchback; tests/unit/test_proxy_server.py::test_overlapping_metadata_finalization_uses_natural_completion_order"),
    _s("C35", "explicit vLLM budget vs identity", "vllm-discovery-matrix", "vllm-deferred-inference"),

    # D. Direct/native public Python APIs (D01-D31)
    _o("D01", "ProxyServer.start/stop/url", "tests/unit/test_proxy_proxy.py; smoke lifecycle helper"),
    _o("D02", "setup_backend", "tests/unit/test_server.py (Python API)"),
    _o("D03", "native identity validation", "tests/unit/test_server.py (Python API)"),
    _o("D04", "native Ollama process/KV no-ops", "tests/unit/test_server.py (Python API)"),
    _o("D05", "native manual budget", "tests/unit/test_server.py (Python API)"),
    _o("D06", "native common-call no-ops", "tests/unit/test_server.py (Python API)"),
    _o("D07", "native vLLM cache/slot/KV rejections", "tests/unit/test_server.py (Python API)"),
    _o("D08", "ServerManager facade", "tests/unit/test_server.py (Python API/real-process doubles)"),
    _o("D09", "WorkflowRunner coupling", "tests/unit/test_runner.py (Python API)"),
    _o("D10", "ContextManager API", "tests/unit/test_context_manager.py; tests/unit/test_context_thresholds.py"),
    _o("D11", "context usage API", "tests/unit/test_context_manager.py (public API owner)"),
    _o("D12", "update_token_count", "tests/unit/test_context_manager.py"),
    _o("D13", "built-in compaction trigger", "tests/unit/test_strategies.py; tests/unit/test_context_manager.py"),
    _o("D14", "CompactEvent values", "tests/unit/test_context_manager.py"),
    _o("D15", "custom CompactStrategy.compact", "tests/unit/test_strategies.py (Python API)"),
    _o("D16", "built-in strategy availability", "tests/unit/test_strategies.py (Python API)"),
    _o("D17", "run_inference/InferenceResult/serialization helpers", "tests/unit/test_inference_passthrough.py; tests/unit/test_runner.py"),
    _o("D18", "LLMClient send/send_stream", "tests/unit/test_clients_base.py; backend-specific client tests"),
    _o("D19", "TokenUsage/last_usage", "tests/unit/test_clients_base.py; backend-specific client tests"),
    _o("D20", "removed discover_backend_metadata Python API", "tests/unit/test_vllm_client.py; tests/unit/test_llamafile_client.py"),
    _s("D21", "ordinary Proxy automatic vLLM identity discovery", "generic-deferred-discovery", "vllm-deferred-inference"),
    _o("D22", "LLMClient.get_context_length", "tests/unit/test_clients_base.py; backend-specific client tests"),
    _o("D23", "Llamafile context getter", "tests/unit/test_llamafile_client.py"),
    _o("D24", "Ollama context getter", "tests/unit/test_ollama_client.py"),
    _o("D25", "generic OpenAI context getter", "tests/unit/test_openai_compat_client.py"),
    _o("D26", "vLLM context getter", "tests/unit/test_vllm_client.py"),
    _o("D27", "vLLM served-name getter", "tests/unit/test_vllm_client.py"),
    _o("D28", "Anthropic context getter", "tests/unit/test_anthropic_client.py"),
    _o("D29", "direct constructors/URL meanings", "backend-specific client tests (Python API)"),
    _o("D30", "direct vLLM passthrough/raw-tool args", "tests/unit/test_vllm_client.py (Python API)"),
    _o("D31", "other direct passthrough behavior", "backend-specific client tests (Python API)"),

    # E. Internal architecture (E01-E09)
    _o("E01", "backend profiles", "tests/unit/test_proxy_proxy.py (current architecture seams)"),
    _o("E02", "endpoint address book", "tests/unit/test_backend_profiles.py; backend-specific client tests"),
    _o("E03", "metadata courier", "tests/unit/test_proxy_server.py (routing seam)"),
    _o("E04", "managed setup primitive", "tests/unit/test_server.py; tests/unit/test_proxy_proxy.py"),
    _o("E05", "pure configuration normalizer", "tests/unit/test_proxy_config.py; tests/unit/test_proxy_proxy.py"),
    _o("E06", "request-scoped usage", "tests/unit/test_inference_passthrough.py::test_usage_capture_is_task_local_for_overlapping_attempts"),
    _o("E07", "vLLM metadata/identity internals", "tests/unit/test_vllm_client.py; tests/unit/test_proxy_handler.py"),
    _o("E08", "one-snapshot state holder", "tests/unit/test_context_manager.py"),
    _o("E09", "startup orphan cleanup", "tests/unit/test_server.py (process doubles); real failure lifecycle requires integration"),
)


def _validate_coverage() -> None:
    expected_keys = (
        [f"A{i:02d}" for i in range(1, 36)]
        + [f"B{i:02d}" for i in range(1, 19)]
        + [f"C{i:02d}" for i in range(1, 36)]
        + [f"D{i:02d}" for i in range(1, 32)]
        + [f"E{i:02d}" for i in range(1, 10)]
    )
    keys = [row.key for row in COVERAGE]
    assert keys == expected_keys, "coverage ledger keys/order do not match the matrix"
    for row in COVERAGE:
        assert bool(row.scenarios) != bool(row.owner), f"{row.key}: choose scenarios or owner"
        for scenario in row.scenarios:
            assert scenario in SCENARIOS, f"{row.key}: unknown scenario {scenario}"
    referenced = {scenario for row in COVERAGE for scenario in row.scenarios}
    assert referenced == set(SCENARIOS), (
        f"unreferenced scenarios: {sorted(set(SCENARIOS) - referenced)}"
    )


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    passed: bool
    elapsed: float
    detail: str = ""


async def _run_scenario(
    scenario_id: str, scenario: Callable[[], Awaitable[None]],
) -> ScenarioResult:
    started = time.monotonic()
    try:
        await asyncio.wait_for(scenario(), timeout=60.0)
    except BaseException as exc:
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return ScenarioResult(scenario_id, False, time.monotonic() - started, detail)
    return ScenarioResult(scenario_id, True, time.monotonic() - started)


async def main() -> int:
    try:
        _validate_coverage()
    except BaseException as exc:
        print(f"[FAIL] coverage-ledger: {type(exc).__name__}: {exc}")
        return 1

    smoke_rows = sum(bool(row.scenarios) for row in COVERAGE)
    owned_rows = len(COVERAGE) - smoke_rows
    print(
        f"Forge 0.9.0 Proxy contract: {len(SCENARIOS)} scenarios; "
        f"matrix rows smoke={smoke_rows} non-smoke-owner={owned_rows}"
    )
    print(
        "[coverage-note] partial-delivery retention, natural completion order, "
        "same-model reuse/switch-back refresh, exact whitespace/non-string "
        "session cases, and request-local overlapping usage remain with their "
        "named deterministic unit owners."
    )

    results: list[ScenarioResult] = []
    for scenario_id, scenario in SCENARIOS.items():
        print(f"[RUN ] {scenario_id}", flush=True)
        result = await _run_scenario(scenario_id, scenario)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        suffix = f" - {result.detail}" if result.detail else ""
        print(f"[{status}] {scenario_id} ({result.elapsed:.2f}s){suffix}", flush=True)

    print("\nScenario summary")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  {status:4} {result.scenario_id:36} {result.elapsed:6.2f}s")
    failures = [result for result in results if not result.passed]
    print(
        f"\nCollected {len(results)} scenarios: "
        f"{len(results) - len(failures)} passed, {len(failures)} failed."
    )
    if failures:
        print("Failing scenario ids: " + ", ".join(r.scenario_id for r in failures))
        return 1
    print("[PASS] Forge 0.9.0 Proxy contract is clean.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
