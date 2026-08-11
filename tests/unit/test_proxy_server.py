"""Tests for proxy HTTP server."""

import asyncio
import gzip
import json
from datetime import datetime, timezone

import pytest
from unittest.mock import AsyncMock, MagicMock

import forge.proxy.server as server_module
from forge._backend_profiles import (
    ClientAdapter,
    MetadataFormat,
    ModelCatalog,
    ModelCatalogEntry,
    parse_vllm_model_catalog,
)
from forge.clients.base import TokenUsage
from forge.context.manager import ContextManager
from forge.context.observations import ContextSession, ContextUsage
from forge.context.strategies import NoCompact
from forge.core.workflow import TextResponse, ToolCall
from forge.errors import BackendError
from forge.proxy.handler import LazyDiscovery, RequestFacts
from forge.proxy.server import HTTPServer


# ── Helpers ──────────────────────────────────────────────────


def _mock_client(response):
    """Create a mock LLMClient that returns the given response."""
    client = AsyncMock()
    client.api_format = "ollama"
    client.model = "mock-model"
    client.send = AsyncMock(return_value=response)
    return client


@pytest.fixture
async def server_factory():
    """Factory fixture that creates an HTTPServer on a random port."""
    servers = []

    async def _make(response, serialize=False):
        client = _mock_client(response)
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
        srv = HTTPServer(
            client=client,
            context_manager=ctx,
            client_adapter=ClientAdapter.LLAMAFILE,
            host="127.0.0.1",
            port=0,  # OS picks a free port
            serialize_requests=serialize,
        )
        await srv.start()
        # Get the actual port from the server
        sock = srv._server.sockets[0]
        port = sock.getsockname()[1]
        servers.append(srv)
        return srv, port

    yield _make

    for srv in servers:
        await srv.stop()


async def _http_request(port, method, path, body=None):
    """Send an HTTP request and return (status, headers_dict, body_str)."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        if body is not None:
            body_bytes = json.dumps(body).encode()
            request = (
                f"{method} {path} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body_bytes)}\r\n"
                f"\r\n"
            ).encode() + body_bytes
        else:
            request = (
                f"{method} {path} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"\r\n"
            ).encode()

        writer.write(request)
        await writer.drain()

        # Read response
        response_data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
        response_str = response_data.decode("utf-8", errors="replace")

        # Parse status line
        lines = response_str.split("\r\n")
        status = int(lines[0].split(" ", 2)[1])

        # Find body (after blank line)
        body_start = response_str.find("\r\n\r\n")
        response_body = response_str[body_start + 4:] if body_start >= 0 else ""

        return status, response_body
    finally:
        writer.close()
        await writer.wait_closed()


async def _raw_http_response(port, method, path, header_lines=()):
    """Send a bodyless request and return status, headers, and exact bytes."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        extra = "".join(f"{line}\r\n" for line in header_lines)
        writer.write(
            (
                f"{method} {path} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"{extra}\r\n"
            ).encode()
        )
        await writer.drain()
        data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
        head, _, body = data.partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        status = int(lines[0].split(" ", 2)[1])
        headers = {
            name.lower(): value.strip()
            for name, value in (line.split(":", 1) for line in lines[1:] if ":" in line)
        }
        return status, headers, body
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.fixture
async def metadata_pair_factory():
    """Create deterministic socket backends and HTTPServer metadata couriers."""
    resources = []

    async def _make(
        responses,
        *,
        mount_suffix="",
        protocol="openai",
        static_key=None,
        timeout=1.0,
        delay=0.0,
        client=None,
        context_manager=None,
        lazy_discovery=None,
        serialize=False,
        client_adapter=ClientAdapter.LLAMAFILE,
    ):
        requests = []

        async def handle_backend(reader, writer):
            request_line = (await reader.readline()).decode("latin-1").rstrip("\r\n")
            _, target, _ = request_line.split(" ", 2)
            request_headers = {}
            while True:
                line = (await reader.readline()).decode("latin-1").rstrip("\r\n")
                if not line:
                    break
                name, value = line.split(":", 1)
                request_headers[name.lower()] = value.strip()
            requests.append((target, request_headers))
            if delay:
                await asyncio.sleep(delay)
            status, response_headers, body = responses.get(
                target,
                (404, {"Content-Type": "text/plain"}, b"backend-miss"),
            )
            reason = {200: "OK", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found", 503: "Service Unavailable"}.get(status, "Result")
            header_block = "".join(
                f"{name}: {value}\r\n" for name, value in response_headers.items()
            )
            writer.write(
                (
                    f"HTTP/1.1 {status} {reason}\r\n"
                    f"{header_block}"
                    f"Content-Length: {len(body)}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("latin-1") + body
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        backend = await asyncio.start_server(handle_backend, "127.0.0.1", 0)
        backend_port = backend.sockets[0].getsockname()[1]
        client = client or _mock_client(TextResponse(content="ok"))
        context_manager = context_manager or ContextManager(
            strategy=NoCompact(), budget_tokens=8192,
        )
        proxy = HTTPServer(
            client=client,
            context_manager=context_manager,
            host="127.0.0.1",
            port=0,
            serialize_requests=serialize,
            backend_protocol=protocol,
            client_adapter=client_adapter,
            backend_api_key_present=bool(static_key),
            lazy_discovery=lazy_discovery,
        )
        proxy._configure_metadata_courier(
            mount_root=f"http://127.0.0.1:{backend_port}{mount_suffix}",
            backend_api_key=static_key,
            timeout=timeout,
            private_catalog_url=(
                f"http://127.0.0.1:{backend_port}{mount_suffix}/v1/models"
                if lazy_discovery is not None
                else None
            ),
            catalog_parser=(
                parse_vllm_model_catalog if lazy_discovery is not None else None
            ),
        )
        await proxy.start()
        proxy_port = proxy._server.sockets[0].getsockname()[1]
        resources.append((proxy, backend))
        return proxy, proxy_port, requests, client, context_manager

    yield _make

    for proxy, backend in resources:
        await proxy.stop()
        backend.close()
        await backend.wait_closed()


async def _sse_request(port, body):
    """Send a streaming request and return list of SSE data lines."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        body_bytes = json.dumps(body).encode()
        request = (
            f"POST /v1/chat/completions HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"\r\n"
        ).encode() + body_bytes

        writer.write(request)
        await writer.drain()

        response_data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
        response_str = response_data.decode("utf-8", errors="replace")

        # Extract SSE data lines from chunked transfer encoding
        data_lines = []
        for line in response_str.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                data_lines.append(line[6:])

        return data_lines
    finally:
        writer.close()
        await writer.wait_closed()


async def _anthropic_backend_server(model):
    client = _mock_client(TextResponse(content="ok"))
    client.model = model
    ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
    srv = HTTPServer(
        client=client,
        context_manager=ctx,
        client_adapter=ClientAdapter.ANTHROPIC,
        host="127.0.0.1",
        port=0,
        serialize_requests=False,
        backend_protocol="anthropic",
    )
    await srv.start()
    return srv, srv._server.sockets[0].getsockname()[1], client


# ── Health & Models ──────────────────────────────────────────


class TestHealthAndModels:
    @pytest.mark.asyncio
    async def test_forge_health_is_exact_local_response(self, server_factory):
        srv, port = await server_factory(TextResponse(content=""))
        status, headers, body = await _raw_http_response(port, "GET", "/forge/health")
        assert status == 200
        assert headers["content-type"] == "application/json"
        assert body == b'{"status":"ok"}'

    @pytest.mark.asyncio
    async def test_old_health_without_courier_is_not_synthetic(self, server_factory):
        srv, port = await server_factory(TextResponse(content=""))
        status, _ = await _http_request(port, "GET", "/health")
        assert status == 502

    @pytest.mark.asyncio
    async def test_not_found(self, server_factory):
        srv, port = await server_factory(TextResponse(content=""))
        status, body = await _http_request(port, "GET", "/nonexistent")
        assert status == 404

    @pytest.mark.asyncio
    async def test_cors_preflight(self, server_factory):
        srv, port = await server_factory(TextResponse(content=""))
        status, _ = await _http_request(port, "OPTIONS", "/v1/chat/completions")
        assert status == 204


# ── Chat Completions ────────────────────────────────────────


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_no_tools_text_response(self, server_factory):
        srv, port = await server_factory(TextResponse(content="Hello!"))
        body = {"messages": [{"role": "user", "content": "hi"}]}
        status, response_body = await _http_request(
            port, "POST", "/v1/chat/completions", body,
        )
        assert status == 200
        data = json.loads(response_body)
        assert data["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_tool_call_response(self, server_factory):
        srv, port = await server_factory(
            [ToolCall(tool="search", args={"q": "test"})],
        )
        body = {
            "messages": [{"role": "user", "content": "search for test"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
        }
        status, response_body = await _http_request(
            port, "POST", "/v1/chat/completions", body,
        )
        assert status == 200
        data = json.loads(response_body)
        tc = data["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, server_factory):
        srv, port = await server_factory(TextResponse(content=""))
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            bad_body = b"not json"
            request = (
                f"POST /v1/chat/completions HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(bad_body)}\r\n"
                f"\r\n"
            ).encode() + bad_body
            writer.write(request)
            await writer.drain()
            response_data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
            assert b"400" in response_data
        finally:
            writer.close()
            await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_invalid_content_length_returns_400(self, server_factory):
        srv, port = await server_factory(TextResponse(content=""))
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            request = (
                f"POST /v1/chat/completions HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: abc\r\n"
                f"\r\n"
            ).encode()
            writer.write(request)
            await writer.drain()
            response_data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
            assert b"400" in response_data
        finally:
            writer.close()
            await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_non_object_body_returns_400(self, server_factory):
        srv, port = await server_factory(TextResponse(content=""))
        # Valid JSON but not an object (array) must be rejected before the
        # handler calls body.get(...), which would otherwise raise.
        status, _ = await _http_request(
            port, "POST", "/v1/chat/completions", body=[1, 2, 3]
        )
        assert status == 400


# ── SSE Streaming ───────────────────────────────────────────


class TestSSEStreaming:
    @pytest.mark.asyncio
    async def test_streaming_text_response(self, server_factory):
        srv, port = await server_factory(TextResponse(content="Hello!"))
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        data_lines = await _sse_request(port, body)
        # Should have content events and [DONE]
        assert "[DONE]" in data_lines
        json_events = [json.loads(d) for d in data_lines if d != "[DONE]"]
        assert len(json_events) > 0

    @pytest.mark.asyncio
    async def test_streaming_tool_call(self, server_factory):
        srv, port = await server_factory(
            [ToolCall(tool="search", args={"q": "x"})],
        )
        body = {
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
            "stream": True,
        }
        data_lines = await _sse_request(port, body)
        assert "[DONE]" in data_lines
        json_events = [json.loads(d) for d in data_lines if d != "[DONE]"]
        # Should have tool call deltas
        has_tool_call = any(
            "tool_calls" in e["choices"][0].get("delta", {})
            for e in json_events
        )
        assert has_tool_call


class TestMissingAnthropicModel:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
    @pytest.mark.parametrize("stream", [False, True])
    async def test_missing_model_returns_http_400_before_dispatch(self, stream, path):
        srv, port, client = await _anthropic_backend_server(None)
        try:
            status, body = await _http_request(
                port,
                "POST",
                path,
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": stream,
                },
            )
            assert status == 400
            assert "No model was supplied" in body
            assert "text/event-stream" not in body
            assert "data:" not in body
            client.send.assert_not_awaited()
        finally:
            await srv.stop()


@pytest.mark.asyncio
async def test_usage_route_is_empty_then_emits_exact_local_snapshot(server_factory):
    srv, port = await server_factory(TextResponse(content="ok"))
    status, headers, body = await _raw_http_response(port, "GET", "/forge/usage")
    assert status == 204
    assert headers["content-length"] == "0"
    assert body == b""

    srv._context_manager.record_published_usage(ContextUsage(
        current_usage_tokens=300,
        context_window_tokens=200,
        model="model-a",
        context_window_source="backend_metadata",
        observed_at=datetime(2026, 8, 6, 18, 42, 17, tzinfo=timezone.utc),
        session=ContextSession("opaque", "claude_code"),
    ))
    status, _, body = await _raw_http_response(port, "GET", "/forge/usage")
    assert status == 200
    assert json.loads(body) == {
        "current_usage_tokens": 300,
        "context_window_tokens": 200,
        "usage_percent": 150.0,
        "model": "model-a",
        "context_window_source": "backend_metadata",
        "observed_at": "2026-08-06T18:42:17Z",
        "session": {"id": "opaque", "source": "claude_code"},
    }


@pytest.mark.asyncio
async def test_delivered_request_publishes_request_owned_usage_and_session():
    client = _mock_client(TextResponse(content="ok"))

    async def send(*args, **kwargs):
        TokenUsage(25, 7, 32)
        return TextResponse(content="ok")

    client.send.side_effect = send
    manager = ContextManager(NoCompact(), budget_tokens=100)
    server = HTTPServer(
        client=client,
        context_manager=manager,
        client_adapter=ClientAdapter.LLAMAFILE,
        host="127.0.0.1",
        port=0,
        serialize_requests=False,
    )
    server._configure_context_reporting(
        managed=False,
        context_window_tokens=100,
        metadata_format=MetadataFormat.NONE,
    )
    await server.start()
    port = server._server.sockets[0].getsockname()[1]
    try:
        status, _ = await _http_request(port, "POST", "/v1/chat/completions", {
            "model": "model-a",
            "messages": [{"role": "user", "content": "hi"}],
            "litellm_session_id": "opaque",
        })
        assert status == 200
        status, _, body = await _raw_http_response(port, "GET", "/forge/usage")
        assert status == 200
        payload = json.loads(body)
        assert payload["current_usage_tokens"] == 25
        assert payload["usage_percent"] == 25.0
        assert payload["session"] == {"id": "opaque", "source": "litellm"}
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_finalize_uses_cache_occupancy_and_operator_precedence():
    manager = ContextManager(NoCompact(), budget_tokens=1000)
    server = HTTPServer(
        client=_mock_client(TextResponse(content="ok")),
        context_manager=manager,
        client_adapter=ClientAdapter.LLAMAFILE,
        serialize_requests=False,
    )
    server._configure_context_reporting(
        managed=False,
        context_window_tokens=1000,
        metadata_format=MetadataFormat.VLLM_MODELS,
        metadata_url="http://must-not-query/v1/models",
    )
    server._fetch_reporting_json = AsyncMock(side_effect=AssertionError("queried"))
    facts = RequestFacts(
        effective_model="model-a",
        session=ContextSession("session", "litellm"),
        completed=True,
        usage=TokenUsage(
            prompt_tokens=5,
            completion_tokens=99,
            total_tokens=104,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=20,
        ),
    )

    await server._finalize_context_report(facts, {}, "openai")

    usage = manager.published_usage
    assert usage.current_usage_tokens == 35
    assert usage.context_window_tokens == 1000
    assert usage.context_window_source == "operator_config"
    assert usage.session == facts.session
    server._fetch_reporting_json.assert_not_awaited()


@pytest.mark.asyncio
async def test_managed_window_uses_managed_provenance_without_metadata_query():
    manager = ContextManager(NoCompact(), budget_tokens=4096)
    server = HTTPServer(
        client=_mock_client(TextResponse(content="ok")),
        context_manager=manager,
        client_adapter=ClientAdapter.LLAMAFILE,
        serialize_requests=False,
    )
    server._configure_context_reporting(
        managed=True,
        context_window_tokens=4096,
        metadata_format=MetadataFormat.LLAMA_PROPERTIES,
        metadata_url="http://must-not-query/props",
    )
    server._fetch_reporting_json = AsyncMock(side_effect=AssertionError("queried"))

    await server._finalize_context_report(RequestFacts(
        effective_model="managed-model",
        completed=True,
        usage=TokenUsage(100, 1, 101),
    ), {}, "openai")

    usage = manager.published_usage
    assert usage.context_window_tokens == 4096
    assert usage.context_window_source == "managed_backend"
    server._fetch_reporting_json.assert_not_awaited()


@pytest.mark.asyncio
async def test_llama_props_and_official_anthropic_exact_metadata_adapters():
    llama_manager = ContextManager(NoCompact(), budget_tokens=None)
    llama = HTTPServer(
        client=_mock_client(TextResponse(content="ok")),
        context_manager=llama_manager,
        client_adapter=ClientAdapter.LLAMAFILE,
        serialize_requests=False,
    )
    llama._configure_context_reporting(
        managed=False,
        context_window_tokens=None,
        metadata_format=MetadataFormat.LLAMA_PROPERTIES,
        metadata_url="http://backend/props",
    )
    llama._fetch_reporting_json = AsyncMock(return_value={
        "default_generation_settings": {"n_ctx": 32768},
    })
    await llama._finalize_context_report(RequestFacts(
        effective_model="llama-model",
        completed=True,
        usage=TokenUsage(12, 0, 12),
    ), {}, "openai")
    assert llama_manager.published_usage.context_window_tokens == 32768
    assert llama_manager.published_usage.context_window_source == (
        "backend_metadata"
    )

    anthropic_client = _mock_client(TextResponse(content="ok"))
    anthropic_client._get_context_length_for_model = AsyncMock(return_value=200000)
    anthropic_manager = ContextManager(NoCompact(), budget_tokens=None)
    anthropic = HTTPServer(
        client=anthropic_client,
        context_manager=anthropic_manager,
        client_adapter=ClientAdapter.ANTHROPIC,
        serialize_requests=False,
        backend_protocol="anthropic",
    )
    anthropic._configure_context_reporting(
        managed=False,
        context_window_tokens=None,
        metadata_format=MetadataFormat.ANTHROPIC_MODELS,
    )
    await anthropic._finalize_context_report(RequestFacts(
        effective_model="claude-exact",
        completed=True,
        usage=TokenUsage(20, 0, 20),
    ), {"x-api-key": "forwarded"}, "anthropic")
    anthropic_client._get_context_length_for_model.assert_awaited_once_with(
        "claude-exact", {"x-api-key": "forwarded"},
    )
    assert anthropic_manager.published_usage.context_window_tokens == 200000
    assert anthropic_manager.published_usage.context_window_source == (
        "backend_metadata"
    )


@pytest.mark.asyncio
async def test_reporting_metadata_failure_after_http_delivery_clears_to_204():
    client = _mock_client(TextResponse(content="ok"))

    async def send(*args, **kwargs):
        TokenUsage(25, 1, 26)
        return TextResponse(content="ok")

    client.send.side_effect = send
    manager = ContextManager(NoCompact(), budget_tokens=None)
    manager.record_published_usage(ContextUsage(
        current_usage_tokens=1,
        context_window_tokens=10,
        model="prior",
        context_window_source="backend_metadata",
        observed_at=datetime.now(timezone.utc),
    ))
    server = HTTPServer(
        client=client,
        context_manager=manager,
        client_adapter=ClientAdapter.LLAMAFILE,
        host="127.0.0.1",
        port=0,
        serialize_requests=False,
    )
    server._configure_context_reporting(
        managed=False,
        context_window_tokens=None,
        metadata_format=MetadataFormat.LLAMA_PROPERTIES,
        metadata_url="http://backend/props",
    )
    server._fetch_reporting_json = AsyncMock(side_effect=BackendError(503))
    await server.start()
    port = server._server.sockets[0].getsockname()[1]
    try:
        status, response_body = await _http_request(
            port,
            "POST",
            "/v1/chat/completions",
            {"model": "new", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert status == 200
        assert json.loads(response_body)["choices"][0]["message"]["content"] == "ok"
        status, _, body = await _raw_http_response(port, "GET", "/forge/usage")
        assert status == 204
        assert body == b""
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_partial_buffered_sse_delivery_retains_prior_snapshot():
    class ClosingWriter:
        def __init__(self):
            self.drains = 0

        def write(self, _payload):
            pass

        async def drain(self):
            self.drains += 1

        def is_closing(self):
            # Header drain succeeds, then the first buffered event disconnects.
            return self.drains >= 2

    manager = ContextManager(NoCompact(), budget_tokens=100)
    prior = ContextUsage(
        current_usage_tokens=1,
        context_window_tokens=100,
        model="prior",
        context_window_source="operator_config",
        observed_at=datetime.now(timezone.utc),
    )
    manager.record_published_usage(prior)
    client = _mock_client(TextResponse(content="two events"))

    async def send(*args, **kwargs):
        TokenUsage(25, 1, 26)
        return TextResponse(content="two events")

    client.send.side_effect = send
    server = HTTPServer(
        client=client,
        context_manager=manager,
        client_adapter=ClientAdapter.LLAMAFILE,
        serialize_requests=False,
    )
    server._configure_context_reporting(
        managed=False,
        context_window_tokens=100,
        metadata_format=MetadataFormat.NONE,
    )

    await server._handle_completions(
        ClosingWriter(),
        json.dumps({
            "model": "new",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }).encode(),
    )

    assert manager.published_usage is prior


@pytest.mark.asyncio
async def test_subagent_retains_prior_and_delivered_unavailable_clears():
    manager = ContextManager(NoCompact(), budget_tokens=None)
    prior = ContextUsage(
        current_usage_tokens=1,
        context_window_tokens=10,
        model="old",
        context_window_source="backend_metadata",
        observed_at=datetime.now(timezone.utc),
    )
    manager.record_published_usage(prior)
    server = HTTPServer(
        client=_mock_client(TextResponse(content="ok")),
        context_manager=manager,
        client_adapter=ClientAdapter.LLAMAFILE,
        serialize_requests=False,
    )
    server._configure_context_reporting(
        managed=False,
        context_window_tokens=None,
        metadata_format=MetadataFormat.NONE,
    )
    subagent = RequestFacts(
        effective_model="new",
        reporting_eligible=False,
        completed=True,
        usage=TokenUsage(2, 0, 2),
    )
    await server._finalize_context_report(subagent, {}, "openai")
    assert manager.published_usage is prior

    eligible = RequestFacts(
        effective_model="new",
        completed=True,
        usage=TokenUsage(2, 0, 2),
    )
    await server._finalize_context_report(eligible, {}, "openai")
    assert manager.published_usage is None


@pytest.mark.asyncio
async def test_backend_window_reuses_only_current_exact_model_and_requeries_switchback():
    manager = ContextManager(NoCompact(), budget_tokens=None)
    server = HTTPServer(
        client=_mock_client(TextResponse(content="ok")),
        context_manager=manager,
        client_adapter=ClientAdapter.LLAMAFILE,
        serialize_requests=False,
    )
    server._configure_context_reporting(
        managed=False,
        context_window_tokens=None,
        metadata_format=MetadataFormat.VLLM_MODELS,
        metadata_url="http://backend/v1/models",
    )
    server._catalog_parser = parse_vllm_model_catalog
    server._fetch_reporting_json = AsyncMock(side_effect=[
        {"data": [{"id": "a", "max_model_len": 100}]},
        {"data": [{"id": "b", "max_model_len": 200}]},
        {"data": [{"id": "a", "max_model_len": 300}]},
    ])

    async def publish(model):
        await server._finalize_context_report(RequestFacts(
            effective_model=model,
            completed=True,
            usage=TokenUsage(1, 0, 1),
        ), {}, "openai")

    await publish("a")
    await publish("a")
    assert server._fetch_reporting_json.await_count == 1
    await publish("b")
    await publish("a")

    assert server._fetch_reporting_json.await_count == 3
    assert manager.published_usage.model == "a"
    assert manager.published_usage.context_window_tokens == 300


@pytest.mark.asyncio
async def test_overlapping_metadata_finalization_uses_natural_completion_order():
    manager = ContextManager(NoCompact(), budget_tokens=None)
    client = _mock_client(TextResponse(content="ok"))
    first_release = asyncio.Event()
    second_release = asyncio.Event()

    async def resolve(model, _headers):
        if model == "first":
            await first_release.wait()
            return 100
        await second_release.wait()
        return 200

    client._get_context_length_for_model = AsyncMock(side_effect=resolve)
    server = HTTPServer(
        client=client,
        context_manager=manager,
        client_adapter=ClientAdapter.ANTHROPIC,
        serialize_requests=False,
        backend_protocol="anthropic",
    )
    server._configure_context_reporting(
        managed=False,
        context_window_tokens=None,
        metadata_format=MetadataFormat.ANTHROPIC_MODELS,
    )

    async def finalize(model):
        await server._finalize_context_report(RequestFacts(
            effective_model=model,
            completed=True,
            usage=TokenUsage(1, 0, 1),
        ), {}, "anthropic")

    first = asyncio.create_task(finalize("first"))
    await asyncio.sleep(0)
    second = asyncio.create_task(finalize("second"))
    await asyncio.sleep(0)
    second_release.set()
    await second
    assert manager.published_usage.model == "second"
    first_release.set()
    await first
    assert manager.published_usage.model == "first"


# ── Serialization ───────────────────────────────────────────


class TestSerialization:
    @pytest.mark.asyncio
    async def test_serialized_requests_processed(self, server_factory):
        """Serialized mode processes requests through the queue."""
        srv, port = await server_factory(
            TextResponse(content="ok"), serialize=True,
        )
        body = {"messages": [{"role": "user", "content": "hi"}]}
        status, response_body = await _http_request(
            port, "POST", "/v1/chat/completions", body,
        )
        assert status == 200
        data = json.loads(response_body)
        assert data["choices"][0]["message"]["content"] == "ok"


# ── Inbound credential threading (v0.8.0) ────────────────────


async def _http_request_with_auth(port, body, auth_header):
    """POST /v1/chat/completions with an extra Authorization header."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        body_bytes = json.dumps(body).encode()
        request = (
            f"POST /v1/chat/completions HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Authorization: {auth_header}\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"\r\n"
        ).encode() + body_bytes
        writer.write(request)
        await writer.drain()
        await asyncio.wait_for(reader.read(65536), timeout=10.0)
    finally:
        writer.close()
        await writer.wait_closed()


async def _auth_server(serialize, backend_api_key_present=False):
    """An HTTPServer fronting an Anthropic-wire backend, with a mock client."""
    client = _mock_client(TextResponse(content="ok"))
    ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
    srv = HTTPServer(
        client=client,
        context_manager=ctx,
        client_adapter=ClientAdapter.ANTHROPIC,
        host="127.0.0.1",
        port=0,
        serialize_requests=serialize,
        backend_protocol="anthropic",
        backend_api_key_present=backend_api_key_present,
    )
    await srv.start()
    port = srv._server.sockets[0].getsockname()[1]
    return srv, port, client


async def _raw_request(port, header_lines, body):
    """POST with arbitrary extra header lines; return (status, body_str)."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        body_bytes = json.dumps(body).encode()
        extra = "".join(f"{line}\r\n" for line in header_lines)
        request = (
            f"POST /v1/chat/completions HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\n"
            f"{extra}"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"\r\n"
        ).encode() + body_bytes
        writer.write(request)
        await writer.drain()
        data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
        text = data.decode("utf-8", errors="replace")
        status = int(text.split("\r\n", 1)[0].split(" ", 2)[1])
        start = text.find("\r\n\r\n")
        return status, (text[start + 4:] if start >= 0 else "")
    finally:
        writer.close()
        await writer.wait_closed()


class TestInboundCredentialThreading:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("serialize", [False, True])
    async def test_inbound_auth_relocated_to_backend_client(self, serialize):
        # Both dispatch paths (direct and the serialized queue worker) must
        # carry the inbound header to the handler. Source openai endpoint →
        # anthropic backend: Bearer stripped, relocated to x-api-key.
        srv, port, client = await _auth_server(serialize)
        try:
            await _http_request_with_auth(
                port,
                {"messages": [{"role": "user", "content": "hi"}]},
                "Bearer INBOUND",
            )
            assert client.send.await_count == 1
            assert client.send.call_args.kwargs["extra_headers"] == {"x-api-key": "INBOUND"}
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_duplicate_auth_header_refused_400_no_secret(self):
        # Two same-name Authorization headers must be refused (never pick a
        # winner), as a 400 client error, with no secret in the response body.
        srv, port, client = await _auth_server(serialize=False)
        try:
            status, resp_body = await _raw_request(
                port,
                ["Authorization: Bearer SECRET-ONE", "Authorization: Bearer SECRET-TWO"],
                {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 400
            assert "SECRET-ONE" not in resp_body
            assert "SECRET-TWO" not in resp_body
            client.send.assert_not_awaited()
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_inbound_plus_static_key_refused_400(self):
        # Inbound credential + configured --backend-api-key → 400 client error.
        srv, port, client = await _auth_server(
            serialize=False, backend_api_key_present=True,
        )
        try:
            status, resp_body = await _raw_request(
                port,
                ["Authorization: Bearer SECRET-INBOUND"],
                {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 400
            assert "SECRET-INBOUND" not in resp_body
            client.send.assert_not_awaited()
        finally:
            await srv.stop()


# ── Deferred discovery → status mapping ─────────────────────


async def _discovery_server(*, side_effect=None, budget=50000):
    """An OpenAI-wire server whose first request resolves vLLM identity."""
    client = _mock_client(TextResponse(content="ok"))
    client.model = "default"
    client._set_model_identity = MagicMock(
        side_effect=lambda model: setattr(client, "model", model),
    )
    ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
    srv = HTTPServer(
        client=client,
        context_manager=ctx,
        client_adapter=ClientAdapter.VLLM,
        host="127.0.0.1",
        port=0,
        serialize_requests=False,
        backend_protocol="openai",
        lazy_discovery=LazyDiscovery(),
    )
    if side_effect is not None:
        srv._fetch_private_catalog = AsyncMock(side_effect=side_effect)
    else:
        srv._fetch_private_catalog = AsyncMock(return_value=ModelCatalog(
            (ModelCatalogEntry("served", budget),),
            first_served_id="served",
        ))
    await srv.start()
    port = srv._server.sockets[0].getsockname()[1]
    return srv, port, client, ctx


class TestMetadataForwarding:
    @pytest.mark.asyncio
    async def test_local_health_never_uses_backend_or_auth(
        self, metadata_pair_factory,
    ):
        _, port, requests, _, _ = await metadata_pair_factory({})
        status, _, body = await _raw_http_response(
            port,
            "GET",
            "/forge/health",
            ["Authorization: Bearer ONE", "x-api-key: TWO"],
        )
        assert (status, body) == (200, b'{"status":"ok"}')
        assert requests == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        ["/health", "/v1/health", "/v1/models", "/models", "/props"],
    )
    async def test_closed_get_allowlist_is_forwarded(self, metadata_pair_factory, path):
        expected_target = f"/prefix{path}?raw=%2f%2F&item=a+b"
        _, port, requests, _, _ = await metadata_pair_factory(
            {expected_target: (200, {"Content-Type": "application/json"}, b"{}")},
            mount_suffix="/prefix?discard=root",
        )
        status, _, body = await _raw_http_response(
            port, "GET", f"{path}?raw=%2f%2F&item=a+b",
        )
        assert (status, body) == (200, b"{}")
        assert requests[0][0] == expected_target

    @pytest.mark.asyncio
    async def test_missing_inbound_query_removes_mount_root_query(
        self, metadata_pair_factory,
    ):
        _, port, requests, _, _ = await metadata_pair_factory(
            {"/prefix/health": (200, {}, b"ready")},
            mount_suffix="/prefix?discard=root",
        )
        status, _, body = await _raw_http_response(port, "GET", "/health")
        assert (status, body) == (200, b"ready")
        assert requests[0][0] == "/prefix/health"

    @pytest.mark.asyncio
    async def test_empty_query_marker_is_preserved(self, metadata_pair_factory):
        _, port, requests, _, _ = await metadata_pair_factory({
            "/health?": (200, {}, b"ready"),
        })
        status, _, body = await _raw_http_response(port, "GET", "/health?")
        assert (status, body) == (200, b"ready")
        assert requests[0][0] == "/health?"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/forge/private"),
            ("GET", "/models/sse"),
            ("GET", "/models/load"),
            ("GET", "/unknown"),
            ("GET", "/%68ealth"),
            ("GET", "/v1/%6dodels"),
            ("POST", "/props"),
            ("POST", "/models"),
            ("DELETE", "/models"),
        ],
    )
    async def test_declined_and_encoded_routes_stay_local_404(
        self, metadata_pair_factory, method, path,
    ):
        _, port, requests, _, _ = await metadata_pair_factory({})
        status, _, _ = await _raw_http_response(port, method, path)
        assert status == 404
        assert requests == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend_status", [401, 403, 404, 503])
    async def test_backend_status_and_raw_body_are_honest(
        self, metadata_pair_factory, backend_status,
    ):
        raw = b'not-json:\x00\xff'
        _, port, _, _, _ = await metadata_pair_factory({
            "/props": (
                backend_status,
                {
                    "Content-Type": "application/octet-stream",
                    "X-Backend-Secret": "omit-me",
                    "Keep-Alive": "timeout=30",
                },
                raw,
            ),
        })
        status, headers, body = await _raw_http_response(port, "GET", "/props")
        assert (status, body) == (backend_status, raw)
        assert headers["content-type"] == "application/octet-stream"
        assert headers["content-length"] == str(len(raw))
        assert headers["connection"] == "close"
        assert headers["access-control-allow-origin"] == "*"
        assert "x-backend-secret" not in headers
        assert "keep-alive" not in headers

    @pytest.mark.asyncio
    @pytest.mark.parametrize("body", [b"", b"{malformed", b"\x00\x80binary"])
    async def test_empty_malformed_and_binary_bodies_are_not_parsed(
        self, metadata_pair_factory, body,
    ):
        _, port, _, _, _ = await metadata_pair_factory({
            "/models": (200, {}, body),
        })
        status, headers, received = await _raw_http_response(port, "GET", "/models")
        assert (status, received) == (200, body)
        assert headers["content-length"] == str(len(body))
        assert "content-type" not in headers

    @pytest.mark.asyncio
    async def test_gzip_is_buffered_as_decoded_bytes_without_encoding_header(
        self, metadata_pair_factory,
    ):
        decoded = b'{"ready":true}'
        compressed = gzip.compress(decoded)
        _, port, _, _, _ = await metadata_pair_factory({
            "/health": (
                200,
                {"Content-Type": "application/json", "Content-Encoding": "gzip"},
                compressed,
            ),
        })
        status, headers, body = await _raw_http_response(port, "GET", "/health")
        assert (status, body) == (200, decoded)
        assert headers["content-length"] == str(len(decoded))
        assert "content-encoding" not in headers

    @pytest.mark.asyncio
    async def test_transport_connection_and_timeout_failures_map_502(
        self, metadata_pair_factory,
    ):
        _, timeout_port, _, _, _ = await metadata_pair_factory(
            {"/health": (200, {}, b"late")}, timeout=0.01, delay=0.1,
        )
        timeout_status, _, timeout_body = await _raw_http_response(
            timeout_port, "GET", "/health",
        )
        assert timeout_status == 502
        assert json.loads(timeout_body)["error"]["type"] == "proxy_error"

        client = _mock_client(TextResponse(content="ok"))
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
        proxy = HTTPServer(
            client=client,
            context_manager=ctx,
            client_adapter=ClientAdapter.LLAMAFILE,
            host="127.0.0.1",
            port=0,
            serialize_requests=False,
        )
        proxy._configure_metadata_courier(
            mount_root="http://127.0.0.1:1", backend_api_key=None, timeout=0.05,
        )
        await proxy.start()
        try:
            proxy_port = proxy._server.sockets[0].getsockname()[1]
            connection_status, _, _ = await _raw_http_response(
                proxy_port, "GET", "/health",
            )
            assert connection_status == 502
        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_auth_conflicts_and_overlapping_requests_do_not_bleed(
        self, metadata_pair_factory,
    ):
        responses = {
            "/models?one": (200, {}, b"one"),
            "/models?two": (200, {}, b"two"),
        }
        _, port, requests, _, _ = await metadata_pair_factory(
            responses, protocol="anthropic",
        )
        results = await asyncio.gather(
            _raw_http_response(
                port, "GET", "/models?one", ["Authorization: Bearer FIRST"],
            ),
            _raw_http_response(port, "GET", "/models?two", ["x-api-key: SECOND"]),
        )
        assert [result[2] for result in results] == [b"one", b"two"]
        auth_by_target = {target: headers for target, headers in requests}
        assert auth_by_target["/models?one"]["x-api-key"] == "FIRST"
        assert auth_by_target["/models?two"]["x-api-key"] == "SECOND"

        conflict_status, _, _ = await _raw_http_response(
            port,
            "GET",
            "/models",
            ["Authorization: Bearer ONE", "x-api-key: TWO"],
        )
        duplicate_status, _, _ = await _raw_http_response(
            port,
            "GET",
            "/models",
            ["Authorization: Bearer ONE", "Authorization: Bearer TWO"],
        )
        assert (conflict_status, duplicate_status) == (400, 400)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("protocol", "static_key", "inbound", "expected_name", "expected_value"),
        [
            ("openai", None, "Authorization: Bearer TOKEN", "authorization", "Bearer TOKEN"),
            ("anthropic", None, "x-api-key: TOKEN", "x-api-key", "TOKEN"),
            ("anthropic", None, "Authorization: Bearer TOKEN", "x-api-key", "TOKEN"),
            ("openai", None, "x-api-key: TOKEN", "authorization", "Bearer TOKEN"),
            ("openai", "STATIC", None, "authorization", "Bearer STATIC"),
            ("anthropic", "STATIC", None, "x-api-key", "STATIC"),
        ],
    )
    async def test_forwarded_credential_matrix_and_static_key(
        self,
        metadata_pair_factory,
        protocol,
        static_key,
        inbound,
        expected_name,
        expected_value,
    ):
        _, port, requests, _, _ = await metadata_pair_factory(
            {"/models": (200, {}, b"ok")},
            protocol=protocol,
            static_key=static_key,
        )
        header_lines = [] if inbound is None else [inbound]
        status, _, _ = await _raw_http_response(
            port, "GET", "/models", header_lines,
        )
        assert status == 200
        assert requests[0][1][expected_name] == expected_value

    @pytest.mark.asyncio
    async def test_static_plus_inbound_refused_and_no_key_is_omitted(
        self, metadata_pair_factory,
    ):
        _, static_port, static_requests, _, _ = await metadata_pair_factory(
            {"/models": (200, {}, b"ok")}, static_key="STATIC",
        )
        status, _, _ = await _raw_http_response(
            static_port, "GET", "/models", ["x-api-key: INBOUND"],
        )
        assert status == 400
        assert static_requests == []

        _, no_key_port, no_key_requests, _, _ = await metadata_pair_factory(
            {"/models": (200, {}, b"ok")},
        )
        status, _, _ = await _raw_http_response(no_key_port, "GET", "/models")
        assert status == 200
        assert "authorization" not in no_key_requests[0][1]
        assert "x-api-key" not in no_key_requests[0][1]

    @pytest.mark.asyncio
    async def test_public_models_does_not_touch_private_discovery_state(
        self, metadata_pair_factory,
    ):
        client = _mock_client(TextResponse(content="ok"))
        client.model = "default"
        client._set_model_identity = MagicMock(
            side_effect=lambda model: setattr(client, "model", model),
        )
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
        lazy = LazyDiscovery()
        raw_catalog = b'{"data":[{"id":"served","max_model_len":64000}]}'
        _, port, requests, _, _ = await metadata_pair_factory(
            {
                "/v1/models": (
                    200,
                    {"Content-Type": "application/json"},
                    raw_catalog,
                ),
            },
            client=client,
            context_manager=ctx,
            lazy_discovery=lazy,
        )

        status, _, body = await _raw_http_response(port, "GET", "/v1/models")
        assert (status, body) == (200, raw_catalog)
        assert [target for target, _ in requests] == ["/v1/models"]
        client._set_model_identity.assert_not_called()
        assert client.model == "default"
        assert ctx.budget_tokens == 8192
        assert lazy.done is False

        completion_status, _ = await _raw_request(
            port, [], {"messages": [{"role": "user", "content": "hi"}]},
        )
        assert completion_status == 200
        assert [target for target, _ in requests] == ["/v1/models", "/v1/models"]
        client._set_model_identity.assert_called_once_with("served")
        assert client.model == "served"
        assert ctx.budget_tokens == 8192
        assert lazy.done is True

    @pytest.mark.asyncio
    async def test_unpinned_auth_budget_matrix_uses_one_private_prefixed_get(
        self, metadata_pair_factory,
    ):
        for static_key in (None, "STATIC"):
            for budget in (None, 4096):
                case = (static_key, budget)
                client = _mock_client(TextResponse(content="ok"))
                client.model = "default"
                client._set_model_identity = MagicMock(
                    side_effect=lambda model: setattr(client, "model", model),
                )
                ctx = ContextManager(strategy=NoCompact(), budget_tokens=budget)
                _, port, requests, _, _ = await metadata_pair_factory(
                    {
                        "/deploy/v1/models": (
                            200,
                            {"Content-Type": "application/json"},
                            b'{"data":[{"id":"served"}]}',
                        ),
                    },
                    mount_suffix="/deploy",
                    static_key=static_key,
                    client=client,
                    context_manager=ctx,
                    lazy_discovery=LazyDiscovery(),
                )
                inbound = (
                    [] if static_key else ["Authorization: Bearer FORWARDED"]
                )
                status, _ = await _raw_request(
                    port,
                    inbound,
                    {"messages": [{"role": "user", "content": "hi"}]},
                )
                assert status == 200, case
                assert [target for target, _ in requests] == [
                    "/deploy/v1/models"
                ], case
                expected = "Bearer STATIC" if static_key else "Bearer FORWARDED"
                assert requests[0][1]["authorization"] == expected, case
                assert client.model == "served", case
                assert ctx.budget_tokens == budget, case

    @pytest.mark.asyncio
    @pytest.mark.parametrize("serialize", [False, True])
    async def test_streaming_predispatch_carries_facts_without_second_query(
        self, metadata_pair_factory, serialize, monkeypatch,
    ):
        client = _mock_client(TextResponse(content="ok"))
        client.model = "default"
        client._set_model_identity = MagicMock(
            side_effect=lambda model: setattr(client, "model", model),
        )
        captured_facts = []

        real_handler = server_module.handle_chat_completions

        async def capture_facts(**kwargs):
            captured_facts.append(kwargs["request_facts"])
            return await real_handler(**kwargs)

        monkeypatch.setattr(server_module, "handle_chat_completions", capture_facts)
        _, port, requests, _, _ = await metadata_pair_factory(
            {
                "/v1/models": (
                    200,
                    {"Content-Type": "application/json"},
                    b'{"data":[{"id":"served","max_model_len":64000}]}',
                ),
            },
            client=client,
            lazy_discovery=LazyDiscovery(),
            serialize=serialize,
            client_adapter=ClientAdapter.VLLM,
        )
        for alias in ("caller-alias-one", "caller-alias-two"):
            status, _ = await _raw_request(
                port,
                [],
                {
                    "model": alias,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
            assert status == 200
        assert [target for target, _ in requests] == ["/v1/models"]
        assert [facts.effective_model for facts in captured_facts] == [
            "served",
            "served",
        ]
        assert captured_facts[0] is not captured_facts[1]
        assert captured_facts[0].model_catalog is not None
        assert captured_facts[1].model_catalog is None

    @pytest.mark.asyncio
    async def test_missing_first_identity_is_502_and_does_not_latch(
        self, metadata_pair_factory,
    ):
        client = _mock_client(TextResponse(content="ok"))
        client.model = "default"
        client._set_model_identity = MagicMock()
        lazy = LazyDiscovery()
        _, port, requests, _, _ = await metadata_pair_factory(
            {
                "/v1/models": (
                    200,
                    {"Content-Type": "application/json"},
                    b'{"data":[{"max_model_len":64000},{"id":"later"}]}',
                ),
            },
            client=client,
            lazy_discovery=lazy,
        )
        for _ in range(2):
            status, _ = await _raw_request(
                port, [], {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 502
        assert len(requests) == 2
        assert lazy.done is False
        assert client.model == "default"
        client.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_courier_closes_on_stop_and_bind_failure(self):
        client = _mock_client(TextResponse(content="ok"))
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
        server = HTTPServer(
            client=client,
            context_manager=ctx,
            client_adapter=ClientAdapter.LLAMAFILE,
            host="127.0.0.1",
            port=0,
            serialize_requests=False,
        )
        server._configure_metadata_courier(
            mount_root="http://127.0.0.1:1", backend_api_key=None, timeout=1.0,
        )
        courier = server._metadata_courier
        await server.start()
        await server.stop()
        await server.stop()
        assert courier._http.is_closed

        occupied = await asyncio.start_server(lambda _reader, _writer: None, "127.0.0.1", 0)
        occupied_port = occupied.sockets[0].getsockname()[1]
        failed = HTTPServer(
            client=client,
            context_manager=ctx,
            client_adapter=ClientAdapter.LLAMAFILE,
            host="127.0.0.1",
            port=occupied_port,
            serialize_requests=True,
        )
        failed._configure_metadata_courier(
            mount_root="http://127.0.0.1:1", backend_api_key=None, timeout=1.0,
        )
        failed_courier = failed._metadata_courier
        try:
            with pytest.raises(OSError):
                await failed.start()
            assert failed._worker_task is None
            assert failed_courier._http.is_closed
        finally:
            occupied.close()
            await occupied.wait_closed()


class TestDeferredDiscoveryStatusMapping:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("backend_status", "expected_status"),
        [(401, 401), (502, 502)],
        ids=["auth-rejection", "backend-fault"],
    )
    async def test_discovery_errors_map_to_http_status(
        self, backend_status, expected_status,
    ):
        srv, port, client, _ = await _discovery_server(side_effect=BackendError(
            backend_status, "discovery failed",
        ))
        try:
            status, _ = await _raw_request(
                port, [], {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == expected_status
            client.send.assert_not_awaited()
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_success_adopts_identity_without_applying_window(self):
        srv, port, client, ctx = await _discovery_server(budget=50000)
        try:
            status, _ = await _raw_request(
                port, [], {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 200
            assert ctx.budget_tokens == 8192
            assert client.model == "served"
            client._set_model_identity.assert_called_once_with("served")
            srv._fetch_private_catalog.assert_awaited_once()
            discovery_headers = srv._fetch_private_catalog.await_args.args[0]
            assert "authorization" not in discovery_headers
            assert "x-api-key" not in discovery_headers
        finally:
            await srv.stop()


# ── Backend-error status, credential hygiene, and CORS ──────


async def _error_server(exc):
    """An openai-wire server whose backend client.send raises ``exc``."""
    client = _mock_client(TextResponse(content="x"))
    client.send = AsyncMock(side_effect=exc)
    ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
    srv = HTTPServer(
        client=client, context_manager=ctx, host="127.0.0.1", port=0,
        client_adapter=ClientAdapter.LLAMAFILE,
        serialize_requests=False, backend_protocol="openai",
    )
    await srv.start()
    return srv, srv._server.sockets[0].getsockname()[1]


async def _raw_response(port, method, path):
    """Return the full raw HTTP response string for a bodyless request."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(
            f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode()
        )
        await writer.drain()
        data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
        return data.decode("utf-8", errors="replace")
    finally:
        writer.close()
        await writer.wait_closed()


class TestBackendErrorStatusMapping:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("backend_status", "expected_status"),
        [(401, 401), (403, 401), (500, 502)],
    )
    async def test_backend_dispatch_errors_map_to_http_status(
        self, backend_status, expected_status,
    ):
        srv, port = await _error_server(BackendError(backend_status, "failed"))
        try:
            status, _ = await _raw_request(
                port, [], {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == expected_status
        finally:
            await srv.stop()


class TestSecretHygiene:
    @pytest.mark.asyncio
    async def test_backend_error_body_secret_not_leaked_to_client(self):
        # A backend that echoes the inbound auth header in its raw error body
        # must not leak it: the raw body rides exc.body (off the message), so the
        # proxy returns only the safe "Backend returned <status>" summary.
        srv, port = await _error_server(
            BackendError(500, raw_body="rejected Authorization: Bearer sk-leak-7777"),
        )
        try:
            _, body = await _raw_request(
                port, [], {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert "sk-leak-7777" not in body
            assert "Backend returned 500" in body  # safe status summary only
        finally:
            await srv.stop()


class TestStreamingErrorStatus:
    """Pre-dispatch errors on a streaming request return a real HTTP status,
    not a 200 + SSE error event (the SSE header is flushed only after the
    credential + first-request discovery checks pass)."""

    @pytest.mark.asyncio
    async def test_streaming_duplicate_auth_returns_400_not_200(self):
        srv, port, client = await _auth_server(serialize=False)
        try:
            status, body = await _raw_request(
                port,
                ["Authorization: Bearer SECRET-ONE", "Authorization: Bearer SECRET-TWO"],
                {"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert status == 400  # real status, not 200 + an SSE error event
            assert "SECRET-ONE" not in body and "SECRET-TWO" not in body
            client.send.assert_not_awaited()
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_streaming_discovery_failure_returns_401_not_200(self):
        srv, port, client, _ = await _discovery_server(
            side_effect=BackendError(401, "unauthorized"),
        )
        try:
            status, _ = await _raw_request(
                port, [],
                {"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert status == 401  # deferred-discovery 401 surfaces before the stream
            client.send.assert_not_awaited()
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_streaming_success_still_sse_200(self):
        # The success path is unchanged: header flushes and SSE events stream.
        srv, port, client = await _auth_server(serialize=False)
        try:
            status, body = await _raw_request(
                port,
                ["Authorization: Bearer GOODKEY"],
                {"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert status == 200
            assert "data:" in body
        finally:
            await srv.stop()


class TestChatCompletionsAlias:
    """POST /chat/completions (no /v1 prefix) is served like the canonical
    route — llama.cpp serves both spellings and llama.cpp-native clients
    (pi-llama-cpp) POST the unprefixed one."""

    @pytest.mark.asyncio
    async def test_unprefixed_chat_completions_routes(self, server_factory):
        srv, port = await server_factory(TextResponse(content="Hello!"))
        body = {"messages": [{"role": "user", "content": "hi"}]}
        status, response_body = await _http_request(
            port, "POST", "/chat/completions", body,
        )
        assert status == 200
        data = json.loads(response_body)
        assert data["choices"][0]["message"]["content"] == "Hello!"


class TestCorsAllowsApiKey:
    @pytest.mark.asyncio
    async def test_preflight_allows_x_api_key(self):
        srv, _, _ = await _auth_server(serialize=False)
        port = srv._server.sockets[0].getsockname()[1]
        try:
            raw = await _raw_response(port, "OPTIONS", "/v1/chat/completions")
            assert raw.startswith("HTTP/1.1 204")
            allow = next(
                line for line in raw.splitlines()
                if line.lower().startswith("access-control-allow-headers")
            )
            assert "x-api-key" in allow.lower()
        finally:
            await srv.stop()
