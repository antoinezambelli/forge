"""Tests for the proxy server -- Phase 1 (pass-through)."""

import json

import httpx
import pytest

from forge.proxy.handler import ChatHandler
from forge.proxy.server import create_app
from forge.proxy.stream import BufferedStream, buffer_sse_stream, iter_sse_bytes


# -- SSE stream utilities --------------------------------------


class TestBufferedStream:
    def test_iter_sse_bytes_formats_chunks(self):
        buf = BufferedStream(
            chunks=['data: {"id":"1"}', 'data: {"id":"2"}', "data: [DONE]"],
            complete=True,
        )
        result = iter_sse_bytes(buf)
        assert len(result) == 3
        assert result[0] == b'data: {"id":"1"}\n\n'
        assert result[1] == b'data: {"id":"2"}\n\n'
        assert result[2] == b"data: [DONE]\n\n"

    def test_empty_stream(self):
        buf = BufferedStream()
        assert iter_sse_bytes(buf) == []


class TestBufferSseStream:
    @pytest.fixture
    def sse_lines(self):
        return [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"hello"}}]}',
            "",
            "data: [DONE]",
            "",
        ]

    async def test_buffers_sse_lines(self, sse_lines):
        """buffer_sse_stream extracts data: lines and detects [DONE]."""

        class FakeResponse:
            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

        buf = await buffer_sse_stream(FakeResponse())
        assert len(buf.chunks) == 3
        assert buf.complete is True
        assert buf.chunks[0] == 'data: {"choices":[{"delta":{"role":"assistant"}}]}'
        assert buf.chunks[-1] == "data: [DONE]"

    async def test_incomplete_stream(self):
        """Stream without [DONE] marks complete=False."""

        class FakeResponse:
            async def aiter_lines(self):
                yield 'data: {"choices":[]}'
                yield ""

        buf = await buffer_sse_stream(FakeResponse())
        assert buf.complete is False
        assert len(buf.chunks) == 1


# -- Mock backend as raw ASGI app ------------------------------


def _mock_backend_app():
    """Minimal ASGI app that acts as a model server."""

    tool_call_response = {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_0",
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
    }

    streaming_chunks = [
        {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {"tool_calls": [
             {"id": "call_0", "type": "function",
              "function": {"name": "get_weather", "arguments": ""}}]}}]},
        {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {"tool_calls": [
             {"id": "call_0", "type": "function",
              "function": {"arguments": '{"city": "Paris"}'}}]}}]},
        {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    ]

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            msg = await receive()
            if msg["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
            return

        if scope["type"] != "http":
            return

        path = scope["path"]

        if path == "/v1/chat/completions":
            # Read body
            body = b""
            while True:
                msg = await receive()
                body += msg.get("body", b"")
                if not msg.get("more_body", False):
                    break
            parsed = json.loads(body)
            is_streaming = parsed.get("stream", False)

            if is_streaming:
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [[b"content-type", b"text/event-stream"]],
                })
                for chunk in streaming_chunks:
                    line = f"data: {json.dumps(chunk)}\n\n".encode()
                    await send({"type": "http.response.body", "body": line, "more_body": True})
                await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": True})
                await send({"type": "http.response.body", "body": b""})
            else:
                resp = json.dumps(tool_call_response).encode()
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        [b"content-type", b"application/json"],
                        [b"content-length", str(len(resp)).encode()],
                    ],
                })
                await send({"type": "http.response.body", "body": resp})

        elif path == "/v1/models":
            resp = json.dumps({
                "object": "list",
                "data": [{"id": "test-model", "object": "model"}],
            }).encode()
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(resp)).encode()],
                ],
            })
            await send({"type": "http.response.body", "body": resp})

        elif path == "/v1/custom":
            resp = json.dumps({"custom": True}).encode()
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(resp)).encode()],
                ],
            })
            await send({"type": "http.response.body", "body": resp})

    return app


def _make_handler_with_mock():
    """Create a ChatHandler wired to the mock backend."""
    transport = httpx.ASGITransport(app=_mock_backend_app())
    handler = ChatHandler(backend_url="http://mockbackend")
    handler._client = httpx.AsyncClient(transport=transport)
    return handler


# -- Handler tests (core proxy logic) -------------------------


class TestProxyPassthrough:
    async def test_streaming_passthrough(self):
        """Streaming request is buffered and replayed."""
        handler = _make_handler_with_mock()
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "stream": True,
        }

        chunks, status = await handler.handle(body)
        assert status == 200
        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.endswith(b"\n\n")
            text = chunk.decode().strip()
            assert text.startswith("data: ")

        assert chunks[-1] == b"data: [DONE]\n\n"

        all_text = b"".join(chunks).decode()
        assert "get_weather" in all_text
        assert "Paris" in all_text

    async def test_batch_passthrough(self):
        """Non-streaming request is forwarded and returned."""
        handler = _make_handler_with_mock()
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }

        chunks, status = await handler.handle(body)
        assert status == 200
        assert len(chunks) == 1

        response = json.loads(chunks[0])
        assert response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

    async def test_backend_error_forwarded(self):
        """Backend errors are forwarded to the client."""

        async def error_app(scope, receive, send):
            if scope["type"] != "http":
                return
            body = b""
            while True:
                msg = await receive()
                body += msg.get("body", b"")
                if not msg.get("more_body", False):
                    break
            resp = json.dumps({"error": {"message": "model not found"}}).encode()
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(resp)).encode()],
                ],
            })
            await send({"type": "http.response.body", "body": resp})

        transport = httpx.ASGITransport(app=error_app)
        handler = ChatHandler(backend_url="http://mockbackend")
        handler._client = httpx.AsyncClient(transport=transport)

        chunks, status = await handler.handle({
            "model": "nonexistent",
            "messages": [],
            "stream": True,
        })
        assert status == 404


# -- ASGI app tests (endpoint routing) ------------------------


class TestProxyApp:
    async def test_health_endpoint(self):
        """Health endpoint works through the full ASGI app.

        Uses lifespan protocol to initialize the app before sending
        requests, matching how uvicorn would run it.
        """
        app = create_app(
            backend_url="http://mockbackend",
            compact_enabled=False,
        )

        # Simulate lifespan startup (handler init needs async)
        proxy_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=proxy_transport, base_url="http://proxy") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}


# -- CLI tests -------------------------------------------------


class TestProxyCli:
    def test_cli_requires_backend_argument(self):
        """CLI exits with error if neither --backend-url nor --backend provided."""
        import subprocess
        result = subprocess.run(
            ["python", "-m", "forge.proxy"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "backend" in result.stderr.lower()

    def test_cli_managed_requires_gguf(self):
        """CLI exits with error if --backend without --gguf."""
        import subprocess
        result = subprocess.run(
            ["python", "-m", "forge.proxy", "--backend", "llamaserver"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode != 0
        assert "gguf" in result.stderr.lower()
