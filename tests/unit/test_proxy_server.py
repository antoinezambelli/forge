"""Tests for ProxyServer programmatic API."""

import json
import time

import httpx
import pytest

from forge.proxy import ProxyServer


# -- Mock backend for testing ----------------------------------


def _start_mock_backend(port: int = 19876):
    """Start a minimal mock backend on a background thread."""
    import asyncio
    import threading

    async def handle(reader, writer):
        request_line = await reader.readline()
        headers = {}
        content_length = 0
        while True:
            line = await reader.readline()
            if line == b"\r\n" or not line:
                break
            decoded = line.decode().strip()
            if ":" in decoded:
                k, v = decoded.split(":", 1)
                headers[k.strip().lower()] = v.strip()
                if k.strip().lower() == "content-length":
                    content_length = int(v.strip())

        body = b""
        if content_length > 0:
            body = await reader.readexactly(content_length)

        method, path, _ = request_line.decode().strip().split(" ", 2)

        if path == "/v1/models":
            resp_body = json.dumps({
                "data": [{"id": "test-model", "object": "model"}],
            }).encode()
        elif path == "/v1/chat/completions":
            parsed = json.loads(body) if body else {}
            if parsed.get("stream"):
                # Streaming response
                writer.write(b"HTTP/1.1 200 OK\r\n")
                writer.write(b"Content-Type: text/event-stream\r\n\r\n")
                chunks = [
                    {"choices": [{"delta": {"role": "assistant"}}]},
                    {"choices": [{"delta": {"tool_calls": [
                        {"index": 0, "id": "call_0", "type": "function",
                         "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
                    ]}}]},
                    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
                ]
                for c in chunks:
                    writer.write(f"data: {json.dumps(c)}\n\n".encode())
                writer.write(b"data: [DONE]\n\n")
                await writer.drain()
                writer.close()
                return
            else:
                resp_body = json.dumps({
                    "choices": [{"message": {
                        "role": "assistant", "content": None,
                        "tool_calls": [{"id": "call_0", "type": "function",
                                        "function": {"name": "get_weather",
                                                     "arguments": '{"city":"Paris"}'}}],
                    }, "finish_reason": "tool_calls"}],
                }).encode()
        elif path == "/props":
            resp_body = json.dumps({
                "default_generation_settings": {"n_ctx": 4096},
            }).encode()
        elif path == "/health":
            resp_body = json.dumps({"status": "ok"}).encode()
        else:
            resp_body = b"{}"

        writer.write(f"HTTP/1.1 200 OK\r\nContent-Length: {len(resp_body)}\r\nContent-Type: application/json\r\n\r\n".encode())
        writer.write(resp_body)
        await writer.drain()
        writer.close()

    loop = asyncio.new_event_loop()
    server = None
    started = threading.Event()

    async def run():
        nonlocal server
        server = await asyncio.start_server(handle, "127.0.0.1", port)
        started.set()
        await server.serve_forever()

    def target():
        try:
            loop.run_until_complete(run())
        except asyncio.CancelledError:
            pass

    t = threading.Thread(target=target, daemon=True)
    t.start()
    started.wait(timeout=5)
    return loop, server, t


class TestProxyServer:
    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """Start a mock backend and clean up after."""
        self.backend_port = 19876
        self.loop, self.server, self.thread = _start_mock_backend(self.backend_port)
        yield
        self.loop.call_soon_threadsafe(self.server.close)

    def test_start_stop(self):
        proxy = ProxyServer(
            backend_url=f"http://localhost:{self.backend_port}",
            port=19877,
            compact_enabled=False,
        )
        proxy.start()
        try:
            resp = httpx.get(f"{proxy.url}/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}
        finally:
            proxy.stop()

    def test_chat_completion_batch(self):
        proxy = ProxyServer(
            backend_url=f"http://localhost:{self.backend_port}",
            port=19878,
            compact_enabled=False,
        )
        proxy.start()
        try:
            resp = httpx.post(
                f"{proxy.url}/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
                    "stream": False,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        finally:
            proxy.stop()

    def test_model_list(self):
        proxy = ProxyServer(
            backend_url=f"http://localhost:{self.backend_port}",
            port=19879,
            compact_enabled=False,
        )
        proxy.start()
        try:
            resp = httpx.get(f"{proxy.url}/v1/models")
            assert resp.status_code == 200
            assert resp.json()["data"][0]["id"] == "test-model"
        finally:
            proxy.stop()

    def test_double_start_raises(self):
        proxy = ProxyServer(
            backend_url=f"http://localhost:{self.backend_port}",
            port=19880,
            compact_enabled=False,
        )
        proxy.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                proxy.start()
        finally:
            proxy.stop()

    def test_url_property(self):
        proxy = ProxyServer(
            backend_url=f"http://localhost:{self.backend_port}",
            port=19881,
        )
        assert proxy.url == "http://127.0.0.1:19881"


class TestProxyServerValidation:
    def test_requires_backend(self):
        with pytest.raises(ValueError, match="backend_url or backend"):
            ProxyServer()

    def test_rejects_both(self):
        with pytest.raises(ValueError, match="not both"):
            ProxyServer(backend_url="http://x", backend="llamaserver", gguf="x.gguf")

    def test_managed_requires_gguf(self):
        with pytest.raises(ValueError, match="gguf"):
            ProxyServer(backend="llamaserver")
