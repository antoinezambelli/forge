"""Tests for the proxy handler -- guardrail validation and compaction."""

import json

import httpx
import pytest

from forge.context.manager import ContextManager
from forge.context.strategies import TieredCompact
from forge.proxy.handler import ChatHandler


# -- Helper: build mock backends that return scripted responses --


def _mock_backend(responses):
    """ASGI app that returns a sequence of responses.

    Each entry in `responses` is either:
      - dict with "tool_calls" key -> streamed tool call response
      - str -> streamed text response
    """
    call_count = {"n": 0}

    async def app(scope, receive, send):
        if scope["type"] != "http":
            return

        # Read request body
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body", False):
                break

        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        resp = responses[idx]

        parsed = json.loads(body)
        is_streaming = parsed.get("stream", False)

        if isinstance(resp, str):
            # Text response
            if is_streaming:
                await _send_sse_text(send, resp)
            else:
                await _send_batch_text(send, resp)
        else:
            # Tool call response
            if is_streaming:
                await _send_sse_tool_calls(send, resp["tool_calls"])
            else:
                await _send_batch_tool_calls(send, resp["tool_calls"])

    return app, call_count


async def _send_sse_text(send, content):
    await send({"type": "http.response.start", "status": 200,
                "headers": [[b"content-type", b"text/event-stream"]]})
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": content}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    for c in chunks:
        await send({"type": "http.response.body",
                     "body": f"data: {json.dumps(c)}\n\n".encode(), "more_body": True})
    await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": True})
    await send({"type": "http.response.body", "body": b""})


async def _send_sse_tool_calls(send, tool_calls):
    await send({"type": "http.response.start", "status": 200,
                "headers": [[b"content-type", b"text/event-stream"]]})
    chunks = [{"choices": [{"delta": {"role": "assistant"}}]}]
    for i, tc in enumerate(tool_calls):
        chunks.append({"choices": [{"delta": {"tool_calls": [{
            "index": i, "id": f"call_{i}", "type": "function",
            "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
        }]}}]})
    chunks.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    for c in chunks:
        await send({"type": "http.response.body",
                     "body": f"data: {json.dumps(c)}\n\n".encode(), "more_body": True})
    await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": True})
    await send({"type": "http.response.body", "body": b""})


async def _send_batch_text(send, content):
    resp = json.dumps({
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }).encode()
    await send({"type": "http.response.start", "status": 200,
                "headers": [[b"content-type", b"application/json"],
                             [b"content-length", str(len(resp)).encode()]]})
    await send({"type": "http.response.body", "body": resp})


async def _send_batch_tool_calls(send, tool_calls):
    resp = json.dumps({
        "choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [
                {"id": f"call_{i}", "type": "function",
                 "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                for i, tc in enumerate(tool_calls)
            ],
        }, "finish_reason": "tool_calls"}],
    }).encode()
    await send({"type": "http.response.start", "status": 200,
                "headers": [[b"content-type", b"application/json"],
                             [b"content-length", str(len(resp)).encode()]]})
    await send({"type": "http.response.body", "body": resp})


def _make_handler(responses, **kwargs):
    backend, call_count = _mock_backend(responses)
    transport = httpx.ASGITransport(app=backend)
    handler = ChatHandler(backend_url="http://mock", **kwargs)
    handler._client = httpx.AsyncClient(transport=transport)
    return handler, call_count


TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]


def _request(stream=True, tools=None):
    return {
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": TOOLS if tools is None else tools,
        "stream": stream,
    }


# -- Tests: valid responses pass through -----------------------


class TestValidPassthrough:
    async def test_valid_tool_call_streams_through(self):
        handler, counts = _make_handler([
            {"tool_calls": [{"name": "get_weather", "args": {"city": "Paris"}}]},
        ])
        chunks, status = await handler.handle(_request(stream=True))
        assert status == 200
        assert counts["n"] == 1  # single request, no retry
        all_text = b"".join(chunks).decode()
        assert "get_weather" in all_text

    async def test_valid_tool_call_batch(self):
        handler, counts = _make_handler([
            {"tool_calls": [{"name": "get_weather", "args": {"city": "Paris"}}]},
        ])
        chunks, status = await handler.handle(_request(stream=False))
        assert status == 200
        assert counts["n"] == 1
        data = json.loads(chunks[0])
        assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

    async def test_no_tools_passes_through_without_validation(self):
        handler, counts = _make_handler(["Just some text"])
        chunks, status = await handler.handle(_request(stream=True, tools=[]))
        assert status == 200
        assert counts["n"] == 1  # no retry even though response is text


# -- Tests: retry on text response -----------------------------


class TestRetryOnText:
    async def test_retries_on_text_then_succeeds(self):
        """Model returns text on first try, valid tool call on second."""
        handler, counts = _make_handler([
            "I think the answer is 42",
            {"tool_calls": [{"name": "get_weather", "args": {"city": "Paris"}}]},
        ])
        chunks, status = await handler.handle(_request(stream=True))
        assert status == 200
        assert counts["n"] == 2  # one retry
        all_text = b"".join(chunks).decode()
        assert "get_weather" in all_text

    async def test_retries_exhausted_returns_last_response(self):
        """Model keeps returning text, proxy gives up after max_retries."""
        handler, counts = _make_handler([
            "bad 1", "bad 2", "bad 3", "bad 4",
        ], max_retries=2)
        chunks, status = await handler.handle(_request(stream=True))
        assert status == 200
        assert counts["n"] == 3  # initial + 2 retries

    async def test_retry_batch_mode(self):
        handler, counts = _make_handler([
            "not a tool call",
            {"tool_calls": [{"name": "get_weather", "args": {}}]},
        ])
        chunks, status = await handler.handle(_request(stream=False))
        assert status == 200
        assert counts["n"] == 2


# -- Tests: rescue from text ----------------------------------


class TestRescue:
    async def test_rescues_tool_call_from_text(self):
        """Model returns tool call as text, proxy rescues it."""
        text_with_json = '```json\n{"tool": "get_weather", "args": {"city": "Paris"}}\n```'
        handler, counts = _make_handler([text_with_json])
        chunks, status = await handler.handle(_request(stream=True))
        assert status == 200
        assert counts["n"] == 1  # no retry needed, rescued
        all_text = b"".join(chunks).decode()
        assert "get_weather" in all_text
        # Should be synthesized SSE, not raw text
        assert "tool_calls" in all_text

    async def test_rescue_disabled_retries_instead(self):
        text_with_json = '```json\n{"tool": "get_weather", "args": {"city": "Paris"}}\n```'
        handler, counts = _make_handler([
            text_with_json,
            {"tool_calls": [{"name": "get_weather", "args": {"city": "Paris"}}]},
        ], rescue_enabled=False)
        chunks, status = await handler.handle(_request(stream=True))
        assert status == 200
        assert counts["n"] == 2  # couldn't rescue, had to retry


# -- Tests: unknown tool retry ---------------------------------


class TestUnknownTool:
    async def test_retries_on_unknown_tool(self):
        """Model calls a tool not in the tools array."""
        handler, counts = _make_handler([
            {"tool_calls": [{"name": "nonexistent", "args": {}}]},
            {"tool_calls": [{"name": "get_weather", "args": {"city": "Paris"}}]},
        ])
        chunks, status = await handler.handle(_request(stream=True))
        assert status == 200
        assert counts["n"] == 2
        all_text = b"".join(chunks).decode()
        assert "get_weather" in all_text


# -- Tests: context compaction ---------------------------------


class TestCompaction:
    def _make_handler_with_compaction(self, responses, budget_tokens, **kwargs):
        """Create a handler with a ContextManager attached."""
        backend, call_count = _mock_backend(responses)
        transport = httpx.ASGITransport(app=backend)
        ctx = ContextManager(
            strategy=TieredCompact(keep_recent=1),
            budget_tokens=budget_tokens,
        )
        handler = ChatHandler(
            backend_url="http://mock",
            context_manager=ctx,
            **kwargs,
        )
        handler._client = httpx.AsyncClient(transport=transport)
        return handler, call_count

    async def test_compaction_runs_before_backend_request(self):
        """Handler compacts messages before sending to backend."""
        handler, counts = self._make_handler_with_compaction(
            [{"tool_calls": [{"name": "get_weather", "args": {}}]}],
            budget_tokens=200,  # very tight budget to trigger compaction
        )
        # Build a request with many messages to exceed budget
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's the weather?"},
        ]
        # Add several tool call/result pairs to bulk up the history
        for i in range(10):
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": f"c{i}", "type": "function",
                                "function": {"name": "search",
                                             "arguments": json.dumps({"q": "x" * 200})}}],
            })
            messages.append({
                "role": "tool", "content": "result " * 50,
                "name": "search", "tool_call_id": f"c{i}",
            })

        body = {
            "model": "test",
            "messages": messages,
            "tools": TOOLS,
            "stream": True,
        }
        chunks, status = await handler.handle(body)
        assert status == 200
        assert counts["n"] >= 1

    async def test_no_compaction_when_disabled(self):
        """Handler without context_manager skips compaction."""
        handler, counts = _make_handler([
            {"tool_calls": [{"name": "get_weather", "args": {}}]},
        ])
        # handler has no context_manager (default None)
        assert handler.context_manager is None
        chunks, status = await handler.handle(_request(stream=True))
        assert status == 200
        assert counts["n"] == 1
