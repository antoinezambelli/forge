"""Text retries through real parsers, canonical history, and backend lowering."""

import json

import httpx
import pytest

from forge._backend_profiles import ClientAdapter
from forge.clients.llamafile import LlamafileClient
from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.runner import WorkflowRunner
from forge.core.workflow import ToolDefinition, Workflow
from forge.proxy.handler import handle_chat_completions
from tests.http_mocks import build_with_mock_http


SCHEMA = {"type": "object", "properties": {}}


def _tool_reply(name, mode):
    if mode == "prompt":
        return {"content": json.dumps({"tool": name, "args": {}})}
    return {
        "content": "",
        "tool_calls": [{
            "id": "backend-call",
            "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }],
    }


def _client(mode, replies, requests):
    def respond(request):
        body = json.loads(request.content)
        message = replies[len(requests)]
        requests.append(body)
        if body.get("stream"):
            delta = dict(message)
            if delta.get("tool_calls"):
                delta["tool_calls"] = [
                    {**call, "index": i} for i, call in enumerate(delta["tool_calls"])
                ]
            content = "data: " + json.dumps({"choices": [{"delta": delta}]})
            return httpx.Response(
                200, text=content + "\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json={"choices": [{"message": message}]})

    client = build_with_mock_http(
        LlamafileClient, "model.gguf", base_url="http://backend/v1", mode=mode
    )
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return client


def _assert_text_retries(messages):
    assert messages[1]["content"] == "original question"
    tail = messages[-4:]
    assert [m["role"] for m in tail] == ["assistant", "user", "assistant", "user"]
    assert tail[0]["content"] == "alpha"
    assert tail[2]["content"] == "beta"
    assert "tool" in tail[1]["content"].lower()
    assert "tool" in tail[3]["content"].lower()
    assert tail[0]["reasoning_content"] == "reason alpha"
    assert tail[2]["reasoning_content"] == "reason beta"


@pytest.mark.parametrize("mode", ["native", "prompt"])
@pytest.mark.parametrize(
    ("stream", "field"),
    [(True, {}), (True, {"tool_calls": []}),
     (False, {}), (False, {"tool_calls": None}), (False, {"tool_calls": []})],
    ids=["stream-missing", "stream-empty-delta", "buffer-missing", "buffer-null", "buffer-empty"],
)
async def test_runner_text_retries_preserve_next_wire_history(mode, stream, field):
    replies = [
        _tool_reply("read", mode),
        {"content": "alpha", "reasoning_content": "reason alpha", **field},
        {"content": "beta", "reasoning_content": "reason beta", **field},
        _tool_reply("finish", mode),
    ]
    requests = []
    client = _client(mode, replies, requests)
    workflow = Workflow(
        name="text-retry", description="text retry boundary",
        tools={name: ToolDefinition(name, input_schema=SCHEMA) for name in ("read", "finish")},
        handlers={"read": lambda: "read result", "finish": lambda: "finished"},
        required_steps=["read"], terminal_tool="finish",
        system_prompt_template="Use read, then finish.",
    )
    try:
        result = await WorkflowRunner(
            client, ContextManager(NoCompact(), 8192), stream=stream,
            max_iterations=8, reasoning_replay="full",
        ).run(workflow, "original question")
    finally:
        await client.aclose()

    assert result == "finished"
    assert len(requests) == 4
    _assert_text_retries(requests[-1]["messages"])
    for message in requests[-1]["messages"][-4::2]:
        if not stream and field == {"tool_calls": []}:
            assert message["tool_calls"] == []
        else:
            assert "tool_calls" not in message
    if mode == "native":
        call, result = requests[-1]["messages"][2:4]
        assert result["tool_call_id"] == call["tool_calls"][0]["id"]


@pytest.mark.parametrize(
    ("protocol", "mode", "caller_stream"),
    [("openai", "native", False), ("anthropic", "prompt", True),
     ("openai", "prompt", True), ("anthropic", "native", False)],
)
async def test_proxy_buffered_empty_calls_preserve_retry_history(protocol, mode, caller_stream):
    replies = [
        {"content": text, "reasoning_content": "reason " + text, "tool_calls": []}
        for text in ("alpha", "beta")
    ] + [_tool_reply("finish", mode)]
    requests = []
    client = _client(mode, replies, requests)
    tool = {"name": "finish", "description": "Finish", "input_schema": SCHEMA}
    if protocol == "openai":
        tool = {"type": "function", "function": {
            "name": "finish", "description": "Finish", "parameters": SCHEMA,
        }}
    body = {
        "model": "model", "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "original question"},
        ], "tools": [tool], "stream": caller_stream,
    }
    if protocol == "anthropic":
        body["system"] = body["messages"].pop(0)["content"]
        body["max_tokens"] = 128
    try:
        await handle_chat_completions(
            body, client, ContextManager(NoCompact(), 8192),
            client_adapter=ClientAdapter.LLAMAFILE, protocol=protocol,
            reasoning_replay="full",
        )
    finally:
        await client.aclose()

    assert len(requests) == 3
    assert all(not request.get("stream") for request in requests)
    _assert_text_retries(requests[-1]["messages"])
    assert all(m["tool_calls"] == [] for m in requests[-1]["messages"][-4::2])


@pytest.mark.parametrize("mode", ["native", "prompt"])
async def test_proxy_caller_empty_calls_do_not_hide_historical_text(mode):
    requests = []
    client = _client(mode, [{"content": "ok"}], requests)
    messages = [
        {"role": "user", "content": "original question"},
        {"role": "assistant", "content": "historical text", "tool_calls": []},
        {"role": "user", "content": "follow-up question"},
    ]
    try:
        await handle_chat_completions(
            {"model": "model", "messages": messages}, client,
            ContextManager(NoCompact(), 8192), client_adapter=ClientAdapter.LLAMAFILE,
        )
    finally:
        await client.aclose()

    assert requests[0]["messages"] == messages
