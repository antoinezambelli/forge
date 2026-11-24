from tests.canonical import send, send_stream
from forge.clients.base import RequestMetadata

"HTTP-boundary regressions for native API versions and SDK transport parity."
import asyncio
import json
import httpx
import pytest
from forge._backend_profiles import ClientAdapter
from forge.clients.anthropic import AnthropicClient
from forge.clients.openai_compat import OpenAICompatClient
from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.errors import BackendError
from forge.proxy.handler import handle_chat_completions


def _reply(content=None):
    return {
        "content": content or [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


def _http(client, handler):
    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=client._http.base_url,
        headers=client._http.headers,
    )


def _body(**extra):
    return {
        "model": "route",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }


async def _proxy(client, body, *, protocol="anthropic", backend="anthropic", headers=None):
    return await handle_chat_completions(
        body,
        client,
        ContextManager(strategy=NoCompact(), budget_tokens=None),
        protocol=protocol,
        backend_protocol=backend,
        headers=headers,
        client_adapter=ClientAdapter.ANTHROPIC
        if backend == "anthropic"
        else ClientAdapter.OPENAI_COMPAT,
    )


@pytest.mark.asyncio
async def test_native_versions_preserve_bodies_without_leaking_between_calls():
    client = AnthropicClient(model=None, api_key="static")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=_reply())

    _http(client, handler)
    for version, extra in [
        ("2023-06-01", {"temperature": 0.4, "top_p": 0.9, "top_k": 5}),
        ("2026-10-21", {}),
        (None, {}),
        ("future-gateway-contract", {}),
    ]:
        body = _body(**extra)
        headers = {"Anthropic-Beta": "not-forwarded"}
        if version is not None:
            headers["Anthropic-Version"] = version
        await _proxy(client, body, headers=headers)
        request = requests[-1]
        assert request.headers["anthropic-version"] == (version or "2023-06-01")
        assert "anthropic-beta" not in request.headers
        assert json.loads(request.content) == body
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["2023-06-01", "2026-10-21"])
async def test_native_version_survives_guardrail_rebuild_and_concurrent_requests(version):
    client = AnthropicClient(model=None, api_key="static")
    requests = []
    attempts = {}

    async def handler(request):
        body = json.loads(request.content)
        requests.append(request)
        model = body["model"]
        attempts[model] = attempts.get(model, 0) + 1
        await asyncio.sleep(0)
        if attempts[model] == 1:
            return httpx.Response(200, json=_reply())
        return httpx.Response(
            200, json=_reply([{"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}])
        )

    _http(client, handler)
    tools = [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]
    other = "2026-10-21" if version == "2023-06-01" else "2023-06-01"
    await asyncio.gather(
        *[
            _proxy(client, _body(model=model, tools=tools), headers={"anthropic-version": selected})
            for model, selected in [("route-a", version), ("route-b", other)]
        ]
    )
    for model, selected in [("route-a", version), ("route-b", other)]:
        calls = [r for r in requests if json.loads(r.content)["model"] == model]
        assert len(calls) == 2
        assert all((r.headers["anthropic-version"] == selected for r in calls))
        assert len(json.loads(calls[1].content)["messages"]) > 1
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["anthropic", "openai"])
async def test_cross_protocol_version_policy(backend):
    client = (
        AnthropicClient(model=None, api_key="static")
        if backend == "anthropic"
        else OpenAICompatClient(base_url="https://backend/v1", model="route")
    )
    requests = []

    def handler(request):
        requests.append(request)
        payload = (
            _reply() if backend == "anthropic" else {"choices": [{"message": {"content": "ok"}}]}
        )
        return httpx.Response(200, json=payload)

    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), headers=client._http.headers
    )
    if backend == "anthropic":
        client._http.base_url = "https://api.anthropic.com/"
    await _proxy(
        client,
        _body(),
        backend=backend,
        protocol="openai" if backend == "anthropic" else "anthropic",
        headers={"anthropic-version": "2026-10-21"},
    )
    assert requests[0].headers.get("anthropic-version") == (
        "2023-06-01" if backend == "anthropic" else None
    )
    await client.aclose()


@pytest.mark.asyncio
async def test_invalid_new_contract_body_reaches_backend_unchanged():
    client = AnthropicClient(model=None, api_key="static")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(400, json={"error": {"message": "temperature unsupported"}})

    _http(client, handler)
    body = _body(temperature=0.2)
    with pytest.raises(BackendError) as error:
        await _proxy(client, body, headers={"anthropic-version": "2026-10-21"})
    assert error.value.status_code == 400
    assert len(requests) == 1
    assert json.loads(requests[0].content) == body
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "body-timeout", "rate-limit", "server-error"])
async def test_transport_retry_preserves_request_and_honors_retry_after(monkeypatch, failure):
    client = AnthropicClient(model="route", api_key="static", max_retries=1)
    requests, delays = ([], [])

    async def sleep(delay):
        delays.append(delay)

    class TimeoutBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"content":'
            raise httpx.ReadTimeout("mid-body timeout")

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("slow", request=request)
            if failure == "body-timeout":
                return httpx.Response(200, stream=TimeoutBody())
            return httpx.Response(
                429 if failure == "rate-limit" else 503, headers={"retry-after-ms": "25"}
            )
        return httpx.Response(200, json=_reply())

    monkeypatch.setattr("forge.clients.anthropic.asyncio.sleep", sleep)
    _http(client, handler)
    await send(client, [], extra_headers={"anthropic-version": "2026-10-21"})
    assert len(requests) == 2 and len(delays) == 1
    assert requests[0].content == requests[1].content
    assert all((r.headers["anthropic-version"] == "2026-10-21" for r in requests))
    if failure in ("rate-limit", "server-error"):
        assert delays == [0.025]
    await client.aclose()


@pytest.mark.asyncio
async def test_retry_budget_and_backend_directive(monkeypatch):
    client = AnthropicClient(model="route", api_key="static", max_retries=1)
    requests = []

    async def sleep(delay):
        pass

    def handler(request):
        requests.append(request)
        return httpx.Response(503, headers={"x-should-retry": "false"})

    monkeypatch.setattr("forge.clients.anthropic.asyncio.sleep", sleep)
    _http(client, handler)
    with pytest.raises(BackendError):
        await send(client, [])
    assert len(requests) == 1
    requests.clear()
    _http(client, lambda r: requests.append(r) or httpx.Response(503))
    with pytest.raises(BackendError):
        await send(client, [])
    assert len(requests) == 2
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ["error", "eof"])
async def test_stream_failure_never_replays_or_produces_final(end):
    client = AnthropicClient(model="route", api_key="static", max_retries=3)
    requests, chunks = ([], [])

    def handler(request):
        requests.append(request)
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "partial"},
            },
        ]
        if end == "error":
            events.append({"type": "error", "error": {"type": "overloaded_error"}})
        return httpx.Response(200, content="".join((f"data: {json.dumps(e)}\n\n" for e in events)))

    _http(client, handler)
    with pytest.raises(BackendError):
        async for chunk in send_stream(client, []):
            chunks.append(chunk)
    assert len(requests) == 1
    assert [c.content for c in chunks] == ["partial"]
    await client.aclose()


@pytest.mark.parametrize("inbound", ["openai", "anthropic"])
@pytest.mark.parametrize("backend", ["openai", "anthropic", "prompt"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_complete_canonical_tools_on_initial_and_retry(inbound, backend, stream):
    from copy import deepcopy
    from forge.clients.llamafile import LlamafileClient
    from forge.clients.vllm import VLLMClient

    if backend == "anthropic":
        client = AnthropicClient(model=None, api_key="test")
        adapter = ClientAdapter.ANTHROPIC
    elif backend == "openai":
        client = VLLMClient(model_path="backend-model", base_url="http://test/v1")
        adapter = ClientAdapter.VLLM
    else:
        client = LlamafileClient(gguf_path="test.gguf", base_url="http://test", mode="prompt")
        adapter = ClientAdapter.LLAMAFILE
    schema = {
        "type": "object",
        "$defs": {"Item": {"oneOf": [{"type": "string"}, {"type": "number"}]}},
        "properties": {"value": {"$ref": "#/$defs/Item"}, "any": True, "never": False},
        "additionalProperties": False,
        "x-unknown": {"items": [1, 2]},
    }
    extras = {"strict": True, "cache_control": {"type": "ephemeral"}, "x-extension": [1, {"a": 2}]}
    if inbound == "openai":
        tools = [
            {
                "type": "function",
                "outer-ignored": "not-a-function-extra",
                "function": {
                    "name": name,
                    "description": "Complete",
                    "parameters": deepcopy(schema),
                    **extras,
                },
            }
            for name in ("lookup", "search")
        ]
    else:
        tools = [
            {"name": name, "description": "Complete", "input_schema": deepcopy(schema), **extras}
            for name in ("lookup", "search")
        ]
    body = _body(tools=tools, stream=stream)
    before = deepcopy(body)
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        first = len(sent) == 1
        if backend == "anthropic":
            content = (
                [{"type": "text", "text": "try again"}]
                if first
                else [
                    {"type": "tool_use", "id": "r", "name": "respond", "input": {"message": "done"}}
                ]
            )
            if not stream:
                return httpx.Response(200, json=_reply(content))
            if first:
                events = [
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "try again"},
                    },
                ]
            else:
                events = [
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "tool_use", "id": "r", "name": "respond"},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "input_json_delta", "partial_json": '{"message":"done"}'},
                    },
                ]
            events.append({"type": "message_stop"})
        else:
            message = {"role": "assistant", "content": "try again"}
            if not first:
                if backend == "prompt":
                    message["content"] = '{"tool":"respond","args":{"message":"done"}}'
                else:
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "r",
                                "type": "function",
                                "function": {"name": "respond", "arguments": '{"message":"done"}'},
                            }
                        ],
                    }
            if not stream:
                return httpx.Response(200, json={"choices": [{"message": message}]})
            delta = {key: value for key, value in message.items() if key != "role"}
            if "tool_calls" in delta:
                delta["tool_calls"][0]["index"] = 0
            events = [{"choices": [{"delta": delta}]}]
        data = "".join(("data: " + json.dumps(event) + "\n\n" for event in events))
        if backend != "anthropic":
            data += "data: [DONE]\n\n"
        return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})

    _http(client, handler)
    try:
        if stream:
            from forge.core.inference import run_inference
            from forge.proxy.handler import _extract_tool_definitions
            from forge.proxy.convert import openai_to_messages
            from forge.proxy.convert_anthropic import (
                anthropic_tools_to_definitions,
                anthropic_to_messages,
            )
            from forge.guardrails import ResponseValidator, ErrorTracker
            from forge.tools.respond import respond_definition

            definitions = (
                _extract_tool_definitions(tools)
                if inbound == "openai"
                else anthropic_tools_to_definitions(tools)
            )
            definitions.append(respond_definition())
            messages = (
                openai_to_messages(body["messages"])
                if inbound == "openai"
                else anthropic_to_messages(body["messages"])
            )
            seen = []
            original_stream = client.send_stream

            async def recording_stream(*args, **kwargs):
                seen.append(args[0].payload.get("tools"))
                async for chunk in original_stream(*args, **kwargs):
                    yield chunk

            client.send_stream = recording_stream
            result = await run_inference(
                state=messages,
                client=client,
                context_manager=ContextManager(strategy=NoCompact(), budget_tokens=None),
                validator=ResponseValidator([tool.name for tool in definitions]),
                error_tracker=ErrorTracker(max_retries=1),
                tool_definitions=definitions,
                metadata=RequestMetadata(model="route", source_protocol=inbound),
                stream=True,
            )
            assert len(seen) == 2
        else:
            result = await handle_chat_completions(
                body,
                client,
                ContextManager(strategy=NoCompact(), budget_tokens=None),
                protocol=inbound,
                backend_protocol="anthropic" if backend == "anthropic" else "openai",
                client_adapter=adapter,
                max_retries=1,
                inject_respond_tool=True,
            )
    finally:
        await client.aclose()
    assert result is not None
    assert len(sent) == 2
    if backend == "prompt":
        prompts = [request["messages"][0]["content"] for request in sent]
        for prompt in prompts:
            assert "Input schema: " + json.dumps(schema) in prompt
            assert '"x-extension": [1, {"a": 2}]' in prompt
            assert "outer-ignored" not in prompt
            assert (
                prompt.index("## lookup") < prompt.index("## search") < prompt.index("## respond")
            )
            assert prompt.count("## respond") == 1
    else:
        assert sent[0]["tools"] == sent[1]["tools"]
        definitions = sent[0]["tools"]
        if backend == "openai":
            definitions = [tool["function"] for tool in definitions]
        assert [tool["name"] for tool in definitions] == ["lookup", "search", "respond"]
        key = "parameters" if backend == "openai" else "input_schema"
        for definition in definitions[:2]:
            assert definition[key] == schema
            for name, value in extras.items():
                assert definition[name] == value
            assert "outer-ignored" not in definition
    assert body == before


@pytest.mark.parametrize(
    "inject, names", [(True, None), (True, []), (False, ["lookup"]), (True, ["lookup", "respond"])]
)
@pytest.mark.asyncio
async def test_native_anthropic_copied_body_respond_policy(inject, names):
    from copy import deepcopy

    client = AnthropicClient(model=None, api_key="test")
    tools = (
        None
        if names is None
        else [
            {"name": name, "description": "Caller definition", "input_schema": {}, "x-tool": name}
            for name in names
        ]
    )
    body = _body(**{"tools": tools} if tools is not None else {})
    before = deepcopy(body)
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        if not names:
            return httpx.Response(200, json=_reply())
        name = names[-1]
        return httpx.Response(
            200,
            json=_reply(
                [
                    {
                        "type": "tool_use",
                        "id": "call",
                        "name": name,
                        "input": {"message": "done"} if name == "respond" else {},
                    }
                ]
            ),
        )

    _http(client, handler)
    try:
        await handle_chat_completions(
            body,
            client,
            ContextManager(strategy=NoCompact(), budget_tokens=None),
            client_adapter=ClientAdapter.ANTHROPIC,
            protocol="anthropic",
            backend_protocol="anthropic",
            inject_respond_tool=inject,
        )
    finally:
        await client.aclose()
    if tools is None:
        assert "tools" not in sent[0]
    else:
        assert sent[0]["tools"] == tools
    assert body == before


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
async def test_native_gateway_extras_and_known_translation_survive_actual_retry(protocol):
    client = AnthropicClient(model="pinned", api_key="static")
    sent = []

    def handler(request):
        sent.append((json.loads(request.content), dict(request.headers)))
        content = (
            [{"type": "text", "text": "retry"}]
            if len(sent) == 1
            else [{"type": "tool_use", "name": "search", "input": {"q": "done"}}]
        )
        return httpx.Response(200, json=_reply(content))

    _http(client, handler)
    tools = (
        [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        if protocol == "openai"
        else [{"name": "search", "input_schema": {}}]
    )
    body = _body(
        tools=tools,
        metadata={"trace_id": "retained", "nested": {"anything": [1, False]}},
        vendor_extension={"route": "gateway"},
    )
    body.update(
        {"stop": ["END"], "tool_choice": "required"}
        if protocol == "openai"
        else {
            "stop_sequences": ["END"],
            "tool_choice": {"type": "any"},
            "thinking": {"type": "adaptive"},
            "service_tier": "auto",
        }
    )
    body["messages"][0] = {
        "role": "user",
        "name": "caller",
        "content": [
            {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "b", "x-part": 2},
        ],
    }
    result = await _proxy(
        client, body, protocol=protocol, headers={"anthropic-version": "gateway-version"}
    )
    assert len(sent) == 2
    for rendered, headers in sent:
        assert rendered["model"] == "pinned"
        assert rendered["metadata"] == body["metadata"]
        assert rendered["vendor_extension"] == body["vendor_extension"]
        assert rendered["stop_sequences"] == ["END"] and "stop" not in rendered
        assert rendered["tool_choice"] == {"type": "any"}
        assert rendered["messages"][0] == body["messages"][0]
        if protocol == "anthropic":
            assert rendered["thinking"] == {"type": "adaptive"}
            assert rendered["service_tier"] == "auto"
            assert headers["anthropic-version"] == "gateway-version"
    assert result is not None
    await client.aclose()
