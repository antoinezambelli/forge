"""HTTP-boundary regressions for native API versions and SDK transport parity."""

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
        transport=httpx.MockTransport(handler), base_url=client._http.base_url,
        headers=client._http.headers,
    )


def _body(**extra):
    return {"model": "route", "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}], **extra}


async def _proxy(client, body, *, protocol="anthropic", backend="anthropic", headers=None):
    return await handle_chat_completions(
        body, client, ContextManager(strategy=NoCompact(), budget_tokens=None),
        protocol=protocol, backend_protocol=backend, headers=headers,
        client_adapter=ClientAdapter.ANTHROPIC if backend == "anthropic" else ClientAdapter.OPENAI_COMPAT,
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
        return httpx.Response(200, json=_reply([{
            "type": "tool_use", "id": "toolu_1", "name": "search", "input": {},
        }]))

    _http(client, handler)
    tools = [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]
    other = "2026-10-21" if version == "2023-06-01" else "2023-06-01"
    await asyncio.gather(*[
        _proxy(client, _body(model=model, tools=tools),
               headers={"anthropic-version": selected})
        for model, selected in [("route-a", version), ("route-b", other)]
    ])
    for model, selected in [("route-a", version), ("route-b", other)]:
        calls = [r for r in requests if json.loads(r.content)["model"] == model]
        assert len(calls) == 2
        assert all(r.headers["anthropic-version"] == selected for r in calls)
        assert len(json.loads(calls[1].content)["messages"]) > 1
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["anthropic", "openai"])
async def test_cross_protocol_version_policy(backend):
    client = (AnthropicClient(model=None, api_key="static") if backend == "anthropic"
              else OpenAICompatClient(base_url="https://backend/v1", model="route"))
    requests = []

    def handler(request):
        requests.append(request)
        payload = _reply() if backend == "anthropic" else {"choices": [{"message": {"content": "ok"}}]}
        return httpx.Response(200, json=payload)

    # The OpenAI client has no base_url on its HTTP session; use its endpoint.
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=client._http.headers)
    if backend == "anthropic":
        client._http.base_url = "https://api.anthropic.com/"
    await _proxy(client, _body(), backend=backend,
                 protocol="openai" if backend == "anthropic" else "anthropic",
                 headers={"anthropic-version": "2026-10-21"})
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
    requests, delays = [], []

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
            return httpx.Response(429 if failure == "rate-limit" else 503,
                                  headers={"retry-after-ms": "25"})
        return httpx.Response(200, json=_reply())

    monkeypatch.setattr("forge.clients.anthropic.asyncio.sleep", sleep)
    _http(client, handler)
    await client.send([], extra_headers={"anthropic-version": "2026-10-21"})
    assert len(requests) == 2 and len(delays) == 1
    assert requests[0].content == requests[1].content
    assert all(r.headers["anthropic-version"] == "2026-10-21" for r in requests)
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
        await client.send([])
    assert len(requests) == 1
    requests.clear()
    _http(client, lambda r: (requests.append(r) or httpx.Response(503)))
    with pytest.raises(BackendError):
        await client.send([])
    assert len(requests) == 2
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ["error", "eof"])
async def test_stream_failure_never_replays_or_produces_final(end):
    client = AnthropicClient(model="route", api_key="static", max_retries=3)
    requests, chunks = [], []

    def handler(request):
        requests.append(request)
        events = [{"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": "partial"}}]
        if end == "error":
            events.append({"type": "error", "error": {"type": "overloaded_error"}})
        return httpx.Response(200, content="".join(f"data: {json.dumps(e)}\n\n" for e in events))

    _http(client, handler)
    with pytest.raises(BackendError):
        async for chunk in client.send_stream([]):
            chunks.append(chunk)
    assert len(requests) == 1
    assert [c.content for c in chunks] == ["partial"]
    await client.aclose()
