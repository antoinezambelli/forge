"""Tests for proxy request handler."""

from dataclasses import replace
import json
from copy import deepcopy

import pytest
from unittest.mock import AsyncMock, MagicMock

from forge._backend_profiles import ClientAdapter, ModelCatalog, ModelCatalogEntry
from forge.clients.llamafile import LlamafileClient
from forge.clients.ollama import OllamaClient
from forge.clients.vllm import VLLMClient
from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.workflow import TextResponse, ToolCall
from forge.clients.base import TokenUsage
from forge.errors import BackendDiscoveryError, BackendError, MissingModelError
from forge.proxy.handler import (
    LazyDiscovery,
    RequestFacts,
    handle_chat_completions,
    observe_request_context,
    _extract_tool_specs,
)


pytestmark = pytest.mark.usefixtures("mock_httpx_client_constructor")


# ── Helpers ──────────────────────────────────────────────────


def _mock_client(response):
    """Create a mock LLMClient that returns the given response."""
    client = AsyncMock()
    client.api_format = "ollama"
    client.model = "backend-model"
    client.last_usage = {}
    client._slot_id = 0

    async def send(*args, **kwargs):
        usage = client.last_usage.get(0)
        if usage is not None:
            # Built-in clients publish a new immutable object for every
            # response that carries usage.
            client.last_usage[0] = replace(usage)
        return response

    client.send = AsyncMock(side_effect=send)
    return client


def _context_manager():
    return ContextManager(strategy=NoCompact(), budget_tokens=8192)


def _body(messages=None, tools=None, stream=False, model="test"):
    """Build a minimal request body."""
    b = {"messages": messages or [{"role": "user", "content": "hi"}], "model": model}
    if tools is not None:
        b["tools"] = tools
    if stream:
        b["stream"] = True
    return b


def _tool_def(name="search", description="Search", parameters=None):
    """Build an OpenAI-format tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }


@pytest.mark.parametrize(
    ("body_value", "header_value", "expected_id", "expected_source"),
    [
        ("lite", "claude", "claude", "claude_code"),
        ("lite", "   ", "lite", "litellm"),
        (" opaque ", None, " opaque ", "litellm"),
        (None, None, None, None),
        (123, None, None, None),
    ],
)
def test_request_session_precedence_and_opacity(
    body_value, header_value, expected_id, expected_source,
):
    body = {"litellm_session_id": body_value}
    headers = (
        {"x-claude-code-session-id": header_value}
        if header_value is not None else {}
    )
    facts = RequestFacts()

    observe_request_context(body, headers, facts)

    if expected_id is None:
        assert facts.session is None
    else:
        assert facts.session.id == expected_id
        assert facts.session.source == expected_source


@pytest.mark.parametrize(
    "header_name",
    ["x-claude-code-agent-id", "x-claude-code-parent-agent-id"],
)
def test_nonempty_claude_agent_headers_are_ineligible(header_name):
    facts = RequestFacts()
    observe_request_context({}, {header_name: "agent"}, facts)
    assert facts.reporting_eligible is False

    observe_request_context({}, {header_name: "  "}, facts)
    assert facts.reporting_eligible is True


# ── _extract_tool_specs ──────────────────────────────────────


class TestExtractToolSpecs:
    def test_absent_tools_return_empty(self):
        for tools in (None, []):
            assert _extract_tool_specs(tools) == [], tools

    def test_extracts_function_tools(self):
        specs = _extract_tool_specs([_tool_def("search"), _tool_def("fetch")])
        assert len(specs) == 2
        assert specs[0].name == "search"
        assert specs[1].name == "fetch"

    def test_skips_non_function_types(self):
        tools = [{"type": "retrieval"}, _tool_def("search")]
        specs = _extract_tool_specs(tools)
        assert len(specs) == 1
        assert specs[0].name == "search"


# ── No tools → passthrough ──────────────────────────────────


class TestNoToolsPassthrough:
    @pytest.mark.asyncio
    async def test_text_response_passthrough(self):
        client = _mock_client(TextResponse(content="Hello!"))
        result = await handle_chat_completions(
            _body(), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
        )
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_text_response_passthrough_stream(self):
        client = _mock_client(TextResponse(content="Hello!"))
        result = await handle_chat_completions(
            _body(stream=True), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
        )
        # SSE events list
        assert isinstance(result, list)
        assert result[-1]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_model_name_propagated(self):
        client = _mock_client(TextResponse(content="hi"))
        result = await handle_chat_completions(
            _body(model="my-model"), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
        )
        assert result["model"] == "my-model"


class TestEffectiveModelContracts:
    @staticmethod
    def _openai_response(content="ok", tool_name=None):
        message = {"role": "assistant", "content": content, "tool_calls": []}
        finish_reason = "stop"
        if tool_name is not None:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": '{"q":"done"}',
                    },
                }],
            }
            finish_reason = "tool_calls"
        response = MagicMock()
        response.status_code = 200
        response.text = ""
        response.json.return_value = {
            "choices": [{"message": message, "finish_reason": finish_reason}],
        }
        return response

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        ("body_model", "effective_model"),
        [(None, "configured"), ("opaque.gateway/route", "opaque.gateway/route")],
    )
    async def test_generic_response_matches_downstream_model(
        self, body_model, effective_model, stream,
    ):
        client = LlamafileClient(
            gguf_path="configured.gguf",
            base_url="http://test:8080/v1",
            mode="native",
        )
        client._http = AsyncMock()
        client._http.post.return_value = self._openai_response()
        body = {"messages": [{"role": "user", "content": "hi"}]}
        if body_model is not None:
            body["model"] = body_model
        if stream:
            body["stream"] = True
        facts = RequestFacts()

        result = await handle_chat_completions(
            body,
            client,
            _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            request_facts=facts,
        )

        downstream = client._http.post.call_args.kwargs["json"]
        assert downstream["model"] == effective_model
        if stream:
            assert all(event["model"] == downstream["model"] for event in result)
        else:
            assert result["model"] == downstream["model"]
        assert facts.effective_model == downstream["model"]

    @pytest.mark.asyncio
    async def test_vllm_discovery_identity_and_buffered_stream_match_wire(self):
        client = VLLMClient(
            model_path="default",
            base_url="http://test:8000/v1",
        )
        client._http = AsyncMock()
        client._http.post.return_value = self._openai_response()
        nested = {"route": "gold"}
        body = {
            "model": "caller-alias",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "max_tokens": 144,
            "provider_extension": nested,
        }
        facts = RequestFacts()

        result = await handle_chat_completions(
            body,
            client,
            _context_manager(),
            client_adapter=ClientAdapter.VLLM,
            lazy_discovery=LazyDiscovery(),
            request_facts=facts,
            catalog_fetcher=AsyncMock(return_value=ModelCatalog(
                (ModelCatalogEntry("served-nemotron-120b", 64000),),
                first_served_id="served-nemotron-120b",
            )),
        )

        downstream = client._http.post.call_args.kwargs["json"]
        assert downstream["model"] == "served-nemotron-120b"
        assert downstream["stream"] is False
        assert "stream_options" not in downstream
        assert downstream["max_tokens"] == 144
        assert downstream["provider_extension"] is nested
        assert all(event["model"] == downstream["model"] for event in result)
        assert facts.effective_model == downstream["model"]
        assert facts.model_catalog is not None

    @pytest.mark.asyncio
    async def test_vllm_empty_tools_array_remains_authoritative(self):
        client = VLLMClient(
            model_path="wire-pin",
            base_url="http://test:8000/v1",
        )
        client._http = AsyncMock()
        client._http.post.return_value = self._openai_response()

        await handle_chat_completions(
            {
                "model": "caller-alias",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [],
            },
            client,
            _context_manager(),
            client_adapter=ClientAdapter.VLLM,
        )

        downstream = client._http.post.call_args.kwargs["json"]
        assert downstream["tools"] == []
        assert "tool_choice" not in downstream

    @pytest.mark.asyncio
    async def test_vllm_retry_keeps_passthrough_but_rebuilds_raw_artifacts(self):
        client = VLLMClient(
            model_path="wire-pin",
            base_url="http://test:8000/v1",
        )
        client._http = AsyncMock()
        client._http.post.side_effect = [
            self._openai_response(content="not a tool"),
            self._openai_response(content=None, tool_name="search"),
        ]
        raw_tools = [_tool_def("search", parameters={
            "type": "object",
            "properties": {"q": {"type": "string", "x-provider": "raw"}},
            "required": ["q"],
        })]
        body = {
            "model": "caller-alias",
            "messages": [{"role": "user", "content": "hi", "vendor": "raw"}],
            "tools": raw_tools,
            "tool_choice": "required",
            "max_tokens": 333,
            "stop": ["END"],
            "provider_extension": {"route": "retry-stable"},
        }
        body_before = deepcopy(body)

        result = await handle_chat_completions(
            body,
            client,
            _context_manager(),
            max_retries=1,
            client_adapter=ClientAdapter.VLLM,
        )

        first, retry = [call.kwargs["json"] for call in client._http.post.call_args_list]
        assert first["tools"] == raw_tools
        assert first["messages"][0]["vendor"] == "raw"
        assert retry["tools"] != raw_tools
        assert "x-provider" not in retry["tools"][0]["function"]["parameters"][
            "properties"
        ]["q"]
        assert retry["messages"][0]["content"] == "hi"
        assert len(retry["messages"]) > 1
        for request in (first, retry):
            assert request["model"] == "wire-pin"
            assert request["stream"] is False
            assert request["tool_choice"] == "required"
            assert request["max_tokens"] == 333
            assert request["stop"] == ["END"]
            assert request["provider_extension"] == {"route": "retry-stable"}
        assert result["model"] == "wire-pin"
        assert body == body_before

    @pytest.mark.asyncio
    async def test_vllm_synthetic_respond_does_not_mutate_or_escape(self):
        client = VLLMClient(
            model_path="wire-pin",
            base_url="http://test:8000/v1",
        )
        client._http = AsyncMock()
        response = self._openai_response(content=None, tool_name="respond")
        response.json.return_value["choices"][0]["message"]["tool_calls"][0][
            "function"
        ]["arguments"] = '{"message":"finished"}'
        client._http.post.return_value = response
        tools = [_tool_def("search")]
        body = _body(tools=tools)
        original_tools = deepcopy(tools)

        result = await handle_chat_completions(
            body,
            client,
            _context_manager(),
            inject_respond_tool=True,
            client_adapter=ClientAdapter.VLLM,
        )

        downstream_tools = client._http.post.call_args.kwargs["json"]["tools"]
        assert [tool["function"]["name"] for tool in downstream_tools] == [
            "search",
            "respond",
        ]
        assert tools == original_tools
        assert result["choices"][0]["message"]["content"] == "finished"
        assert "tool_calls" not in result["choices"][0]["message"]


# ── Deferred external-mode discovery ─────────────────────────


class TestLazyDiscovery:
    @staticmethod
    def _catalog(window: int | None = 50000) -> ModelCatalog:
        return ModelCatalog(
            (ModelCatalogEntry("served", window),),
            first_served_id="served",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("budget", [None, 8192])
    async def test_first_request_adopts_identity_without_changing_budget(
        self, budget: int | None,
    ) -> None:
        client = _mock_client(TextResponse(content="hello"))
        client.model = "default"
        client._set_model_identity = MagicMock(
            side_effect=lambda model: setattr(client, "model", model),
        )
        ctx = ContextManager(NoCompact(), budget_tokens=budget)
        lazy = LazyDiscovery()
        facts = RequestFacts()
        fetch = AsyncMock(return_value=self._catalog())

        await handle_chat_completions(
            {"messages": [], "stream": False},
            client,
            ctx,
            client_adapter=ClientAdapter.VLLM,
            lazy_discovery=lazy,
            request_facts=facts,
            catalog_fetcher=fetch,
        )

        fetch.assert_awaited_once_with({})
        client._set_model_identity.assert_called_once_with("served")
        assert client.model == "served"
        assert lazy.done is True
        assert ctx.budget_tokens == budget
        assert facts.model_catalog == self._catalog()

    @pytest.mark.asyncio
    async def test_missing_window_does_not_block_identity(self) -> None:
        client = _mock_client(TextResponse(content="hello"))
        client._set_model_identity = MagicMock()
        lazy = LazyDiscovery()
        facts = RequestFacts()
        await handle_chat_completions(
            {"messages": [], "stream": False},
            client,
            ContextManager(NoCompact(), budget_tokens=None),
            client_adapter=ClientAdapter.VLLM,
            lazy_discovery=lazy,
            request_facts=facts,
            catalog_fetcher=AsyncMock(return_value=self._catalog(None)),
        )
        client._set_model_identity.assert_called_once_with("served")
        assert lazy.done is True
        assert facts.model_catalog.context_length_for("served") is None

    @pytest.mark.asyncio
    async def test_failure_does_not_latch_and_next_request_retries(self) -> None:
        client = _mock_client(TextResponse(content="hello"))
        client._set_model_identity = MagicMock()
        lazy = LazyDiscovery()
        fetch = AsyncMock(side_effect=[
            BackendError(401),
            self._catalog(),
        ])
        kwargs = {
            "body": {"messages": [], "stream": False},
            "client": client,
            "context_manager": ContextManager(NoCompact(), budget_tokens=None),
            "client_adapter": ClientAdapter.VLLM,
            "lazy_discovery": lazy,
            "catalog_fetcher": fetch,
        }

        with pytest.raises(BackendDiscoveryError) as exc_info:
            await handle_chat_completions(**kwargs)
        assert exc_info.value.status_code == 401
        assert lazy.done is False
        client._set_model_identity.assert_not_called()

        await handle_chat_completions(**kwargs)
        assert lazy.done is True
        assert fetch.await_count == 2
        client._set_model_identity.assert_called_once_with("served")

    @pytest.mark.asyncio
    async def test_latched_request_skips_catalog_query(self) -> None:
        client = _mock_client(TextResponse(content="hello"))
        fetch = AsyncMock()
        await handle_chat_completions(
            {"messages": [], "stream": False},
            client,
            _context_manager(),
            client_adapter=ClientAdapter.VLLM,
            lazy_discovery=LazyDiscovery(done=True),
            catalog_fetcher=fetch,
        )
        fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_original_lowercase_headers_reach_private_fetcher(self) -> None:
        client = _mock_client(TextResponse(content="hello"))
        client._set_model_identity = MagicMock()
        fetch = AsyncMock(return_value=self._catalog())
        headers = {"x-api-key": "inbound-token"}
        await handle_chat_completions(
            {"messages": [], "stream": False},
            client,
            _context_manager(),
            client_adapter=ClientAdapter.VLLM,
            headers=headers,
            lazy_discovery=LazyDiscovery(),
            catalog_fetcher=fetch,
        )
        fetch.assert_awaited_once_with(headers)

    @pytest.mark.asyncio
    async def test_concurrent_first_requests_are_duplicate_safe(self) -> None:
        client = _mock_client(TextResponse(content="hello"))
        client._set_model_identity = MagicMock()
        lazy = LazyDiscovery()
        arrived = 0
        release = __import__("asyncio").Event()

        async def fetch(_: dict[str, str]) -> ModelCatalog:
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                release.set()
            await release.wait()
            return self._catalog(70000)

        await __import__("asyncio").gather(*[
            handle_chat_completions(
                {"messages": [], "stream": False},
                client,
                ContextManager(NoCompact(), budget_tokens=None),
                client_adapter=ClientAdapter.VLLM,
                lazy_discovery=lazy,
                catalog_fetcher=fetch,
            )
            for _ in range(2)
        ])
        assert arrived == 2
        assert client._set_model_identity.call_count == 2
        assert lazy.done is True


class TestRequestLocalUsage:
    @staticmethod
    def _client_with_stale_usage(response):
        client = _mock_client(response)
        stale = TokenUsage(prompt_tokens=91, completion_tokens=9, total_tokens=100)
        client.last_usage = {0: stale}
        client.send = AsyncMock(return_value=response)
        return client, stale

    @pytest.mark.asyncio
    async def test_no_tools_does_not_fall_back_to_stale_client_usage(self):
        client, stale = self._client_with_stale_usage(TextResponse(content="ok"))
        facts = RequestFacts()

        result = await handle_chat_completions(
            _body(), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            request_facts=facts,
        )

        assert result["usage"] == {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }
        assert facts.usage is None
        assert client.last_usage[0] is stale

    @pytest.mark.asyncio
    async def test_retries_exhausted_does_not_fall_back_to_stale_client_usage(self):
        client, stale = self._client_with_stale_usage(TextResponse(content="nope"))
        facts = RequestFacts()

        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            max_retries=0,
            request_facts=facts,
        )

        assert result["usage"] == {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }
        assert facts.usage is None
        assert client.last_usage[0] is stale

    @pytest.mark.asyncio
    async def test_tool_response_does_not_fall_back_to_stale_client_usage(self):
        response = [ToolCall(tool="search", args={"q": "test"})]
        client, stale = self._client_with_stale_usage(response)
        facts = RequestFacts()

        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            request_facts=facts,
        )

        assert result["usage"] == {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }
        assert facts.usage is None
        assert client.last_usage[0] is stale


class TestWithTools:
    @pytest.mark.asyncio
    async def test_tool_call_returned(self):
        """Valid tool call is returned in OpenAI format."""
        client = _mock_client([ToolCall(tool="search", args={"q": "test"})])
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}
        
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
        )
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search"
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_tool_call_stream(self):
        """Valid tool call returns SSE events."""
        client = _mock_client([ToolCall(tool="search", args={})])
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}
        
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")], stream=True),
            client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
        )
        assert isinstance(result, list)
        assert result[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert result[-1]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_respond_tool_auto_injected(self):
        """With inject_respond_tool=True, a respond() call is stripped to text."""
        client = _mock_client([ToolCall(tool="respond", args={"message": "Hi!"})])
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}

        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            inject_respond_tool=True,
        )
        # respond is stripped — client sees text, not a tool call
        assert result["choices"][0]["message"]["content"] == "Hi!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert "tool_calls" not in result["choices"][0]["message"]
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_respond_stripped_in_stream(self):
        """Respond call in stream mode returns text SSE events."""
        client = _mock_client([ToolCall(tool="respond", args={"message": "Hi!"})])
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}
        
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")], stream=True),
            client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
        )
        assert isinstance(result, list)
        assert result[-1]["choices"][0]["finish_reason"] == "stop"
        assert result[-1]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_mixed_respond_and_tool_calls(self):
        """If respond is mixed with real tool calls, respond is dropped."""
        client = _mock_client([
            ToolCall(tool="search", args={"q": "test"}),
            ToolCall(tool="respond", args={"message": "also this"}),
        ])
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}

        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            inject_respond_tool=True,
        )
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search"
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_respond_not_double_injected(self):
        """If client already provides respond tool, don't inject again."""
        client = _mock_client([ToolCall(tool="respond", args={"message": "Hi!"})])
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}
        
        tools = [_tool_def("search"), _tool_def("respond")]
        result = await handle_chat_completions(
            _body(tools=tools), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            inject_respond_tool=True,
        )
        # Should still work — respond stripped to text (not double-injected)
        assert result["choices"][0]["message"]["content"] == "Hi!"
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


# ── Error paths ─────────────────────────────────────────────


class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_retries_exhausted_returns_text(self):
        """When retries are exhausted, last text is returned to client."""
        # Model always returns text — will exhaust retries
        client = _mock_client(TextResponse(content="I can't do that"))
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]),
            client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE, max_retries=1,
        )
        # Should return the text rather than an error
        assert result["choices"][0]["message"]["content"] == "I can't do that"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_retries_exhausted_stream(self):
        """Retries exhausted in stream mode returns text SSE events."""
        client = _mock_client(TextResponse(content="nope"))
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")], stream=True),
            client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE, max_retries=1,
        )
        assert isinstance(result, list)
        # Should contain the text in SSE events
        content_events = [
            e for e in result
            if e["choices"][0].get("delta", {}).get("content")
        ]
        assert len(content_events) > 0
        assert result[-1]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_malformed_args_bounded_by_max_tool_errors(self):
        """Malformed tool args drain the tool-error budget, not retries — so
        max_tool_errors (not max_retries) bounds the loop. Proves the proxy's
        new max_tool_errors knob threads through to the ErrorTracker."""
        # Known tool, non-dict args → tool_arg_validation every turn.
        client = _mock_client([ToolCall(tool="search", args="bad")])  # type: ignore[arg-type]
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]),
            client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            max_retries=5, max_tool_errors=1,
        )
        # Exhausted on the tool-error budget (1), not max_retries (5):
        # send #1 (error, budget→1) + send #2 (error, 2 > 1 → exhausted).
        assert client.send.call_count == 2
        assert result["choices"][0]["finish_reason"] == "stop"


class TestSamplingPlumbing:
    """Issue A: inbound body sampling fields plumbed through to client.send."""

    @pytest.mark.asyncio
    async def test_no_tools_path_passes_sampling(self):
        """Inbound body sampling fields reach client.send on the no-tools path."""
        client = _mock_client(TextResponse(content="ok"))
        client.last_usage = {0: TokenUsage(1, 1, 2)}
        body = _body(messages=[{"role": "user", "content": "hi"}])
        body["temperature"] = 0.5
        body["top_p"] = 0.9

        result = await handle_chat_completions(
            body, client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE, max_retries=1,
        )

        client.send.assert_called_once()
        sampling = client.send.call_args.kwargs["sampling"]
        assert sampling == {"temperature": 0.5, "top_p": 0.9}
        assert result["usage"] == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    @pytest.mark.asyncio
    async def test_no_tools_path_no_sampling_fields(self):
        """No sampling fields in body → sampling=None."""
        client = _mock_client(TextResponse(content="ok"))

        await handle_chat_completions(
            _body(), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE, max_retries=1,
        )

        sampling = client.send.call_args.kwargs["sampling"]
        assert sampling is None

    @pytest.mark.asyncio
    async def test_tools_path_passes_sampling_to_run_inference(self, monkeypatch):
        """With tools, sampling reaches run_inference (and through it the client)."""
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        captured: dict = {}

        async def fake_run_inference(**kwargs):
            captured["sampling"] = kwargs.get("sampling")
            from forge.core.inference import InferenceResult
            return InferenceResult(
                response=[ToolCall(tool="search", args={"q": "x"})],
                new_messages=[],
                usage=TokenUsage(10, 5, 15),
                tool_call_counter=0,
                attempts=1,
            )

        monkeypatch.setattr(
            "forge.proxy.handler.run_inference", fake_run_inference,
        )

        body = _body(tools=[_tool_def("search")])
        body["seed"] = 42
        body["temperature"] = 0.3

        result = await handle_chat_completions(
            body, client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE, max_retries=1,
        )

        assert captured["sampling"] == {"temperature": 0.3, "seed": 42}
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_per_call_sampling_does_not_mutate_client(self):
        """Per-call sampling overrides do not leak into subsequent calls."""
        client = _mock_client(TextResponse(content="ok"))

        # First request: with temperature override.
        body1 = _body()
        body1["temperature"] = 0.99
        await handle_chat_completions(
            body1, client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE, max_retries=1,
        )
        first_sampling = client.send.call_args.kwargs["sampling"]
        assert first_sampling == {"temperature": 0.99}

        # Second request: no sampling fields.
        await handle_chat_completions(
            _body(), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE, max_retries=1,
        )
        second_sampling = client.send.call_args.kwargs["sampling"]
        assert second_sampling is None

    @pytest.mark.asyncio
    async def test_passthrough_carries_unknown_body_fields(self):
        """Inbound body fields outside sampling/forge-owned flow through passthrough."""
        client = _mock_client(TextResponse(content="ok"))
        body = _body(messages=[{"role": "user", "content": "hi"}])
        body["max_tokens"] = 256
        body["tool_choice"] = "auto"

        await handle_chat_completions(
            body, client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE, max_retries=1,
        )

        passthrough = client.send.call_args.kwargs["passthrough"]
        assert passthrough == {
            "model": "test",
            "max_tokens": 256,
            "tool_choice": "auto",
        }


    @pytest.mark.asyncio
    async def test_stream_options_excluded_from_passthrough(self):
        """stream_options must not leak into passthrough.

        Forge controls streaming independently — when it makes non-streaming
        calls to the backend, a leaked stream_options causes validation
        errors on strict backends (e.g. vLLM rejects stream_options when
        stream is not True).
        """
        client = _mock_client(TextResponse(content="ok"))
        body = _body(messages=[{"role": "user", "content": "hi"}])
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        body["max_tokens"] = 256

        await handle_chat_completions(
            body, client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE, max_retries=1,
        )

        passthrough = client.send.call_args.kwargs["passthrough"]
        assert "stream_options" not in passthrough
        assert passthrough == {"model": "test", "max_tokens": 256}

# ── Anthropic protocol routing ───────────────────────────────


class TestAnthropicProtocol:
    """End-to-end handler tests for the /v1/messages (protocol="anthropic") path."""

    def _anthropic_body(self, messages=None, tools=None, system=None, **extra):
        body = {
            "model": "claude-3-5-sonnet",
            "messages": messages or [{"role": "user", "content": "hi"}],
            "max_tokens": 256,
        }
        if tools is not None:
            body["tools"] = tools
        if system is not None:
            body["system"] = system
        body.update(extra)
        return body

    @pytest.mark.asyncio
    async def test_no_tools_returns_anthropic_shape(self):
        client = _mock_client(TextResponse(content="hello"))
        body = self._anthropic_body()
        result = await handle_chat_completions(
            body, client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            max_retries=1, protocol="anthropic",
        )
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        assert result["content"] == [{"type": "text", "text": "hello"}]

    @pytest.mark.asyncio
    async def test_tool_call_returns_anthropic_shape(self, monkeypatch):
        from forge.core.inference import InferenceResult

        async def fake_run_inference(**kwargs):
            return InferenceResult(
                response=[ToolCall(tool="get_weather", args={"city": "Paris"})],
                new_messages=[],
                tool_call_counter=0,
                attempts=1,
            )
        monkeypatch.setattr("forge.proxy.handler.run_inference", fake_run_inference)

        client = _mock_client([ToolCall(tool="get_weather", args={"city": "Paris"})])
        body = self._anthropic_body(
            tools=[{
                "name": "get_weather",
                "description": "Weather.",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
        )
        result = await handle_chat_completions(
            body, client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            max_retries=1, protocol="anthropic",
        )
        assert result["type"] == "message"
        assert result["stop_reason"] == "tool_use"
        tu_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tu_blocks) == 1
        assert tu_blocks[0]["name"] == "get_weather"
        assert tu_blocks[0]["input"] == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_streaming_returns_anthropic_event_sequence(self):
        client = _mock_client(TextResponse(content="streamed"))
        body = self._anthropic_body(stream=True)
        events = await handle_chat_completions(
            body, client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            max_retries=1, protocol="anthropic",
        )
        assert isinstance(events, list)
        types = [e["type"] for e in events]
        assert types[0] == "message_start"
        assert types[-1] == "message_stop"

    @pytest.mark.asyncio
    async def test_anthropic_passthrough_translates_to_openai_shape(self):
        """tool_choice and stop_sequences land in passthrough in OpenAI shape."""
        client = _mock_client(TextResponse(content="ok"))
        body = self._anthropic_body(
            stop_sequences=["</done>"],
            tool_choice={"type": "any"},
        )
        await handle_chat_completions(
            body, client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            max_retries=1, protocol="anthropic",
        )
        passthrough = client.send.call_args.kwargs["passthrough"]
        assert passthrough["stop"] == ["</done>"]
        assert passthrough["tool_choice"] == "required"
        assert passthrough["model"] == "claude-3-5-sonnet"
        assert passthrough["max_tokens"] == 256
        # Anthropic-only fields with no OpenAI analog don't appear.
        assert "thinking" not in passthrough
        assert "metadata" not in passthrough

    @pytest.mark.asyncio
    async def test_system_top_level_flows_into_messages(self):
        """Anthropic puts system at top level; forge prepends it as a SYSTEM message."""
        client = _mock_client(TextResponse(content="ok"))
        body = self._anthropic_body(system="You are helpful.")
        await handle_chat_completions(
            body, client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            max_retries=1, protocol="anthropic",
        )
        api_messages = client.send.call_args.args[0]
        assert api_messages[0]["role"] == "system"
        assert api_messages[0]["content"] == "You are helpful."


class TestAnthropicBackendModelRouting:
    @pytest.mark.asyncio
    async def test_unpinned_openai_request_uses_opaque_inbound_model(self):
        client = _mock_client(TextResponse(content="ok"))
        client.model = None
        result = await handle_chat_completions(
            _body(model="nemtoron-120b"), client, _context_manager(),
            client_adapter=ClientAdapter.ANTHROPIC,
            backend_protocol="anthropic",
        )
        assert client.send.call_args.kwargs["passthrough"]["model"] == "nemtoron-120b"
        assert result["model"] == "nemtoron-120b"
        assert client.model is None

    @pytest.mark.asyncio
    async def test_literal_claude_is_preserved_when_supplied(self):
        client = _mock_client(TextResponse(content="ok"))
        client.model = None
        result = await handle_chat_completions(
            _body(model="claude"), client, _context_manager(),
            client_adapter=ClientAdapter.ANTHROPIC,
            backend_protocol="anthropic",
        )
        assert client.send.call_args.kwargs["passthrough"]["model"] == "claude"
        assert result["model"] == "claude"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("protocol", ["openai", "anthropic"])
    async def test_configured_pin_overrides_both_inbound_wire_shapes(self, protocol):
        client = _mock_client(TextResponse(content="ok"))
        client.model = "gateway-pin"
        body = (
            _body(model="caller-model")
            if protocol == "openai"
            else {
                "model": "caller-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 32,
            }
        )
        result = await handle_chat_completions(
            body, client, _context_manager(),
            client_adapter=ClientAdapter.ANTHROPIC, protocol=protocol,
            backend_protocol="anthropic",
        )
        sent = client.send.call_args.kwargs
        assert sent["passthrough"]["model"] == "gateway-pin"
        if protocol == "anthropic":
            assert sent["inbound_anthropic_body"]["model"] == "gateway-pin"
        assert result["model"] == "gateway-pin"
        assert body["model"] == "caller-model"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("protocol", ["openai", "anthropic"])
    @pytest.mark.parametrize("stream", [False, True])
    async def test_response_identity_uses_effective_model(self, protocol, stream):
        client = _mock_client(TextResponse(content="ok"))
        client.model = "pinned-model"
        body = {
            "model": "caller-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
        }
        facts = RequestFacts()
        result = await handle_chat_completions(
            body, client, _context_manager(),
            client_adapter=ClientAdapter.ANTHROPIC, protocol=protocol,
            backend_protocol="anthropic", request_facts=facts,
        )
        assert facts.effective_model == "pinned-model"
        if not stream:
            assert result["model"] == "pinned-model"
        elif protocol == "openai":
            assert all(event["model"] == "pinned-model" for event in result)
        else:
            assert result[0]["message"]["model"] == "pinned-model"
            assert all(
                "model" not in event
                for event in result[1:]
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("protocol", ["openai", "anthropic"])
    @pytest.mark.parametrize("model", [None, ""])
    async def test_missing_model_fails_closed(self, model, protocol):
        client = _mock_client(TextResponse(content="ok"))
        client.model = None
        body = {"messages": [{"role": "user", "content": "hi"}], "model": model}
        with pytest.raises(MissingModelError):
            await handle_chat_completions(
                body, client, _context_manager(),
                client_adapter=ClientAdapter.ANTHROPIC, protocol=protocol,
                backend_protocol="anthropic",
            )
        client.send.assert_not_awaited()


# ── Native transparent passthrough ──────────────────────────


class TestNativePassthrough:
    """Native proxy passthrough keeps raw tools by default; raw messages
    are forwarded only when full reasoning replay preserves old behavior."""

    @pytest.mark.asyncio
    async def test_default_reasoning_replay_filters_raw_reasoning_only(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        messages = [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "old",
                "tool_calls": [],
                "name": "a1",
                "vendor": {"kept": True},
            },
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "latest",
                "tool_calls": [],
                "name": "a2",
            },
        ]
        await handle_chat_completions(
            _body(messages=messages, tools=[_tool_def("search")]),
            client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
        )
        # Default policy is "none": every reasoning field is stripped, but the
        # rest of each raw message survives verbatim.
        sent_messages = client.send.call_args.args[0]
        assert sent_messages[0]["name"] == "a1"
        assert sent_messages[0]["vendor"] == {"kept": True}
        assert "reasoning_content" not in sent_messages[0]
        assert sent_messages[1]["name"] == "a2"
        assert "reasoning_content" not in sent_messages[1]

    @pytest.mark.asyncio
    async def test_keep_last_reasoning_replay_keeps_latest_only(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        messages = [
            {"role": "assistant", "content": None, "reasoning_content": "old", "tool_calls": [], "name": "a1"},
            {"role": "assistant", "content": None, "reasoning_content": "latest", "tool_calls": [], "name": "a2"},
        ]
        await handle_chat_completions(
            _body(messages=messages, tools=[_tool_def("search")]),
            client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            reasoning_replay="keep-last",
        )
        sent_messages = client.send.call_args.args[0]
        assert "reasoning_content" not in sent_messages[0]
        assert sent_messages[1]["reasoning_content"] == "latest"

    @pytest.mark.asyncio
    async def test_raw_tools_forwarded_verbatim(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        params = {
            "type": "object",
            "properties": {"q": {"type": "string", "description": "the query"}},
            "required": ["q"],
            "additionalProperties": False,
        }
        tools = [_tool_def("search", parameters=params)]
        await handle_chat_completions(
            _body(tools=tools), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
        )
        # The backend sees the client's exact tools array (full schema, no
        # name/schema drift), not forge's reconstructed format_tool output.
        sent = client.send.call_args.kwargs["raw_openai_tools"]
        assert sent == tools
        # Respond is NOT appended by default.
        assert [t["function"]["name"] for t in sent] == ["search"]
        # tool_specs (validation sidecar) still passed separately.
        assert client.send.call_args.kwargs["tools"][0].name == "search"

    @pytest.mark.asyncio
    async def test_raw_messages_forwarded_verbatim(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        # An extra non-standard key proves no normalization/folding happened.
        messages = [{"role": "user", "content": "hi", "name": "u1"}]
        await handle_chat_completions(
            _body(messages=messages, tools=[_tool_def("search")]),
            client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            reasoning_replay="full",
        )
        sent_messages = client.send.call_args.args[0]
        assert sent_messages == messages

    @pytest.mark.asyncio
    async def test_inbound_body_mutation_does_not_affect_sent(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        tools = [_tool_def("search")]
        body = _body(tools=tools)
        await handle_chat_completions(
            body, client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
        )
        # Mutate the caller's body after the call — detached copy is unaffected.
        body["tools"][0]["function"]["name"] = "MUTATED"
        body["messages"][0]["content"] = "MUTATED"
        sent_tools = client.send.call_args.kwargs["raw_openai_tools"]
        sent_messages = client.send.call_args.args[0]
        assert sent_tools[0]["function"]["name"] == "search"
        assert sent_messages[0]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_respond_not_injected_by_default(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
        )
        sent = client.send.call_args.kwargs["raw_openai_tools"]
        names = [t["function"]["name"] for t in sent]
        assert "respond" not in names
        spec_names = [s.name for s in client.send.call_args.kwargs["tools"]]
        assert "respond" not in spec_names

    @pytest.mark.asyncio
    async def test_respond_injected_into_raw_tools_when_opted_in(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            inject_respond_tool=True,
        )
        sent = client.send.call_args.kwargs["raw_openai_tools"]
        names = [t["function"]["name"] for t in sent]
        assert names == ["search", "respond"]


class TestOllamaProxyIntegration:
    """Proxy → REAL OllamaClient (mocked transport): the seam that let #111/#115
    ship. Every other handler test mocks the client, so it asserts what reaches
    the client, not what the client actually POSTs to /api/chat. These assert the
    on-the-wire body on the raw-passthrough first attempt."""

    def _real_ollama(self, response_data):
        client = OllamaClient(base_url="http://test:11434", model="m", think=False)
        mock_http = AsyncMock()
        mock_http.stream = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = response_data
        resp.text = json.dumps(response_data)
        mock_http.post.return_value = resp
        client._http = mock_http
        return client

    @pytest.mark.asyncio
    async def test_no_tools_array_content_normalized(self):
        """#115 on the no-tools chat path (also raw-passthrough)."""
        client = self._real_ollama({"message": {"role": "assistant", "content": "ok"}})
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        await handle_chat_completions(
            _body(messages=messages), client, _context_manager(),
            client_adapter=ClientAdapter.OLLAMA,
        )
        body = client._http.post.call_args.kwargs["json"]
        assert body["messages"][0]["content"] == "hi"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_fixed_model_response_matches_downstream(self, stream):
        client = self._real_ollama({
            "message": {"role": "assistant", "content": "ok"},
        })
        facts = RequestFacts()

        result = await handle_chat_completions(
            _body(model="caller-alias", stream=stream),
            client,
            _context_manager(),
            client_adapter=ClientAdapter.OLLAMA,
            request_facts=facts,
        )

        downstream = client._http.post.call_args.kwargs["json"]
        assert downstream["model"] == "m"
        assert downstream["stream"] is False
        if stream:
            assert all(event["model"] == downstream["model"] for event in result)
        else:
            assert result["model"] == downstream["model"]
        assert facts.effective_model == downstream["model"]

    @pytest.mark.asyncio
    async def test_tool_history_string_args_normalized(self):
        """#111: a replayed assistant tool_calls turn with JSON-string args is
        coerced to a dict on the wire (the multi-turn 400)."""
        client = self._real_ollama({
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "search", "arguments": {"q": "ok"}}}
                ],
            }
        })
        messages = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "search", "arguments": '{"q": "1"}'}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        await handle_chat_completions(
            _body(messages=messages, tools=[_tool_def("search")]),
            client, _context_manager(),
            client_adapter=ClientAdapter.OLLAMA,
        )
        # First attempt is the raw-passthrough path where the bug lived.
        body = client._http.post.call_args_list[0].kwargs["json"]
        tc_msg = [m for m in body["messages"] if m.get("tool_calls")][0]
        assert tc_msg["tool_calls"][0]["function"]["arguments"] == {"q": "1"}


# ── Prompt capability handoff ───────────────────────────────


class TestPromptCapabilityHandoff:
    """In prompt capability (native_passthrough=False) the handler suppresses
    the verbatim passthrough so the request folds normally and the client's
    prompt path injects the tools. (The injection itself is covered by the
    LlamafileClient prompt-mode tests.)"""

    @pytest.mark.asyncio
    async def test_prompt_mode_suppresses_raw_tools(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            native_passthrough=False,
        )
        # No verbatim tools forwarded — the client's prompt path injects them.
        assert "raw_openai_tools" not in client.send.call_args.kwargs
        # tool_specs (the source for build_tool_prompt) are still passed.
        assert client.send.call_args.kwargs["tools"][0].name == "search"

    @pytest.mark.asyncio
    async def test_prompt_mode_folds_messages_not_verbatim(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        # A non-standard key would survive verbatim passthrough but is dropped
        # by fold_and_serialize — proving the raw transcript was NOT forwarded.
        messages = [{"role": "user", "content": "hi", "name": "u1"}]
        await handle_chat_completions(
            _body(messages=messages, tools=[_tool_def("search")]),
            client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
            native_passthrough=False,
        )
        sent_messages = client.send.call_args.args[0]
        assert sent_messages != messages
        assert "name" not in sent_messages[0]

    @pytest.mark.asyncio
    async def test_native_default_still_forwards_raw(self):
        # Sanity: default (native) path is unaffected by the new param.
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        tools = [_tool_def("search")]
        await handle_chat_completions(
            _body(tools=tools), client, _context_manager(),
            client_adapter=ClientAdapter.LLAMAFILE,
        )
        assert client.send.call_args.kwargs["raw_openai_tools"] == tools
