"""Anthropic-native request rendering and cache-control preservation on the wire."""

from __future__ import annotations
from tests.canonical import render, render_native
from forge.core.messages import ConversationState
import json
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from forge._backend_profiles import ClientAdapter
from forge.clients.anthropic import AnthropicClient
from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.proxy.handler import handle_chat_completions
from forge.proxy.proxy import ProxyServer


class TestProxyServerValidation:
    def test_anthropic_in_managed_mode_rejected(self):
        with pytest.raises(ValueError, match="requires backend_url"):
            ProxyServer(backend="anthropic")

    def test_anthropic_external_default_mode_ok(self):
        proxy = ProxyServer(backend_url="http://localhost:8080", backend="anthropic")
        assert proxy._backend_protocol == "anthropic"

    def test_openai_default_unchanged(self):
        proxy = ProxyServer(backend_url="http://localhost:8080")
        assert proxy._backend_protocol == "openai"

    @pytest.mark.asyncio
    async def test_anthropic_external_receives_backend_timeout(self):
        proxy = ProxyServer(
            backend_url="http://localhost:8080", backend="anthropic", backend_timeout=1800.0
        )
        with patch("forge.clients.anthropic.AnthropicClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_context_length = AsyncMock(return_value=200000)
            mock_client_cls.return_value = mock_client
            client, ctx, lazy = await proxy._setup_external()
        assert client is mock_client
        mock_client.get_context_length.assert_not_awaited()
        assert isinstance(ctx.strategy, NoCompact)
        assert ctx.budget_tokens is None
        assert lazy is None
        mock_client_cls.assert_called_once_with(
            model=None, base_url="http://localhost:8080", timeout=1800.0, api_key=""
        )

    @pytest.mark.asyncio
    async def test_anthropic_external_literal_claude_pin_is_preserved(self):
        proxy = ProxyServer(
            backend_url="http://localhost:8080", backend="anthropic", model="claude"
        )
        with patch("forge.clients.anthropic.AnthropicClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_context_length = AsyncMock(return_value=200000)
            mock_client_cls.return_value = mock_client
            await proxy._setup_external()
        assert mock_client_cls.call_args.kwargs["model"] == "claude"


class TestAnthropicClientNativeRequest:
    def test_native_request_fields_survive_rendering(self):
        client = AnthropicClient(model="claude-3-5-sonnet", api_key="dummy")
        inbound = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "long stable content"}]}
            ],
            "system": [
                {"type": "text", "text": "you are helpful", "cache_control": {"type": "ephemeral"}}
            ],
            "metadata": {"user_id": "test"},
            "stream": True,
        }
        kwargs = render_native(client, inbound).payload
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert kwargs["metadata"] == {"user_id": "test"}
        assert "stream" not in kwargs
        assert kwargs["messages"] == inbound["messages"]

    def test_verbatim_body_sets_model_default(self):
        """If inbound omits model, client's configured model fills in."""
        client = AnthropicClient(model="claude-3-5-sonnet", api_key="dummy")
        inbound = {"max_tokens": 256, "messages": []}
        kwargs = render_native(client, inbound).payload
        assert kwargs["model"] == "claude-3-5-sonnet"

    def test_fixed_client_model_wins_over_inbound_model(self):
        """A direct fixed-model client remains authoritative."""
        client = AnthropicClient(model="claude-default", api_key="dummy")
        inbound = {"model": "claude-opus-4-7", "messages": []}
        kwargs = render_native(client, inbound).payload
        assert kwargs["model"] == "claude-default"

    def test_none_inbound_uses_convert_messages_path(self):
        """When inbound_anthropic_body is None, falls back to rebuild path."""
        client = AnthropicClient(model="claude-3-5-sonnet", api_key="dummy")
        messages = [{"role": "system", "content": "be helpful"}, {"role": "user", "content": "hi"}]
        kwargs = render(client, messages).payload
        assert kwargs["system"] == "be helpful"
        assert kwargs["messages"][0]["role"] == "user"
        assert kwargs["max_tokens"] == client.max_tokens


class TestAnthropicClientBaseURL:
    @pytest.mark.asyncio
    async def test_base_url_routes_to_anthropic_shape_downstream(self):
        client = AnthropicClient(
            model="claude", base_url="http://litellm.local:4000", api_key="dummy"
        )
        urls = []

        def respond(request):
            urls.append(str(request.url))
            return httpx.Response(200, json=_stub_anthropic_response())

        client._http = httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
            base_url=client._http.base_url,
            headers=client._http.headers,
        )
        await client.send(render(client, [{"role": "user", "content": "hi"}]))
        assert urls == ["http://litellm.local:4000/v1/messages"]


def _stub_anthropic_response():
    return {
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _stub_tool_response(name="search", **tool_input):
    return {
        "content": [{"type": "tool_use", "name": name, "input": tool_input}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _stub_thinking_tool_response(name="search", **tool_input):
    return {
        "content": [
            {"type": "thinking", "thinking": "I should search."},
            {"type": "tool_use", "name": name, "input": tool_input},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _mock_anthropic_http(client, responses=None):
    """Capture real request JSON while serving a deterministic response sequence."""
    responses = iter(responses or [_stub_anthropic_response()])
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=next(responses))

    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        headers=client._http.headers,
        base_url=client._http.base_url,
    )
    return calls


class TestCacheControlSurvivesWire:
    """The headline path-1 capability: a cache_control block on inbound must
    reach the downstream HTTP request unchanged."""

    @pytest.mark.asyncio
    async def test_cache_control_on_system_block_reaches_http(self):
        client = AnthropicClient(model="claude-3-5-sonnet", api_key="dummy")
        calls = _mock_anthropic_http(client)
        inbound = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 1024,
            "system": [
                {
                    "type": "text",
                    "text": "large stable system prompt",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }
        await client.send(render_native(client, inbound))
        assert len(calls) == 1
        kwargs = calls[0]
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert kwargs["system"][0]["text"] == "large stable system prompt"

    @pytest.mark.asyncio
    async def test_cache_control_on_message_block_reaches_http(self):
        client = AnthropicClient(model="claude-3-5-sonnet", api_key="dummy")
        calls = _mock_anthropic_http(client)
        inbound = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "huge cached prefix",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": "fresh query"},
                    ],
                }
            ],
        }
        await client.send(render_native(client, inbound))
        kwargs = calls[0]
        msg_blocks = kwargs["messages"][0]["content"]
        assert msg_blocks[0]["cache_control"] == {"type": "ephemeral"}


class TestAnthropicReasoningProxyParity:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_openai_client_receives_keep_last_reasoning(self, stream) -> None:
        client = AnthropicClient(model="claude-test", api_key="dummy")
        _mock_anthropic_http(client, [_stub_thinking_tool_response(q="forge")])
        body = {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "search for forge"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search.",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    },
                }
            ],
            "stream": stream,
        }
        result = await handle_chat_completions(
            body,
            client,
            ContextManager(strategy=NoCompact(), budget_tokens=None),
            client_adapter=ClientAdapter.ANTHROPIC,
            protocol="openai",
            backend_protocol="anthropic",
            reasoning_replay="keep-last",
        )
        if stream:
            assert any(
                (
                    event.get("choices", [{}])[0].get("delta", {}).get("reasoning_content")
                    == "I should search."
                    for event in result
                )
            )
        else:
            assert result["choices"][0]["message"]["reasoning_content"] == "I should search."


class TestRequestLocalModelSurvivesMutation:
    @staticmethod
    def _body(model):
        return {
            "model": model,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "search"}],
            "tools": [
                {
                    "name": "search",
                    "description": "Search.",
                    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                }
            ],
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("client_model", "inbound_model", "effective_model"),
        [(None, "route-opus", "route-opus"), ("claude", "caller-model", "claude")],
    )
    async def test_model_survives_every_retry_rebuild(
        self, client_model, inbound_model, effective_model
    ):
        client = AnthropicClient(model=client_model, api_key="dummy")
        calls = _mock_anthropic_http(
            client,
            [_stub_anthropic_response(), _stub_anthropic_response(), _stub_tool_response(q="done")],
        )
        result = await handle_chat_completions(
            self._body(inbound_model),
            client,
            MagicMock(
                maybe_compact=MagicMock(side_effect=lambda messages, **_: messages),
                check_thresholds=MagicMock(return_value=None),
            ),
            client_adapter=ClientAdapter.ANTHROPIC,
            protocol="anthropic",
            backend_protocol="anthropic",
            max_retries=2,
        )
        assert [call["model"] for call in calls] == [
            effective_model,
            effective_model,
            effective_model,
        ]
        assert calls[0]["messages"] == self._body(inbound_model)["messages"]
        assert result["model"] == effective_model
        assert client.model == client_model

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("client_model", "inbound_model", "effective_model"),
        [(None, "route-sonnet", "route-sonnet"), ("pinned", "caller", "pinned")],
    )
    async def test_model_survives_forced_compaction_rebuild(
        self, client_model, inbound_model, effective_model
    ):
        client = AnthropicClient(model=client_model, api_key="dummy")
        calls = _mock_anthropic_http(client, [_stub_tool_response(q="done")])
        context_manager = MagicMock()
        context_manager.maybe_compact.side_effect = lambda state, **_: ConversationState(
            list(state.messages)
        )
        context_manager.check_thresholds.return_value = None
        result = await handle_chat_completions(
            self._body(inbound_model),
            client,
            context_manager,
            client_adapter=ClientAdapter.ANTHROPIC,
            protocol="anthropic",
            backend_protocol="anthropic",
        )
        kwargs = calls[0]
        assert kwargs["model"] == effective_model
        assert result["model"] == effective_model
        assert client.model == client_model
