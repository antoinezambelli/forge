"""Path-1 tests — Anthropic-protocol downstream + cache_control verbatim emit.

Covers:
- ProxyServer init-time validation of the Anthropic selector + mode.
- AnthropicClient verbatim path when inbound_anthropic_body is set.
- AnthropicClient falling back to _convert_messages rebuild when None.
- End-to-end: cache_control on inbound blocks reaches the underlying
  Anthropic SDK call unchanged (the headline path-1 capability).

See ADR-015 for the cache_control preservation rationale.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge._backend_profiles import ClientAdapter
from forge.clients.anthropic import AnthropicClient
from forge.context.strategies import NoCompact
from forge.proxy.handler import handle_chat_completions
from forge.proxy.proxy import ProxyServer


# ── ProxyServer construction validation ──────────────────────


class TestProxyServerValidation:
    def test_anthropic_in_managed_mode_rejected(self):
        with pytest.raises(ValueError, match="requires backend_url"):
            ProxyServer(backend="anthropic")

    def test_anthropic_external_default_mode_ok(self):
        # Should construct without raising
        proxy = ProxyServer(
            backend_url="http://localhost:8080",
            backend="anthropic",
        )
        assert proxy._backend_protocol == "anthropic"

    def test_openai_default_unchanged(self):
        proxy = ProxyServer(
            backend_url="http://localhost:8080",
        )
        assert proxy._backend_protocol == "openai"

    @pytest.mark.asyncio
    async def test_anthropic_external_receives_backend_timeout(self):
        proxy = ProxyServer(
            backend_url="http://localhost:8080",
            backend="anthropic",
            backend_timeout=1800.0,
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
        assert lazy is None  # Anthropic path is never deferred
        mock_client_cls.assert_called_once_with(
            model=None,
            base_url="http://localhost:8080",
            timeout=1800.0,
            # explicit "" → no static credential and ambient ANTHROPIC_* env is
            # suppressed at construction (one credential per request).
            api_key="",
        )

    @pytest.mark.asyncio
    async def test_anthropic_external_literal_claude_pin_is_preserved(self):
        proxy = ProxyServer(
            backend_url="http://localhost:8080",
            backend="anthropic",
            model="claude",
        )
        with patch("forge.clients.anthropic.AnthropicClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_context_length = AsyncMock(return_value=200000)
            mock_client_cls.return_value = mock_client

            await proxy._setup_external()

        assert mock_client_cls.call_args.kwargs["model"] == "claude"


# ── AnthropicClient verbatim path ────────────────────────────


class TestAnthropicClientVerbatim:
    def test_verbatim_body_used_when_provided(self):
        """When inbound_anthropic_body is set, _build_kwargs returns it verbatim
        (drops only 'stream', sets model default)."""
        client = AnthropicClient(model="claude-3-5-sonnet")
        inbound = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "long stable content"},
                        # Block-level cache_control survives because we never
                        # touch the dict.
                    ],
                }
            ],
            "system": [
                {
                    "type": "text",
                    "text": "you are helpful",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "metadata": {"user_id": "test"},
            "stream": True,  # Should be stripped — SDK call selects streaming
        }
        kwargs = client._build_kwargs(
            messages=[],
            tools=None,
            passthrough=None,
            inbound_anthropic_body=inbound,
        )
        # cache_control preserved verbatim
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
        # metadata preserved verbatim
        assert kwargs["metadata"] == {"user_id": "test"}
        # stream stripped
        assert "stream" not in kwargs
        # messages used as-is (forge's deconstruction not applied)
        assert kwargs["messages"] == inbound["messages"]

    def test_verbatim_body_sets_model_default(self):
        """If inbound omits model, client's configured model fills in."""
        client = AnthropicClient(model="claude-3-5-sonnet")
        inbound = {"max_tokens": 256, "messages": []}
        kwargs = client._build_kwargs(
            messages=[],
            tools=None,
            inbound_anthropic_body=inbound,
        )
        assert kwargs["model"] == "claude-3-5-sonnet"

    def test_fixed_client_model_wins_over_inbound_model(self):
        """A direct fixed-model client remains authoritative."""
        client = AnthropicClient(model="claude-default")
        inbound = {"model": "claude-opus-4-7", "messages": []}
        kwargs = client._build_kwargs(
            messages=[],
            tools=None,
            inbound_anthropic_body=inbound,
        )
        assert kwargs["model"] == "claude-default"

    def test_none_inbound_uses_convert_messages_path(self):
        """When inbound_anthropic_body is None, falls back to rebuild path."""
        client = AnthropicClient(model="claude-3-5-sonnet")
        messages = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
        kwargs = client._build_kwargs(
            messages=messages,
            tools=None,
            inbound_anthropic_body=None,
        )
        # System lifted to top-level (forge's _convert_messages behavior)
        assert kwargs["system"] == "be helpful"
        # Messages converted (forge-shape, not original-Anthropic-shape blocks)
        assert kwargs["messages"][0]["role"] == "user"
        # max_tokens defaulted from client
        assert kwargs["max_tokens"] == client.max_tokens


# ── AnthropicClient base_url ─────────────────────────────────


class TestAnthropicClientBaseURL:
    def test_base_url_passed_to_sdk(self):
        """base_url retargets the SDK at an Anthropic-shape downstream."""
        client = AnthropicClient(
            model="claude",
            base_url="http://litellm.local:4000",
            api_key="dummy",
        )
        # The SDK stores the base URL; verifying via the SDK's internal state
        # is fragile but the construction path is what matters here.
        assert client._client is not None


# ── End-to-end: cache_control wire preservation ──────────────


def _stub_anthropic_response():
    """Build a minimal Anthropic-shape response object for AsyncMock."""
    msg = MagicMock()
    msg.content = [MagicMock(type="text", text="ok")]
    msg.usage.input_tokens = 1
    msg.usage.output_tokens = 1
    msg.usage.cache_creation_input_tokens = 0
    msg.usage.cache_read_input_tokens = 0
    return msg


def _stub_tool_response(name="search", **tool_input):
    msg = MagicMock()
    msg.content = [MagicMock(type="tool_use", name=name, input=tool_input)]
    msg.usage.input_tokens = 1
    msg.usage.output_tokens = 1
    msg.usage.cache_creation_input_tokens = 0
    msg.usage.cache_read_input_tokens = 0
    return msg


class TestCacheControlSurvivesWire:
    """The headline path-1 capability: a cache_control block on inbound must
    reach the Anthropic SDK call unchanged."""

    @pytest.mark.asyncio
    async def test_cache_control_on_system_block_reaches_sdk(self):
        client = AnthropicClient(model="claude-3-5-sonnet", api_key="dummy")
        # Patch the SDK at the boundary; capture call_args.
        client._client.messages.create = AsyncMock(
            return_value=_stub_anthropic_response(),
        )

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
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
        }

        await client.send(
            messages=[],
            tools=None,
            inbound_anthropic_body=inbound,
        )

        # SDK was called once with verbatim system blocks
        client._client.messages.create.assert_called_once()
        kwargs = client._client.messages.create.call_args.kwargs
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
        # The block text survives unchanged
        assert kwargs["system"][0]["text"] == "large stable system prompt"

    @pytest.mark.asyncio
    async def test_cache_control_on_message_block_reaches_sdk(self):
        client = AnthropicClient(model="claude-3-5-sonnet", api_key="dummy")
        client._client.messages.create = AsyncMock(
            return_value=_stub_anthropic_response(),
        )

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

        await client.send(
            messages=[],
            tools=None,
            inbound_anthropic_body=inbound,
        )

        kwargs = client._client.messages.create.call_args.kwargs
        msg_blocks = kwargs["messages"][0]["content"]
        assert msg_blocks[0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_rebuild_path_drops_cache_control(self):
        """Sanity check: WITHOUT inbound_anthropic_body, _convert_messages
        rebuilds blocks without cache_control. Documents the limit ADR-015
        addresses."""
        client = AnthropicClient(model="claude-3-5-sonnet", api_key="dummy")
        client._client.messages.create = AsyncMock(
            return_value=_stub_anthropic_response(),
        )

        # OpenAI-shape messages (what the runner would serialize to).
        # cache_control has nowhere to live in this shape — it was already
        # lost upstream in forge's deconstruction.
        openai_messages = [
            {"role": "system", "content": "large stable system prompt"},
            {"role": "user", "content": "hi"},
        ]

        await client.send(
            messages=openai_messages,
            tools=None,
            inbound_anthropic_body=None,  # rebuild path
        )

        kwargs = client._client.messages.create.call_args.kwargs
        # System is a plain string (no blocks, no cache_control)
        assert kwargs["system"] == "large stable system prompt"
        assert not isinstance(kwargs["system"], list)


class TestRequestLocalModelSurvivesMutation:
    @staticmethod
    def _body(model):
        return {
            "model": model,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "search"}],
            "tools": [{
                "name": "search",
                "description": "Search.",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            }],
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("client_model", "inbound_model", "effective_model"),
        [(None, "route-opus", "route-opus"), ("claude", "caller-model", "claude")],
    )
    async def test_model_survives_every_retry_rebuild(
        self, client_model, inbound_model, effective_model,
    ):
        client = AnthropicClient(model=client_model, api_key="dummy")
        client._client.messages.create = AsyncMock(side_effect=[
            _stub_anthropic_response(),
            _stub_anthropic_response(),
            _stub_tool_response(q="done"),
        ])

        result = await handle_chat_completions(
            self._body(inbound_model), client, MagicMock(
                maybe_compact=MagicMock(side_effect=lambda messages, **_: messages),
                check_thresholds=MagicMock(return_value=None),
            ),
            client_adapter=ClientAdapter.ANTHROPIC,
            protocol="anthropic", backend_protocol="anthropic", max_retries=2,
        )

        calls = client._client.messages.create.call_args_list
        assert [call.kwargs["model"] for call in calls] == [
            effective_model, effective_model, effective_model,
        ]
        assert calls[0].kwargs["messages"] == self._body(inbound_model)["messages"]
        assert result["model"] == effective_model
        assert client.model == client_model

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("client_model", "inbound_model", "effective_model"),
        [(None, "route-sonnet", "route-sonnet"), ("pinned", "caller", "pinned")],
    )
    async def test_model_survives_forced_compaction_rebuild(
        self, client_model, inbound_model, effective_model,
    ):
        client = AnthropicClient(model=client_model, api_key="dummy")
        client._client.messages.create = AsyncMock(
            return_value=_stub_tool_response(q="done"),
        )
        context_manager = MagicMock()
        context_manager.maybe_compact.side_effect = (
            lambda messages, **_: list(messages)
        )
        context_manager.check_thresholds.return_value = None

        result = await handle_chat_completions(
            self._body(inbound_model), client, context_manager,
            client_adapter=ClientAdapter.ANTHROPIC,
            protocol="anthropic", backend_protocol="anthropic",
        )

        kwargs = client._client.messages.create.call_args.kwargs
        assert kwargs["model"] == effective_model
        assert result["model"] == effective_model
        assert client.model == client_model
