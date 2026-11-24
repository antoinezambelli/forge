"""Replay selects complete assistant reasoning on a disposable backend view."""

from copy import deepcopy
import pytest
from forge.core.codecs import (
    openai_state,
    anthropic_state,
    render_openai,
    render_anthropic,
    replay_messages,
)
from forge.core.messages import ReasoningPart
from forge.core.reasoning import validate_reasoning_replay


@pytest.mark.parametrize("source", ["openai", "anthropic"])
@pytest.mark.parametrize("target", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "mode,expected",
    [("none", []), ("keep-last", ["last1", "last2"]), ("full", ["old1", "old2", "last1", "last2"])],
)
def test_complete_last_reasoning_message_and_cross_protocol_nonmutation(
    source, target, mode, expected
):
    if source == "openai":
        state = openai_state(
            [
                {
                    "role": "assistant",
                    "content": "old text",
                    "reasoning_content": "old1",
                    "reasoning": "old2",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "last1",
                    "reasoning_text": "last2",
                },
                {"role": "assistant", "content": "later visible text"},
            ]
        )
    else:
        state = anthropic_state(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "old1"},
                        {"type": "thinking", "thinking": "old2"},
                        {"type": "text", "text": "old text"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "last1", "signature": "a"},
                        {"type": "thinking", "thinking": "last2", "signature": "b"},
                    ],
                },
                {"role": "assistant", "content": "later visible text"},
            ]
        )
    before = deepcopy(state)
    view = replay_messages(state, mode)
    assert [p.text for m in view for p in m.parts if isinstance(p, ReasoningPart)] == expected
    wire = render_openai(view) if target == "openai" else render_anthropic(view)[1]
    serialized = str(wire)
    if source == "openai" and target == "anthropic":
        for text in ("old1", "old2", "last1", "last2"):
            assert text not in serialized
    else:
        for text in expected:
            assert text in serialized
    if source == "anthropic" and target == "anthropic":
        thinking = {
            block["thinking"]: block
            for message in wire
            if isinstance(message["content"], list)
            for block in message["content"]
            if block["type"] == "thinking"
        }
        for text, signature in (("last1", "a"), ("last2", "b")):
            if text in expected:
                assert thinking[text]["signature"] == signature
    if source == "anthropic" and target == "openai":
        assert "signature" not in serialized
    assert "later visible text" in serialized
    assert state == before


def test_anthropic_request_omits_foreign_reasoning_without_changing_history():
    from forge.clients.anthropic import AnthropicClient
    from forge.clients.base import RequestMetadata
    from tests.http_mocks import build_with_mock_http

    state = openai_state(
        [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "visible",
                "reasoning_content": "private one",
                "reasoning": "private two",
                "reasoning_text": "private three",
                "tool_calls": [{"id": "c", "function": {"name": "search", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c", "content": "result"},
        ]
    )
    before = deepcopy(state)
    client = build_with_mock_http(AnthropicClient, "model", api_key="dummy")
    request = client.render_request(
        state, None, RequestMetadata(reasoning_replay="full", source_protocol="openai")
    )
    assert request.payload["messages"] == [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "visible"},
                {"type": "tool_use", "id": "c", "name": "search", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c", "content": "result"}],
        },
    ]
    assert state == before
    openai_view = render_openai(state.messages)[1]
    assert openai_view["reasoning_content"] == "private one"
    assert openai_view["reasoning"] == "private two"
    assert openai_view["reasoning_text"] == "private three"


@pytest.mark.parametrize("visible", ["visible", ""])
def test_anthropic_renderer_omits_inline_openai_reasoning(visible):
    from forge.clients.replies import openai_reply
    from forge.core.messages import ConversationState

    state = ConversationState(
        [openai_reply({"content": "<think>private reasoning</think>" + visible})]
    )
    before = deepcopy(state)
    assert render_openai(state.messages)[0]["reasoning_content"] == "private reasoning"
    _, wire = render_anthropic(replay_messages(state, "full"))
    assert wire == (
        [{"role": "assistant", "content": [{"type": "text", "text": visible}]}] if visible else []
    )
    assert state == before


def test_user_reasoning_siblings_are_extras_preserved_even_under_none():
    raw = [{"role": "user", "content": "x", "reasoning": "user extension"}]
    assert render_openai(replay_messages(openai_state(raw), "none")) == raw


def test_unknown_policy_fails():
    with pytest.raises(ValueError, match="reasoning_replay must be one of"):
        validate_reasoning_replay("latest")
