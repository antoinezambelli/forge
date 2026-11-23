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
    [
        ("none", []),
        ("keep-last", ["last1", "last2"]),
        ("full", ["old1", "old2", "last1", "last2"]),
    ],
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
    assert [
        p.text for m in view for p in m.parts if isinstance(p, ReasoningPart)
    ] == expected
    wire = render_openai(view) if target == "openai" else render_anthropic(view)[1]
    serialized = str(wire)
    for text in expected:
        assert text in serialized
    assert "later visible text" in serialized
    assert state == before


def test_user_reasoning_siblings_are_extras_preserved_even_under_none():
    raw = [{"role": "user", "content": "x", "reasoning": "user extension"}]
    assert render_openai(replay_messages(openai_state(raw), "none")) == raw


def test_unknown_policy_fails():
    with pytest.raises(ValueError, match="reasoning_replay must be one of"):
        validate_reasoning_replay("latest")
