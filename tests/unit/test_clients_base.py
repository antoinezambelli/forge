"""Shared argument decoding, reply parsing, and tool rendering behavior."""

from __future__ import annotations

from forge.clients.replies import openai_reply
from forge.core.codecs import decode_arguments, argument_value
from forge.core.messages import ValidArguments, InvalidArguments
from tests.canonical import reasoning


class TestDecodeToolArgs:
    """decode_tool_args: parse JSON-string args, fail-loud on malformed.

    Contract: return a dict for well-formed object args; return the raw
    (non-dict) value untouched for anything else, so ResponseValidator's
    args-shape check can route it to the tool-error channel. Never coerce a
    malformed payload to ``{}`` and never raise.
    """

    def test_decodes_each_supported_argument_shape(self) -> None:
        decoded = {"city": "Paris"}
        cases = [
            ("json object", '{"city": "Paris"}', {"city": "Paris"}),
            ("empty wire string", "", {}),
            ("JSON-encoded empty string", '""', ""),
            ("received null", None, None),
            ("malformed json", '{"city": ', '{"city": '),
            ("json list", "[1, 2]", [1, 2]),
            ("json number", "42", 42),
            ("json string", '"bare"', "bare"),
            ("decoded integer", 123, 123),
            ("decoded list", [1, 2], [1, 2]),
        ]
        for case, raw, expected in cases:
            decoded_args = decode_arguments(raw)
            assert (
                decoded_args.value
                if isinstance(decoded_args, ValidArguments)
                else decoded_args.received
            ) == expected, case

        # Already-decoded dictionaries pass through without a copy.
        assert decode_arguments(decoded).value is decoded


def test_canonical_tool_fields_win_without_mutating_definitions():
    from copy import deepcopy
    from forge.clients.base import format_tool
    from forge.clients.anthropic import AnthropicClient
    from forge.core.workflow import ToolDefinition

    schema = {
        "$defs": {"Item": {"type": "string"}},
        "additionalProperties": False,
        "properties": {"anything": True, "nothing": False},
        "x-unknown": [1, {"a": 2}],
    }
    extras = {
        "name": "wrong",
        "description": "wrong",
        "input_schema": [],
        "parameters": {"wrong": True},
        "strict": True,
    }
    original = deepcopy((schema, extras))
    definition = ToolDefinition("correct", "Description", schema, extras)
    before = deepcopy(definition)
    openai = format_tool(definition)
    anthropic = AnthropicClient._convert_tools([definition])[0]
    assert openai["function"]["name"] == anthropic["name"] == "correct"
    assert openai["function"]["description"] == anthropic["description"] == "Description"
    assert openai["function"]["parameters"] == anthropic["input_schema"] == schema
    assert openai["function"]["strict"] is anthropic["strict"] is True
    assert definition == before
    assert (schema, extras) == original


def test_empty_wire_arguments_are_distinct_from_decoded_empty_string():
    assert decode_arguments("") == ValidArguments({})
    assert decode_arguments('""') == InvalidArguments("")
    assert decode_arguments("", json_encoded=False) == InvalidArguments("")


def test_missing_function_retains_existing_fallback():
    reply = openai_reply({"tool_calls": [{}]})
    assert reply.tool_calls[0].name == ""
    assert argument_value(reply.tool_calls[0]) == {}


def test_reasoning_belongs_to_message_beside_all_calls():
    reply = openai_reply(
        {
            "reasoning": "because",
            "tool_calls": [
                {"function": {"name": "a", "arguments": "{}"}},
                {"function": {"name": "b", "arguments": "{}"}},
            ],
        }
    )
    assert [part.name for part in reply.tool_calls] == ["a", "b"]
    assert reasoning(reply) == "because"
