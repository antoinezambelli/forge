"""Unit tests for shared client helpers in forge.clients.base."""

from __future__ import annotations

from forge.clients.base import decode_tool_args


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
            ("empty string", "", {}),
            ("missing arguments", None, {}),
            ("malformed json", '{"city": ', '{"city": '),
            ("json list", "[1, 2]", [1, 2]),
            ("json number", "42", 42),
            ("json string", '"bare"', "bare"),
            ("decoded integer", 123, 123),
            ("decoded list", [1, 2], [1, 2]),
        ]
        for case, raw, expected in cases:
            assert decode_tool_args(raw) == expected, case

        # Already-decoded dictionaries pass through without a copy.
        assert decode_tool_args(decoded) is decoded
