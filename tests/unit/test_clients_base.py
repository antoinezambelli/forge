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

    def test_valid_json_object_decoded(self) -> None:
        assert decode_tool_args('{"city": "Paris"}') == {"city": "Paris"}

    def test_empty_string_is_no_arg_call(self) -> None:
        assert decode_tool_args("") == {}

    def test_none_is_no_arg_call(self) -> None:
        # Missing "arguments" key — a no-arg call, not a failure.
        assert decode_tool_args(None) == {}

    def test_already_decoded_dict_passes_through(self) -> None:
        # Ollama and the Anthropic SDK hand back parsed dicts.
        d = {"city": "Paris"}
        assert decode_tool_args(d) is d

    def test_malformed_json_kept_as_raw_string(self) -> None:
        # The crux: malformed args are NOT coerced to {} and do NOT raise —
        # the raw string (a non-dict) survives for the validator to catch.
        assert decode_tool_args('{"city": ') == '{"city": '

    def test_valid_json_non_object_kept_as_is(self) -> None:
        # Parseable but not an object (list / scalar) — a non-dict the
        # validator must reject, so it rides through unchanged.
        assert decode_tool_args("[1, 2]") == [1, 2]
        assert decode_tool_args("42") == 42
        assert decode_tool_args('"bare"') == "bare"

    def test_non_string_non_dict_passes_through(self) -> None:
        # Any other already-decoded shape is left for the validator to judge.
        assert decode_tool_args(123) == 123
        assert decode_tool_args([1, 2]) == [1, 2]
