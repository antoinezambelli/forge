"""Tests for BFCL stub executors and done tool."""

import pytest
from pydantic import ValidationError

from forge.core.workflow import ToolSpec
from tests.eval.bfcl.executors import make_done_tool, make_stub_executor


def _make_spec(name, properties, required):
    """Helper: build a ToolSpec from a JSON Schema dict."""
    return ToolSpec.from_json_schema(name, name, {
        "type": "object",
        "properties": properties,
        "required": required,
    })


class TestMakeStubExecutor:
    def test_returns_success_string(self):
        """Stub returns deterministic success message."""
        spec = _make_spec("my_func", {"x": {"type": "integer"}}, ["x"])
        executor = make_stub_executor(spec)
        result = executor(x=5)
        assert result == "[stub] my_func executed"

    def test_missing_required_raises(self):
        """Missing required param raises ValidationError."""
        spec = _make_spec("my_func", {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
        }, ["x", "y"])
        executor = make_stub_executor(spec)
        with pytest.raises(ValidationError):
            executor(x=5)  # y is missing

    def test_wrong_type_raises(self):
        """Wrong arg type raises ValidationError (strict mode)."""
        spec = _make_spec("my_func", {"x": {"type": "integer"}}, ["x"])
        executor = make_stub_executor(spec)
        with pytest.raises(ValidationError):
            executor(x="not_an_int")

    def test_strict_rejects_string_to_int_coercion(self):
        """Strict mode rejects '5' for an int field (no coercion)."""
        spec = _make_spec("my_func", {"x": {"type": "integer"}}, ["x"])
        executor = make_stub_executor(spec)
        with pytest.raises(ValidationError):
            executor(x="5")

    def test_number_accepts_float(self):
        """'number' type (mapped to float) accepts float."""
        spec = _make_spec("my_func", {"x": {"type": "number"}}, ["x"])
        executor = make_stub_executor(spec)
        assert executor(x=5.0) == "[stub] my_func executed"

    def test_optional_param_can_be_omitted(self):
        """Params not in required list can be absent."""
        spec = _make_spec("my_func", {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
        }, ["x"])
        executor = make_stub_executor(spec)
        assert executor(x=5) == "[stub] my_func executed"

    def test_no_params_schema(self):
        """Function with no parameters works."""
        spec = ToolSpec.from_json_schema("noop", "Nothing", {
            "type": "object", "properties": {},
        })
        executor = make_stub_executor(spec)
        assert executor() == "[stub] noop executed"

    def test_string_type_rejects_int(self):
        """String param rejects non-string in strict mode."""
        spec = _make_spec("my_func", {"name": {"type": "string"}}, ["name"])
        executor = make_stub_executor(spec)
        with pytest.raises(ValidationError):
            executor(name=123)


class TestMakeDoneTool:
    def test_done_tool_name(self):
        """Done tool has correct name."""
        td = make_done_tool()
        assert td.spec.name == "done"

    def test_done_callable_returns_message(self):
        """Done callable echoes back the message."""
        td = make_done_tool()
        result = td.callable(message="all done")
        assert result == "all done"

    def test_done_callable_default(self):
        """Done callable works with no message."""
        td = make_done_tool()
        result = td.callable()
        assert result == ""
