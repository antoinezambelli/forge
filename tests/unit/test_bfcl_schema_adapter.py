"""Unit tests for the BFCL schema adapter."""

from __future__ import annotations

import json
from pathlib import Path

from tests.eval.bfcl.schema_adapter import (
    load_ground_truth,
    load_test_entry,
    make_tool_def,
    normalize_schema,
    sanitize_param_name,
    sanitize_tool_name,
)


# -------------------------------------------------------------------
# normalize_schema
# -------------------------------------------------------------------


class TestNormalizeSchema:
    def test_dict_to_object(self):
        """Top-level 'type: dict' becomes 'type: object'."""
        schema = {"type": "dict", "properties": {"x": {"type": "string"}}}
        result = normalize_schema(schema)
        assert result["type"] == "object"

    def test_nested_dict_to_object(self):
        """Nested 'type: dict' also converted."""
        schema = {
            "type": "dict",
            "properties": {
                "config": {
                    "type": "dict",
                    "properties": {"key": {"type": "string"}},
                }
            },
        }
        result = normalize_schema(schema)
        assert result["type"] == "object"
        assert result["properties"]["config"]["type"] == "object"

    def test_capitalized_string(self):
        """Java/JS-style 'String' becomes 'string'."""
        schema = {"type": "dict", "properties": {"name": {"type": "String"}}}
        result = normalize_schema(schema)
        assert result["properties"]["name"]["type"] == "string"

    def test_capitalized_boolean(self):
        """Java/JS-style 'Boolean' becomes 'boolean'."""
        schema = {"type": "dict", "properties": {"flag": {"type": "Boolean"}}}
        result = normalize_schema(schema)
        assert result["properties"]["flag"]["type"] == "boolean"

    def test_float_to_number(self):
        """Python-style 'float' becomes 'number'."""
        schema = {"type": "dict", "properties": {"val": {"type": "float"}}}
        result = normalize_schema(schema)
        assert result["properties"]["val"]["type"] == "number"

    def test_arraylist_to_array(self):
        """Java-style 'ArrayList' becomes 'array'."""
        schema = {"type": "dict", "properties": {"items": {"type": "ArrayList"}}}
        result = normalize_schema(schema)
        assert result["properties"]["items"]["type"] == "array"

    def test_hashmap_to_object(self):
        """Java-style 'HashMap' becomes 'object'."""
        schema = {"type": "dict", "properties": {"data": {"type": "HashMap"}}}
        result = normalize_schema(schema)
        assert result["properties"]["data"]["type"] == "object"

    def test_long_to_integer(self):
        """Java-style 'long' becomes 'integer'."""
        schema = {"type": "dict", "properties": {"id": {"type": "long"}}}
        result = normalize_schema(schema)
        assert result["properties"]["id"]["type"] == "integer"

    def test_all_bfcl_types_in_one(self):
        """All non-standard BFCL types normalize correctly together."""
        schema = {
            "type": "dict",
            "properties": {
                "a": {"type": "String"},
                "b": {"type": "Boolean"},
                "c": {"type": "float"},
                "d": {"type": "double"},
                "e": {"type": "Array"},
                "f": {"type": "ArrayList"},
                "g": {"type": "HashMap"},
                "h": {"type": "long"},
                "i": {"type": "char"},
                "j": {"type": "tuple"},
                "k": {"type": "any"},
            },
        }
        result = normalize_schema(schema)
        assert result["type"] == "object"
        expected = {
            "a": "string", "b": "boolean", "c": "number", "d": "number",
            "e": "array", "f": "array", "g": "object", "h": "integer",
            "i": "string", "j": "array", "k": "string",
        }
        for key, expected_type in expected.items():
            assert result["properties"][key]["type"] == expected_type, (
                f"{key}: expected {expected_type}, got {result['properties'][key]['type']}"
            )

    def test_non_dict_unchanged(self):
        """Normal 'type: object' passes through."""
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = normalize_schema(schema)
        assert result["type"] == "object"

    def test_deep_copy(self):
        """Input schema is not mutated."""
        schema = {"type": "dict", "properties": {}}
        result = normalize_schema(schema)
        assert schema["type"] == "dict"  # Original unchanged
        assert result["type"] == "object"


# -------------------------------------------------------------------
# sanitize_tool_name
# -------------------------------------------------------------------


class TestSanitizeToolName:
    def test_clean_name_unchanged(self):
        assert sanitize_tool_name("calculate_area") == "calculate_area"

    def test_dotted_name(self):
        assert sanitize_tool_name("math.factorial") == "math_factorial"

    def test_multi_dot(self):
        assert sanitize_tool_name("board_game.chess.get_top_players") == "board_game_chess_get_top_players"

    def test_camelcase_dotted(self):
        assert sanitize_tool_name("AclApi.add_mapping") == "AclApi_add_mapping"

    def test_hyphen_preserved(self):
        assert sanitize_tool_name("my-tool") == "my-tool"

    def test_already_underscored(self):
        assert sanitize_tool_name("get_weather") == "get_weather"


# -------------------------------------------------------------------
# sanitize_param_name
# -------------------------------------------------------------------


class TestSanitizeParamName:
    def test_clean_name_unchanged(self):
        assert sanitize_param_name("player_name") == "player_name"

    def test_underscore_class(self):
        assert sanitize_param_name("_class") == "class_"

    def test_underscore_from(self):
        assert sanitize_param_name("_from") == "from_"

    def test_double_underscore(self):
        assert sanitize_param_name("__private") == "private_"

    def test_normalize_schema_renames_params(self):
        """normalize_schema strips leading underscores from property names."""
        schema = {
            "type": "dict",
            "properties": {
                "name": {"type": "string"},
                "_class": {"type": "string"},
                "_from": {"type": "string"},
            },
            "required": ["name", "_class"],
        }
        result = normalize_schema(schema)
        assert "class_" in result["properties"]
        assert "_class" not in result["properties"]
        assert "from_" in result["properties"]
        assert "_from" not in result["properties"]
        assert "name" in result["properties"]
        assert result["required"] == ["name", "class_"]

    def test_make_tool_def_with_underscore_params(self):
        """make_tool_def succeeds with underscore-prefixed params."""
        func = {
            "name": "create_player",
            "description": "Create a player.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "name": {"type": "string", "description": "Name."},
                    "_class": {"type": "string", "description": "Class."},
                },
                "required": ["name", "_class"],
            },
        }
        td = make_tool_def(func, lambda **kw: "ok")
        assert td.spec.name == "create_player"
        schema = td.spec.get_json_schema()
        assert "class_" in schema["properties"]
        assert "_class" not in schema["properties"]


# -------------------------------------------------------------------
# make_tool_def
# -------------------------------------------------------------------


class TestMakeToolDef:
    def test_simple_function(self):
        """Convert a basic BFCL function schema."""
        func = {
            "name": "calculate_area",
            "description": "Calculate area.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "base": {"type": "integer", "description": "The base."},
                    "height": {"type": "integer", "description": "The height."},
                },
                "required": ["base", "height"],
            },
        }
        stub = lambda **kw: "ok"
        td = make_tool_def(func, stub)
        assert td.spec.name == "calculate_area"
        schema = td.spec.get_json_schema()
        assert "base" in schema["properties"]
        assert "height" in schema["properties"]

    def test_dotted_name_sanitised(self):
        """Dotted function names like 'math.factorial' are sanitised."""
        func = {
            "name": "math.factorial",
            "description": "Factorial.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "number": {"type": "integer", "description": "The number."},
                },
                "required": ["number"],
            },
        }
        td = make_tool_def(func, lambda **kw: "ok")
        assert td.spec.name == "math_factorial"

    def test_optional_params(self):
        """Params not in required list are optional."""
        func = {
            "name": "search",
            "description": "Search.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "query": {"type": "string", "description": "Query."},
                    "limit": {"type": "integer", "description": "Max results."},
                },
                "required": ["query"],
            },
        }
        td = make_tool_def(func, lambda **kw: "ok")
        schema = td.spec.get_json_schema()
        assert "query" in schema.get("required", [])
        assert "limit" not in schema.get("required", [])

    def test_no_parameters_key(self):
        """Function with no parameters field."""
        func = {"name": "noop", "description": "Do nothing."}
        td = make_tool_def(func, lambda **kw: "ok")
        assert td.spec.name == "noop"

    def test_executor_wired(self):
        """The provided executor callable is wired to the ToolDef."""
        called = {}

        def my_exec(**kw):
            called.update(kw)
            return "result"

        func = {
            "name": "test_fn",
            "description": "Test.",
            "parameters": {"type": "dict", "properties": {}, "required": []},
        }
        td = make_tool_def(func, my_exec)
        td.callable(x=1)
        assert called == {"x": 1}


# -------------------------------------------------------------------
# load_test_entry
# -------------------------------------------------------------------


class TestLoadTestEntry:
    def test_parses_entry(self):
        """Extracts id, messages, and function schemas."""
        entry = {
            "id": "simple_python_0",
            "question": [[{"role": "user", "content": "Hello"}]],
            "function": [{"name": "fn", "description": "d", "parameters": {}}],
        }
        tid, msgs, funcs = load_test_entry(entry)
        assert tid == "simple_python_0"
        assert msgs == [{"role": "user", "content": "Hello"}]
        assert len(funcs) == 1


# -------------------------------------------------------------------
# load_ground_truth
# -------------------------------------------------------------------


class TestLoadGroundTruth:
    def test_parses_ground_truth(self):
        """Extracts ground truth list."""
        entry = {
            "id": "simple_python_0",
            "ground_truth": [{"calculate_area": {"base": [10], "height": [5]}}],
        }
        gt = load_ground_truth(entry)
        assert len(gt) == 1
        assert "calculate_area" in gt[0]


# -------------------------------------------------------------------
# Integration: real BFCL data
# -------------------------------------------------------------------


class TestRealBfclData:
    def test_first_simple_python_entry(self):
        """Smoke test: parse first entry of simple_python data."""
        data_path = (
            Path(__file__).parent.parent
            / "eval"
            / "bfcl"
            / "data"
            / "BFCL_v4_simple_python.json"
        )
        with open(data_path) as f:
            entry = json.loads(f.readline())
        tid, msgs, funcs = load_test_entry(entry)
        assert tid == "simple_python_0"
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        # Should convert without error
        td = make_tool_def(funcs[0], lambda **kw: "ok")
        schema = td.spec.get_json_schema()
        assert "properties" in schema
