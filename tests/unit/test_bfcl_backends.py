"""Tests for BFCL backend wiring (multi-turn executors)."""

import json
from pathlib import Path

import pytest

from tests.eval.bfcl.backend_wiring import create_instances, wire_tools
from tests.eval.bfcl.schema_adapter import (
    CLASS_MODULE_MAP,
    STATELESS_CLASSES,
    MULTI_TURN_CATEGORIES,
    load_func_docs,
    load_jsonl,
    load_multi_turn_entry,
    load_multi_turn_ground_truth,
)


_DATA_DIR = Path(__file__).parent.parent / "eval" / "bfcl" / "data"


class TestClassModuleMap:
    """Verify the class-to-module mapping is correct."""

    def test_all_classes_importable(self):
        """Every class in CLASS_MODULE_MAP can be imported."""
        import importlib
        for class_name, module_name in CLASS_MODULE_MAP.items():
            module = importlib.import_module(f"tests.eval.bfcl.backends.{module_name}")
            cls = getattr(module, class_name)
            assert cls is not None

    def test_all_classes_have_init(self):
        """Every class can be instantiated with no args."""
        import importlib
        for class_name, module_name in CLASS_MODULE_MAP.items():
            module = importlib.import_module(f"tests.eval.bfcl.backends.{module_name}")
            cls = getattr(module, class_name)
            instance = cls()
            assert instance is not None

    def test_stateless_class_has_no_load_scenario(self):
        """MathAPI has no _load_scenario method."""
        import importlib
        for class_name in STATELESS_CLASSES:
            module_name = CLASS_MODULE_MAP[class_name]
            module = importlib.import_module(f"tests.eval.bfcl.backends.{module_name}")
            cls = getattr(module, class_name)
            assert not hasattr(cls(), "_load_scenario")


class TestCreateInstances:
    """Test backend instance creation and configuration."""

    def test_creates_gorilla_filesystem(self):
        """GorillaFileSystem instance is created and configured."""
        config = {
            "GorillaFileSystem": {
                "root": {
                    "workspace": {
                        "type": "directory",
                        "contents": {
                            "test.txt": {"type": "file", "content": "hello"}
                        },
                    }
                }
            }
        }
        instances = create_instances(config, ["GorillaFileSystem"])
        assert "GorillaFileSystem" in instances
        fs = instances["GorillaFileSystem"]
        # The filesystem should be configured — cat should work
        result = fs.cat(file_name="test.txt")
        assert "hello" in result.get("file_content", "")

    def test_creates_stateless_math_api(self):
        """MathAPI is created without _load_scenario."""
        instances = create_instances({}, ["MathAPI"])
        assert "MathAPI" in instances
        result = instances["MathAPI"].add(a=2.0, b=3.0)
        assert result["result"] == 5.0

    def test_creates_multiple_instances(self):
        """Multiple classes are instantiated together."""
        config = {
            "GorillaFileSystem": {
                "root": {
                    "workspace": {"type": "directory", "contents": {}}
                }
            },
            "TwitterAPI": {
                "tweet_counter": 0,
                "tweets": {},
                "username": "test",
                "password": "pass",
            },
        }
        instances = create_instances(config, ["GorillaFileSystem", "TwitterAPI"])
        assert len(instances) == 2
        assert "GorillaFileSystem" in instances
        assert "TwitterAPI" in instances

    def test_deep_copies_config(self):
        """initial_config is deep-copied so mutations don't leak."""
        config = {
            "GorillaFileSystem": {
                "root": {
                    "workspace": {"type": "directory", "contents": {}}
                }
            }
        }
        original_config = json.dumps(config)
        create_instances(config, ["GorillaFileSystem"])
        assert json.dumps(config) == original_config


class TestWireTools:
    """Test wiring backend methods as forge ToolDefs."""

    def test_wires_math_api_tools(self):
        """MathAPI methods become ToolDefs with working callables."""
        instances = create_instances({}, ["MathAPI"])
        func_docs = load_func_docs(["MathAPI"])
        tools = wire_tools(instances, func_docs)

        assert "add" in tools
        # Call the tool — should invoke the real MathAPI.add()
        result = tools["add"].callable(a=2.0, b=3.0)
        assert '"result": 5.0' in result

    def test_excluded_functions_omitted(self):
        """Functions in excluded_functions are not wired."""
        instances = create_instances({}, ["MathAPI"])
        func_docs = load_func_docs(["MathAPI"])
        tools = wire_tools(instances, func_docs, excluded_functions=["add", "subtract"])

        assert "add" not in tools
        assert "subtract" not in tools
        assert "multiply" in tools

    def test_missed_functions_omitted_for_turn(self):
        """Functions in missed_function are omitted for the specified turn."""
        instances = create_instances({}, ["MathAPI"])
        func_docs = load_func_docs(["MathAPI"])

        # Turn 2: "add" is missed
        tools = wire_tools(
            instances, func_docs,
            missed_functions={"2": ["add"]},
            turn_index=2,
        )
        assert "add" not in tools

        # Turn 0: "add" is NOT missed
        tools = wire_tools(
            instances, func_docs,
            missed_functions={"2": ["add"]},
            turn_index=0,
        )
        assert "add" in tools

    def test_gorilla_fs_state_persists(self):
        """Tool calls mutate state on the shared instance."""
        config = {
            "GorillaFileSystem": {
                "root": {
                    "workspace": {
                        "type": "directory",
                        "contents": {
                            "docs": {
                                "type": "directory",
                                "contents": {},
                            },
                        },
                    }
                }
            }
        }
        instances = create_instances(config, ["GorillaFileSystem"])
        func_docs = load_func_docs(["GorillaFileSystem"])
        tools = wire_tools(instances, func_docs)

        # Create a file via the tool
        tools["touch"].callable(file_name="newfile.txt")
        # Read it back — should exist
        ls_result = json.loads(tools["ls"].callable())
        assert "newfile.txt" in ls_result["current_directory_content"]

    def test_result_normalization_dict(self):
        """Dict results are JSON-serialized."""
        instances = create_instances({}, ["MathAPI"])
        func_docs = load_func_docs(["MathAPI"])
        tools = wire_tools(instances, func_docs)
        result = tools["add"].callable(a=1.0, b=2.0)
        # Should be a JSON string, not a dict
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["result"] == 3.0

    def test_from_real_test_data(self):
        """Wire tools from a real multi_turn_base entry."""
        path = _DATA_DIR / "BFCL_v4_multi_turn_base.json"
        entries = load_jsonl(path)
        entry = entries[0]

        test_id, turns, initial_config, involved_classes = load_multi_turn_entry(entry)
        instances = create_instances(initial_config, involved_classes)
        func_docs = load_func_docs(involved_classes)
        excluded = entry.get("excluded_function", [])
        tools = wire_tools(instances, func_docs, excluded_functions=excluded)

        # Should have tools from both GorillaFileSystem and TwitterAPI
        assert "cat" in tools or "pwd" in tools  # GorillaFileSystem
        assert "post_tweet" in tools or "authenticate_twitter" in tools  # TwitterAPI
        # Excluded function should not be present
        for fn in excluded:
            assert fn not in tools


class TestLoadFuncDocs:
    """Test multi-turn function schema loading."""

    def test_loads_gorilla_filesystem_docs(self):
        """Loads GorillaFileSystem schemas from JSONL."""
        docs = load_func_docs(["GorillaFileSystem"])
        assert len(docs) > 0
        names = {d["name"] for d in docs}
        assert "cat" in names
        assert "mv" in names
        assert "pwd" in names

    def test_loads_multiple_classes(self):
        """Loads combined schemas from multiple classes."""
        docs = load_func_docs(["GorillaFileSystem", "MathAPI"])
        names = {d["name"] for d in docs}
        assert "cat" in names  # GorillaFileSystem
        assert "add" in names  # MathAPI

    def test_all_docs_have_required_fields(self):
        """Every func doc has name, description, parameters."""
        for class_name in CLASS_MODULE_MAP:
            docs = load_func_docs([class_name])
            for doc in docs:
                assert "name" in doc, f"Missing 'name' in {class_name} doc"
                assert "description" in doc, f"Missing 'description' in {class_name} doc: {doc.get('name')}"
                assert "parameters" in doc, f"Missing 'parameters' in {class_name} doc: {doc.get('name')}"


class TestLoadMultiTurnEntry:
    """Test multi-turn entry parsing."""

    def test_parses_base_entry(self):
        """Parses a multi_turn_base entry."""
        path = _DATA_DIR / "BFCL_v4_multi_turn_base.json"
        entries = load_jsonl(path)
        test_id, turns, config, classes = load_multi_turn_entry(entries[0])

        assert test_id == "multi_turn_base_0"
        assert len(turns) >= 2  # Multiple turns
        assert isinstance(config, dict)
        assert len(classes) > 0

    def test_parses_miss_func_entry(self):
        """Parses a multi_turn_miss_func entry (has missed_function field)."""
        path = _DATA_DIR / "BFCL_v4_multi_turn_miss_func.json"
        entries = load_jsonl(path)
        entry = entries[0]
        test_id, turns, config, classes = load_multi_turn_entry(entry)

        assert "miss_func" in test_id
        # The entry should have a missed_function field
        assert "missed_function" in entry

    def test_ground_truth_is_list_of_turns(self):
        """Ground truth is a list[list[str]] — turns of call strings."""
        path = _DATA_DIR / "possible_answer" / "BFCL_v4_multi_turn_base.json"
        entries = load_jsonl(path)
        gt = load_multi_turn_ground_truth(entries[0])

        assert isinstance(gt, list)
        assert len(gt) >= 2
        for turn_calls in gt:
            assert isinstance(turn_calls, list)
            for call_str in turn_calls:
                assert isinstance(call_str, str)
                assert "(" in call_str  # Looks like a function call
