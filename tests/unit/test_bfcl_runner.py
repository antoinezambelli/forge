"""Tests for BFCL runner — pure-logic functions (no LLM needed)."""

import json

import pytest

from forge.core.messages import (
    Message,
    MessageMeta,
    MessageRole,
    MessageType,
    ToolCallInfo,
)
from tests.eval.bfcl.backend_wiring import create_instances
from tests.eval.bfcl.batch_runner import (
    BfclBatchConfig,
    _entry_key,
    _load_completed,
    _result_to_row,
)
from tests.eval.bfcl.runner import (
    build_multi_turn_workflow,
    build_workflow,
    BfclMultiTurnResult,
    extract_calls,
)
from tests.eval.bfcl.schema_adapter import (
    CATEGORY_FILES,
    DATA_DIR,
    load_func_docs,
    load_jsonl,
    load_multi_turn_entry,
    MULTI_TURN_CATEGORIES,
)


class TestBuildWorkflow:
    def test_creates_workflow_with_bfcl_tools_plus_done(self):
        """Workflow has BFCL tools + synthetic done."""
        schemas = [
            {"name": "fn_a", "description": "A", "parameters": {"type": "dict", "properties": {"x": {"type": "string"}}, "required": ["x"]}},
            {"name": "fn_b", "description": "B", "parameters": {"type": "dict", "properties": {"y": {"type": "integer"}}, "required": ["y"]}},
        ]
        wf, name_map, _ = build_workflow(schemas, "test_cat")
        assert "fn_a" in wf.tools
        assert "fn_b" in wf.tools
        assert "done" in wf.tools
        assert wf.terminal_tool == "done"
        assert wf.required_steps == []
        assert name_map == {}  # No dots, no remapping

    def test_dotted_names_sanitised(self):
        """Dotted function names are sanitised in workflow, with name_map."""
        schemas = [
            {"name": "math.factorial", "description": "F", "parameters": {"type": "dict", "properties": {"n": {"type": "integer"}}, "required": ["n"]}},
        ]
        wf, name_map, _ = build_workflow(schemas, "test")
        assert "math_factorial" in wf.tools
        assert "math.factorial" not in wf.tools
        assert name_map == {"math_factorial": "math.factorial"}


class TestExtractCalls:
    def _make_tool_call_msg(self, name, args, call_id):
        """Helper: make a TOOL_CALL message."""
        return Message(
            role=MessageRole.ASSISTANT,
            content="",
            metadata=MessageMeta(MessageType.TOOL_CALL),
            tool_calls=[ToolCallInfo(name=name, args=args, call_id=call_id)],
        )

    def _make_tool_result_msg(self, content, call_id, tool_name):
        """Helper: make a TOOL_RESULT message."""
        return Message(
            role=MessageRole.TOOL,
            content=content,
            metadata=MessageMeta(MessageType.TOOL_RESULT),
            tool_name=tool_name,
            tool_call_id=call_id,
        )

    def test_extracts_successful_call(self):
        """Extracts one successful non-done call."""
        msgs = [
            self._make_tool_call_msg("calc", {"x": 5}, "c1"),
            self._make_tool_result_msg("[stub] calc executed", "c1", "calc"),
            self._make_tool_call_msg("done", {}, "c2"),
            self._make_tool_result_msg("", "c2", "done"),
        ]
        calls = extract_calls(msgs)
        assert calls == [{"calc": {"x": 5}}]

    def test_skips_done_call(self):
        """The done call is not included."""
        msgs = [
            self._make_tool_call_msg("done", {"message": "hi"}, "c1"),
            self._make_tool_result_msg("hi", "c1", "done"),
        ]
        calls = extract_calls(msgs)
        assert calls == []

    def test_skips_error_calls(self):
        """Failed tool calls (ToolError) are not included."""
        msgs = [
            self._make_tool_call_msg("calc", {"x": "bad"}, "c1"),
            self._make_tool_result_msg("[ToolError] TypeError: ...", "c1", "calc"),
            self._make_tool_call_msg("calc", {"x": 5}, "c2"),
            self._make_tool_result_msg("[stub] calc executed", "c2", "calc"),
            self._make_tool_call_msg("done", {}, "c3"),
            self._make_tool_result_msg("", "c3", "done"),
        ]
        calls = extract_calls(msgs)
        # Only the successful retry, not the failed first attempt
        assert calls == [{"calc": {"x": 5}}]

    def test_parallel_calls_extracted(self):
        """Multiple successful calls extracted (parallel category)."""
        msgs = [
            self._make_tool_call_msg("fn", {"a": 1}, "c1"),
            self._make_tool_result_msg("[stub] fn executed", "c1", "fn"),
            self._make_tool_call_msg("fn", {"a": 2}, "c2"),
            self._make_tool_result_msg("[stub] fn executed", "c2", "fn"),
            self._make_tool_call_msg("done", {}, "c3"),
            self._make_tool_result_msg("", "c3", "done"),
        ]
        calls = extract_calls(msgs)
        assert calls == [{"fn": {"a": 1}}, {"fn": {"a": 2}}]

    def test_irrelevance_no_calls(self):
        """Irrelevance: model calls done directly, no tool calls extracted."""
        msgs = [
            self._make_tool_call_msg("done", {"message": "no tools needed"}, "c1"),
            self._make_tool_result_msg("no tools needed", "c1", "done"),
        ]
        calls = extract_calls(msgs)
        assert calls == []

    def test_empty_messages(self):
        """No messages → no calls."""
        assert extract_calls([]) == []


class TestBuildMultiTurnWorkflow:
    """Test multi-turn workflow building."""

    def test_builds_from_real_entry(self):
        """Workflow builds from a real multi_turn_base entry."""
        path = DATA_DIR / "BFCL_v4_multi_turn_base.json"
        entries = load_jsonl(path)
        entry = entries[0]

        _, _, initial_config, involved_classes = load_multi_turn_entry(entry)
        instances = create_instances(initial_config, involved_classes)
        func_docs = load_func_docs(involved_classes)
        excluded = entry.get("excluded_function", [])

        wf, name_map, _ = build_multi_turn_workflow(func_docs, instances, "multi_turn_base",
                                                     excluded_functions=excluded, turn_index=0)
        assert "done" in wf.tools
        assert wf.terminal_tool == "done"
        # Excluded functions should not be present
        for fn in excluded:
            assert fn not in wf.tools

    def test_missed_func_changes_tool_set_per_turn(self):
        """miss_func category produces different tools per turn."""
        path = DATA_DIR / "BFCL_v4_multi_turn_miss_func.json"
        entries = load_jsonl(path)
        entry = entries[0]

        _, _, initial_config, involved_classes = load_multi_turn_entry(entry)
        instances = create_instances(initial_config, involved_classes)
        func_docs = load_func_docs(involved_classes)
        excluded = entry.get("excluded_function", [])
        missed = entry.get("missed_function", {})

        # Find a turn that has missed functions
        missed_turn = int(list(missed.keys())[0])
        missed_names = missed[str(missed_turn)]

        wf_normal, _, _ = build_multi_turn_workflow(
            func_docs, instances, "multi_turn_miss_func",
            excluded_functions=excluded, turn_index=0,
        )
        wf_missed, _, _ = build_multi_turn_workflow(
            func_docs, instances, "multi_turn_miss_func",
            excluded_functions=excluded, missed_functions=missed,
            turn_index=missed_turn,
        )

        # The missed turn should have fewer tools
        for fn in missed_names:
            if fn in wf_normal.tools:
                assert fn not in wf_missed.tools

    def test_done_always_present(self):
        """Every turn's workflow has the done terminal regardless of exclusions."""
        path = DATA_DIR / "BFCL_v4_multi_turn_miss_func.json"
        entries = load_jsonl(path)
        entry = entries[0]

        _, turns, initial_config, involved_classes = load_multi_turn_entry(entry)
        instances = create_instances(initial_config, involved_classes)
        func_docs = load_func_docs(involved_classes)
        missed = entry.get("missed_function", {})

        for turn_idx in range(len(turns)):
            wf, _, _ = build_multi_turn_workflow(
                func_docs, instances, "multi_turn_miss_func",
                missed_functions=missed, turn_index=turn_idx,
            )
            assert "done" in wf.tools
            assert wf.terminal_tool == "done"


class TestLoadJsonl:
    def test_loads_simple_python(self):
        """Can load the real simple_python JSONL file (via schema_adapter)."""
        path = DATA_DIR / "BFCL_v4_simple_python.json"
        entries = load_jsonl(path)
        assert len(entries) > 0
        assert "id" in entries[0]
        assert "question" in entries[0]
        assert "function" in entries[0]


class TestBatchRunnerResume:
    """Test BFCL batch runner resume logic."""

    def test_load_completed_empty_file(self, tmp_path):
        """Empty JSONL returns empty set."""
        p = tmp_path / "empty.jsonl"
        p.touch()
        assert _load_completed(p, "reforged") == set()

    def test_load_completed_nonexistent_file(self, tmp_path):
        """Missing file returns empty set."""
        p = tmp_path / "nope.jsonl"
        assert _load_completed(p, "reforged") == set()

    def test_load_completed_counts_entries(self, tmp_path):
        """Existing entries appear in completed set."""
        p = tmp_path / "results.jsonl"
        row = {
            "model": "test-model", "backend": "ollama", "mode": "native",
            "ablation": "reforged", "tool_choice": "auto",
            "category": "simple_python",
            "test_id": "simple_python_0", "valid": True,
            "completed": True, "error_type": "", "errors": [],
            "elapsed_s": 1.0, "iterations": 2,
        }
        p.write_text(json.dumps(row) + "\n")
        completed = _load_completed(p, "reforged")
        key = _entry_key("test-model", "ollama", "native", "reforged",
                         "auto", "simple_python", "simple_python_0")
        assert key in completed

    def test_entry_key_format(self):
        """Entry key is pipe-delimited."""
        key = _entry_key("m", "b", "n", "reforged", "auto", "cat", "id")
        assert key == "m|b|n|reforged|auto|cat|id"

    def test_result_to_row_fields(self):
        """Row dict contains all expected fields."""
        config = BfclBatchConfig(
            model="m", backend="ollama", mode="native",
        )
        row = _result_to_row(
            config, "reforged", "simple_python", "test_0",
            valid=True, error_type="", errors=[],
            elapsed_s=1.234, iterations=2, completed=True,
        )
        assert row["model"] == "m"
        assert row["ablation"] == "reforged"
        assert row["category"] == "simple_python"
        assert row["valid"] is True
        assert row["elapsed_s"] == 1.23  # rounded


class TestNewCategories:
    """Verify java/javascript categories load correctly."""

    def test_simple_java_in_category_files(self):
        """simple_java is registered."""
        assert "simple_java" in CATEGORY_FILES

    def test_simple_javascript_in_category_files(self):
        """simple_javascript is registered."""
        assert "simple_javascript" in CATEGORY_FILES

    def test_simple_java_loads(self):
        """simple_java data file loads and has entries."""
        path = DATA_DIR / CATEGORY_FILES["simple_java"]
        entries = load_jsonl(path)
        assert len(entries) == 100

    def test_simple_javascript_loads(self):
        """simple_javascript data file loads and has entries."""
        path = DATA_DIR / CATEGORY_FILES["simple_javascript"]
        entries = load_jsonl(path)
        assert len(entries) == 50

    def test_simple_java_ground_truth_exists(self):
        """simple_java has matching ground truth file."""
        gt_path = DATA_DIR / "possible_answer" / CATEGORY_FILES["simple_java"]
        entries = load_jsonl(gt_path)
        assert len(entries) == 100

    def test_simple_javascript_ground_truth_exists(self):
        """simple_javascript has matching ground truth file."""
        gt_path = DATA_DIR / "possible_answer" / CATEGORY_FILES["simple_javascript"]
        entries = load_jsonl(gt_path)
        assert len(entries) == 50


class TestBatchRunnerCategories:
    """Verify batch runner category definitions are consistent."""

    def test_all_categories_have_data_files(self):
        """Every category in ALL_CATEGORIES has a data file."""
        from tests.eval.bfcl.batch_runner import ALL_CATEGORIES
        for cat in ALL_CATEGORIES:
            if cat in CATEGORY_FILES:
                path = DATA_DIR / CATEGORY_FILES[cat]
            elif cat in MULTI_TURN_CATEGORIES:
                path = DATA_DIR / MULTI_TURN_CATEGORIES[cat]
            else:
                pytest.fail(f"Category {cat!r} not in any file map")
            assert path.exists(), f"Data file missing: {path}"

    def test_all_categories_have_ground_truth(self):
        """Every non-irrelevance category has a matching possible_answer file."""
        from tests.eval.bfcl.batch_runner import ALL_CATEGORIES
        for cat in ALL_CATEGORIES:
            if cat == "irrelevance":
                continue  # scored by "no tools called", no answer file
            if cat in CATEGORY_FILES:
                filename = CATEGORY_FILES[cat]
            else:
                filename = MULTI_TURN_CATEGORIES[cat]
            gt_path = DATA_DIR / "possible_answer" / filename
            assert gt_path.exists(), f"Ground truth missing: {gt_path}"

    def test_config_sets_have_expected_sizes(self):
        """Config sets contain the right number of configs."""
        from tests.eval.bfcl.batch_runner import CONFIG_SETS
        assert len(CONFIG_SETS["ollama"]) == 11     # 11 models
        assert len(CONFIG_SETS["llamaserver"]) == 28  # 14 models x 2 modes
        assert len(CONFIG_SETS["llamafile"]) == 5     # 5 models x 1 mode
        assert len(CONFIG_SETS["anthropic"]) == 3     # 3 models
        assert len(CONFIG_SETS["all"]) == 44          # ollama + llamaserver + llamafile

    def test_category_count(self):
        """11 total categories: 7 single-turn + 4 multi-turn."""
        from tests.eval.bfcl.batch_runner import (
            ALL_CATEGORIES, SINGLE_TURN_CATEGORIES, MULTI_TURN_CATS,
        )
        assert len(SINGLE_TURN_CATEGORIES) == 7
        assert len(MULTI_TURN_CATS) == 4
        assert len(ALL_CATEGORIES) == 11
