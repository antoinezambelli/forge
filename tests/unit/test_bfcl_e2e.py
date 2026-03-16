"""End-to-end test of the BFCL single-turn and multi-turn pipelines (no LLM needed)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from tests.eval.bfcl.runner import (
    build_workflow, extract_calls, BfclRunResult,
    build_multi_turn_workflow, BfclMultiTurnResult,
)
from tests.eval.bfcl.scorer import (
    score_entry, ScoreResult,
    execute_ground_truth, state_checker, score_multi_turn_entry,
    _parse_call_string,
)
from tests.eval.bfcl.schema_adapter import (
    load_test_entry, load_ground_truth, load_jsonl,
    load_func_docs, load_multi_turn_entry, load_multi_turn_ground_truth,
    MULTI_TURN_CATEGORIES,
)
from tests.eval.bfcl.backend_wiring import create_instances, wire_tools


_DATA_DIR = Path(__file__).parent.parent / "eval" / "bfcl" / "data"


class TestEndToEnd:
    """Test the full pipeline: load → build → extract → score."""

    def test_correct_simple_python_0(self):
        """Pipeline produces valid=True for a correct answer."""
        # Load real test data
        with open(_DATA_DIR / "BFCL_v4_simple_python.json") as f:
            test_entry = json.loads(f.readline())
        with open(_DATA_DIR / "possible_answer" / "BFCL_v4_simple_python.json") as f:
            gt_entry = json.loads(f.readline())

        test_id, messages, function_schemas = load_test_entry(test_entry)
        ground_truth = load_ground_truth(gt_entry)

        # Build workflow (verifies schema adapter + executor wiring)
        workflow, name_map, _ = build_workflow(function_schemas, "simple_python")
        assert "calculate_triangle_area" in workflow.tools
        assert "done" in workflow.tools

        # Simulate what the runner would extract from a correct model response
        simulated_messages = [
            Message(
                role=MessageRole.ASSISTANT,
                content="",
                metadata=MessageMeta(MessageType.TOOL_CALL),
                tool_calls=[ToolCallInfo(
                    name="calculate_triangle_area",
                    args={"base": 10, "height": 5},
                    call_id="c1",
                )],
            ),
            Message(
                role=MessageRole.TOOL,
                content="[stub] calculate_triangle_area executed",
                metadata=MessageMeta(MessageType.TOOL_RESULT),
                tool_name="calculate_triangle_area",
                tool_call_id="c1",
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="",
                metadata=MessageMeta(MessageType.TOOL_CALL),
                tool_calls=[ToolCallInfo(name="done", args={}, call_id="c2")],
            ),
            Message(
                role=MessageRole.TOOL,
                content="",
                metadata=MessageMeta(MessageType.TOOL_RESULT),
                tool_name="done",
                tool_call_id="c2",
            ),
        ]

        # Extract calls
        calls = extract_calls(simulated_messages)
        assert len(calls) == 1
        assert "calculate_triangle_area" in calls[0]
        assert calls[0]["calculate_triangle_area"]["base"] == 10

        # Score
        result = score_entry(
            test_id=test_id,
            func_descriptions=function_schemas,
            model_output=calls,
            ground_truth=ground_truth,
            category="simple_python",
        )
        assert result.valid, f"Expected valid, got: {result.errors}"

    def test_wrong_answer_scores_invalid(self):
        """Pipeline produces valid=False for a wrong answer."""
        with open(_DATA_DIR / "BFCL_v4_simple_python.json") as f:
            test_entry = json.loads(f.readline())
        with open(_DATA_DIR / "possible_answer" / "BFCL_v4_simple_python.json") as f:
            gt_entry = json.loads(f.readline())

        test_id, _, function_schemas = load_test_entry(test_entry)
        ground_truth = load_ground_truth(gt_entry)

        # Wrong values
        wrong_calls = [{"calculate_triangle_area": {"base": 999, "height": 999}}]

        result = score_entry(
            test_id=test_id,
            func_descriptions=function_schemas,
            model_output=wrong_calls,
            ground_truth=ground_truth,
            category="simple_python",
        )
        assert not result.valid

    def test_irrelevance_pipeline(self):
        """Irrelevance: zero calls → valid."""
        result = score_entry("irr_0", [], [], [], "irrelevance")
        assert result.valid

    def test_executor_type_error_triggers(self):
        """Stub executor raises ValidationError on wrong type (forge will nudge)."""
        with open(_DATA_DIR / "BFCL_v4_simple_python.json") as f:
            test_entry = json.loads(f.readline())

        _, _, function_schemas = load_test_entry(test_entry)
        workflow, _, _ = build_workflow(function_schemas, "simple_python")

        # Call with wrong type — Pydantic strict mode rejects string→int
        fn = workflow.tools["calculate_triangle_area"].callable
        with pytest.raises(ValidationError):
            fn(base="not_an_int", height="also_wrong")

    def test_executor_missing_required_triggers(self):
        """Stub executor raises ValidationError on missing required param."""
        with open(_DATA_DIR / "BFCL_v4_simple_python.json") as f:
            test_entry = json.loads(f.readline())

        _, _, function_schemas = load_test_entry(test_entry)
        workflow, _, _ = build_workflow(function_schemas, "simple_python")

        fn = workflow.tools["calculate_triangle_area"].callable
        with pytest.raises(ValidationError, match="Field required"):
            fn()  # No args at all


class TestBuildWorkflowFromAllCategories:
    """Verify that every single-turn category can build workflows."""

    @pytest.mark.parametrize("filename", [
        "BFCL_v4_simple_python.json",
        "BFCL_v4_multiple.json",
        "BFCL_v4_parallel.json",
        "BFCL_v4_irrelevance.json",
    ])
    def test_builds_without_error(self, filename):
        """Every entry in the category builds a valid workflow."""
        path = _DATA_DIR / filename
        category = filename.replace("BFCL_v4_", "").replace(".json", "")

        with open(path) as f:
            for i, line in enumerate(f):
                if i >= 5:  # Spot-check first 5 entries
                    break
                entry = json.loads(line)
                _, _, schemas = load_test_entry(entry)
                wf, _, _ = build_workflow(schemas, category)
                assert "done" in wf.tools
                assert wf.terminal_tool == "done"


# ---------------------------------------------------------------------------
# Multi-turn end-to-end tests
# ---------------------------------------------------------------------------


class TestMultiTurnEndToEnd:
    """Test the multi-turn pipeline: load -> instantiate -> execute GT -> score."""

    def test_ground_truth_execution_produces_matching_state(self):
        """Two instances executing identical calls produce identical state."""
        path = _DATA_DIR / "BFCL_v4_multi_turn_base.json"
        gt_path = _DATA_DIR / "possible_answer" / "BFCL_v4_multi_turn_base.json"

        entries = load_jsonl(path)
        gt_entries = load_jsonl(gt_path)
        entry, gt_entry = entries[0], gt_entries[0]

        _, _, initial_config, involved_classes = load_multi_turn_entry(entry)
        ground_truth = load_multi_turn_ground_truth(gt_entry)

        # Create two sets of instances from the same config
        instances_a = create_instances(initial_config, involved_classes)
        instances_b = create_instances(initial_config, involved_classes)

        # Execute ground truth on both
        for turn_calls in ground_truth:
            if turn_calls:
                execute_ground_truth(turn_calls, instances_a)
                execute_ground_truth(turn_calls, instances_b)

        # State should match
        result = state_checker(instances_a, instances_b)
        assert result.valid, f"State mismatch: {result.errors}"

    def test_ground_truth_mutates_state(self):
        """Ground truth calls actually change backend state."""
        config = {
            "GorillaFileSystem": {
                "root": {
                    "workspace": {
                        "type": "directory",
                        "contents": {
                            "test.txt": {"type": "file", "content": "hello"},
                        },
                    }
                }
            }
        }
        instances = create_instances(config, ["GorillaFileSystem"])

        # Before: file exists
        result = instances["GorillaFileSystem"].ls()
        assert "test.txt" in result["current_directory_content"]

        # Execute: create a new file
        execute_ground_truth(["touch(file_name='new.txt')"], instances)

        # After: new file exists
        result = instances["GorillaFileSystem"].ls()
        assert "new.txt" in result["current_directory_content"]

    def test_diverged_state_detected(self):
        """Score detects when model state diverges from ground truth."""
        config = {
            "GorillaFileSystem": {
                "root": {
                    "workspace": {"type": "directory", "contents": {}}
                }
            }
        }
        model_instances = create_instances(config, ["GorillaFileSystem"])
        gt_instances = create_instances(config, ["GorillaFileSystem"])

        # Ground truth creates a file
        execute_ground_truth(["touch(file_name='expected.txt')"], gt_instances)

        # Model creates a different file
        model_instances["GorillaFileSystem"].touch(file_name="wrong.txt")

        # State check should fail
        result = state_checker(model_instances, gt_instances)
        assert not result.valid

    def test_simulated_correct_run_final_state_matches(self):
        """Full pipeline: load -> execute GT -> final state matches.

        Executes ground truth on two independent instance sets and verifies
        final state matches. This tests the complete pipeline without
        score_multi_turn_entry (which checks per-turn state, requiring
        incremental mutation from a real runner).
        """
        path = _DATA_DIR / "BFCL_v4_multi_turn_base.json"
        gt_path = _DATA_DIR / "possible_answer" / "BFCL_v4_multi_turn_base.json"

        entries = load_jsonl(path)
        gt_entries = load_jsonl(gt_path)
        entry, gt_entry = entries[0], gt_entries[0]

        _, _, initial_config, involved_classes = load_multi_turn_entry(entry)
        ground_truth = load_multi_turn_ground_truth(gt_entry)

        # Execute ground truth on two independent instance sets
        model_instances = create_instances(initial_config, involved_classes)
        gt_instances = create_instances(initial_config, involved_classes)

        per_turn_calls: list[list[dict]] = []
        for turn_calls in ground_truth:
            if turn_calls:
                execute_ground_truth(turn_calls, model_instances)
                execute_ground_truth(turn_calls, gt_instances)
            # Parse calls to verify _parse_call_string works on real data
            calls = []
            for call_str in turn_calls:
                func_name, kwargs = _parse_call_string(call_str)
                calls.append({func_name: kwargs})
            per_turn_calls.append(calls)

        # Final state should match
        state = state_checker(model_instances, gt_instances)
        assert state.valid, f"Final state mismatch: {state.errors}"

        # Verify per-turn calls are non-empty for non-empty GT turns
        for i, (gt_turn, model_turn) in enumerate(zip(ground_truth, per_turn_calls)):
            if gt_turn:
                assert model_turn, f"Turn {i}: expected calls but got none"


class TestBuildMultiTurnWorkflowFromAllCategories:
    """Verify that every multi-turn category can build workflows."""

    @pytest.mark.parametrize("category", [
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
    ])
    def test_builds_without_error(self, category):
        """First entry in each category builds a valid workflow."""
        filename = MULTI_TURN_CATEGORIES[category]
        path = _DATA_DIR / filename
        entries = load_jsonl(path)

        entry = entries[0]
        _, _, initial_config, involved_classes = load_multi_turn_entry(entry)
        instances = create_instances(initial_config, involved_classes)
        func_docs = load_func_docs(involved_classes)
        excluded = entry.get("excluded_function", [])

        wf, _, _ = build_multi_turn_workflow(
            func_docs, instances, category,
            excluded_functions=excluded,
            turn_index=0,
        )
        assert "done" in wf.tools
        assert wf.terminal_tool == "done"
