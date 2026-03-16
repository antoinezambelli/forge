"""Tests for BFCL AST scorer."""

import json
from pathlib import Path

from tests.eval.bfcl.scorer import (
    ScoreResult,
    _check_dict,
    _check_list,
    _check_string,
    _match_element,
    _score_irrelevance,
    _score_parallel,
    _standardize_string,
    execute_ground_truth,
    irrelevance_checker,
    response_checker,
    score_entry,
    simple_function_checker,
    state_checker,
)
from tests.eval.bfcl.backend_wiring import create_instances
from tests.eval.bfcl.runner import BfclMultiTurnResult


class TestStandardizeString:
    def test_basic_normalization(self):
        assert _standardize_string("April 1, 2024") == _standardize_string("April 1,2024")

    def test_case_insensitive(self):
        assert _standardize_string("Paris") == _standardize_string("paris")

    def test_quotes_normalized(self):
        assert _standardize_string("it's") == _standardize_string('it"s')


class TestSimpleFunctionChecker:
    def _make_desc(self, name, properties, required):
        return {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def test_correct_call(self):
        """Correct function name, params, and values -> valid."""
        desc = self._make_desc(
            "calc",
            {"base": {"type": "integer"}, "height": {"type": "integer"}},
            ["base", "height"],
        )
        output = {"calc": {"base": 10, "height": 5}}
        gt = {"calc": {"base": [10], "height": [5]}}
        result = simple_function_checker(desc, output, gt)
        assert result.valid

    def test_wrong_function_name(self):
        desc = self._make_desc("calc", {"x": {"type": "integer"}}, ["x"])
        output = {"wrong_func": {"x": 5}}
        gt = {"calc": {"x": [5]}}
        result = simple_function_checker(desc, output, gt)
        assert not result.valid
        assert "wrong_func" in result.error_type or "wrong" in result.errors[0].lower()

    def test_missing_required(self):
        desc = self._make_desc(
            "calc",
            {"x": {"type": "integer"}, "y": {"type": "integer"}},
            ["x", "y"],
        )
        output = {"calc": {"x": 5}}
        gt = {"calc": {"x": [5], "y": [10]}}
        result = simple_function_checker(desc, output, gt)
        assert not result.valid

    def test_wrong_type(self):
        desc = self._make_desc("calc", {"x": {"type": "integer"}}, ["x"])
        output = {"calc": {"x": "five"}}
        gt = {"calc": {"x": [5]}}
        result = simple_function_checker(desc, output, gt)
        assert not result.valid

    def test_wrong_value(self):
        desc = self._make_desc("calc", {"x": {"type": "integer"}}, ["x"])
        output = {"calc": {"x": 99}}
        gt = {"calc": {"x": [5]}}
        result = simple_function_checker(desc, output, gt)
        assert not result.valid

    def test_optional_param_omittable(self):
        """Optional param (has '' in acceptable) can be omitted."""
        desc = self._make_desc(
            "fn",
            {"x": {"type": "integer"}, "y": {"type": "string"}},
            ["x"],
        )
        output = {"fn": {"x": 5}}
        gt = {"fn": {"x": [5], "y": ["default", ""]}}
        result = simple_function_checker(desc, output, gt)
        assert result.valid

    def test_string_case_insensitive(self):
        """String comparison is case-insensitive."""
        desc = self._make_desc("fn", {"city": {"type": "string"}}, ["city"])
        output = {"fn": {"city": "PARIS"}}
        gt = {"fn": {"city": ["Paris", "paris"]}}
        result = simple_function_checker(desc, output, gt)
        assert result.valid

    def test_int_to_float_conversion(self):
        """Int auto-converts to float when expected type is float."""
        desc = self._make_desc("fn", {"x": {"type": "float"}}, ["x"])
        output = {"fn": {"x": 5}}
        gt = {"fn": {"x": [5.0]}}
        result = simple_function_checker(desc, output, gt)
        assert result.valid

    def test_bfcl_boolean_type(self):
        """BFCL 'Boolean' type accepted for Python bool values."""
        desc = self._make_desc("fn", {"flag": {"type": "Boolean"}}, ["flag"])
        output = {"fn": {"flag": True}}
        gt = {"fn": {"flag": [True]}}
        result = simple_function_checker(desc, output, gt)
        assert result.valid

    def test_bfcl_string_type(self):
        """BFCL 'String' type accepted for Python str values."""
        desc = self._make_desc("fn", {"name": {"type": "String"}}, ["name"])
        output = {"fn": {"name": "hello"}}
        gt = {"fn": {"name": ["hello"]}}
        result = simple_function_checker(desc, output, gt)
        assert result.valid

    def test_bfcl_arraylist_type(self):
        """BFCL 'ArrayList' type accepted for Python list values."""
        desc = self._make_desc("fn", {"items": {"type": "ArrayList"}}, ["items"])
        output = {"fn": {"items": [1, 2]}}
        gt = {"fn": {"items": [[1, 2]]}}
        result = simple_function_checker(desc, output, gt)
        assert result.valid


class TestCheckListNestedDicts:
    """Test _check_list with nested dicts (e.g. database.query conditions)."""

    def test_nested_dicts_match_gt_convention(self):
        """Model dicts with plain values match GT dicts with list-wrapped values."""
        value = [
            {"field": "age", "operation": ">", "value": "25"},
            {"field": "job", "operation": "=", "value": "engineer"},
        ]
        acceptable = [[
            {"field": ["age"], "operation": [">"], "value": ["25"]},
            {"field": ["job"], "operation": ["="], "value": ["engineer"]},
        ]]
        result = _check_list("conditions", value, acceptable)
        assert result is None  # Match

    def test_nested_dicts_wrong_value_rejected(self):
        """Nested dict with wrong value is rejected."""
        value = [{"field": "age", "operation": ">", "value": "30"}]
        acceptable = [[{"field": ["age"], "operation": [">"], "value": ["25"]}]]
        result = _check_list("conditions", value, acceptable)
        assert result is not None
        assert "value_error:list" in result.error_type

    def test_nested_dicts_missing_key_rejected(self):
        """Nested dict missing a required key is rejected."""
        value = [{"field": "age", "operation": ">"}]  # missing "value"
        acceptable = [[{"field": ["age"], "operation": [">"], "value": ["25"]}]]
        result = _check_list("conditions", value, acceptable)
        assert result is not None

    def test_nested_dicts_string_normalization(self):
        """String normalization applies inside nested dicts."""
        value = [{"field": "Age", "operation": ">", "value": "25"}]
        acceptable = [[{"field": ["age"], "operation": [">"], "value": ["25"]}]]
        result = _check_list("conditions", value, acceptable)
        assert result is None  # Match — case insensitive

    def test_plain_string_list_still_works(self):
        """Plain string lists still work after the nested dict fix."""
        value = ["red", "blue"]
        acceptable = [["Red", "Blue"]]
        result = _check_list("colors", value, acceptable)
        assert result is None

    def test_length_mismatch_rejected(self):
        """Lists of different lengths are rejected."""
        value = [{"field": "age", "operation": ">", "value": "25"}]
        acceptable = [[
            {"field": ["age"], "operation": [">"], "value": ["25"]},
            {"field": ["job"], "operation": ["="], "value": ["engineer"]},
        ]]
        result = _check_list("conditions", value, acceptable)
        assert result is not None


class TestScoreIrrelevance:
    def test_no_calls_is_valid(self):
        result = _score_irrelevance("test_0", [])
        assert result.valid

    def test_calls_present_is_invalid(self):
        result = _score_irrelevance("test_0", [{"some_func": {"x": 1}}])
        assert not result.valid


class TestScoreParallel:
    def test_order_independent(self):
        """Parallel matching doesn't depend on order."""
        descs = [
            {
                "name": "fn",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            },
        ]
        output = [{"fn": {"x": 2}}, {"fn": {"x": 1}}]  # Reversed order
        gt = [{"fn": {"x": [1]}}, {"fn": {"x": [2]}}]
        result = _score_parallel("test_0", descs, output, gt)
        assert result.valid

    def test_wrong_count(self):
        descs = [
            {
                "name": "fn",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            },
        ]
        output = [{"fn": {"x": 1}}]
        gt = [{"fn": {"x": [1]}}, {"fn": {"x": [2]}}]
        result = _score_parallel("test_0", descs, output, gt)
        assert not result.valid


class TestScoreEntry:
    def test_routes_to_simple(self):
        descs = [
            {
                "name": "fn",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            },
        ]
        output = [{"fn": {"x": 5}}]
        gt = [{"fn": {"x": [5]}}]
        result = score_entry("test_0", descs, output, gt, "simple_python")
        assert result.valid

    def test_routes_to_irrelevance(self):
        result = score_entry("test_0", [], [], [], "irrelevance")
        assert result.valid

    def test_routes_to_parallel(self):
        descs = [
            {
                "name": "fn",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            },
        ]
        output = [{"fn": {"x": 1}}]
        gt = [{"fn": {"x": [1]}}]
        result = score_entry("test_0", descs, output, gt, "parallel")
        assert result.valid


class TestRealGroundTruth:
    def test_first_simple_python_scores_correctly(self):
        """Score a hand-crafted correct answer against real ground truth."""
        data_dir = Path(__file__).parent.parent / "eval" / "bfcl" / "data"

        with open(data_dir / "BFCL_v4_simple_python.json") as f:
            test_entry = json.loads(f.readline())
        with open(data_dir / "possible_answer" / "BFCL_v4_simple_python.json") as f:
            gt_entry = json.loads(f.readline())

        # simple_python_0: calculate_triangle_area(base=10, height=5)
        correct_output = [{"calculate_triangle_area": {"base": 10, "height": 5}}]

        result = score_entry(
            test_id=test_entry["id"],
            func_descriptions=test_entry["function"],
            model_output=correct_output,
            ground_truth=gt_entry["ground_truth"],
            category="simple_python",
        )
        assert result.valid, f"Expected valid, got errors: {result.errors}"


# ---------------------------------------------------------------------------
# Multi-turn scorer tests
# ---------------------------------------------------------------------------


class TestExecuteGroundTruth:
    """Test ground truth call string execution."""

    def test_simple_method_call(self):
        """Executes a simple ground truth call string."""
        instances = create_instances({}, ["MathAPI"])
        results = execute_ground_truth(["add(a=2.0, b=3.0)"], instances)
        assert len(results) == 1
        assert '"result": 5.0' in results[0]

    def test_multiple_calls(self):
        """Executes multiple ground truth calls in sequence."""
        instances = create_instances({}, ["MathAPI"])
        results = execute_ground_truth([
            "add(a=1.0, b=2.0)",
            "multiply(a=3.0, b=4.0)",
        ], instances)
        assert len(results) == 2

    def test_stateful_calls(self):
        """Ground truth calls mutate state across calls."""
        config = {
            "GorillaFileSystem": {
                "root": {
                    "workspace": {
                        "type": "directory",
                        "contents": {
                            "docs": {"type": "directory", "contents": {}},
                        },
                    }
                }
            }
        }
        instances = create_instances(config, ["GorillaFileSystem"])
        # Create a file, then verify it exists
        execute_ground_truth([
            "touch(file_name='test.txt')",
        ], instances)
        results = execute_ground_truth([
            "ls()",
        ], instances)
        assert "test.txt" in results[0]

    def test_positional_args(self):
        """Handles positional args in call strings like sort('file.txt')."""
        config = {
            "GorillaFileSystem": {
                "root": {
                    "workspace": {
                        "type": "directory",
                        "contents": {
                            "a.txt": {"type": "file", "content": "z\na\nm"},
                        },
                    }
                }
            }
        }
        instances = create_instances(config, ["GorillaFileSystem"])
        results = execute_ground_truth(["sort('a.txt')"], instances)
        assert len(results) == 1
        assert "Error" not in results[0]

    def test_unknown_function_returns_error(self):
        """Unknown function names produce error strings, not exceptions."""
        instances = create_instances({}, ["MathAPI"])
        results = execute_ground_truth(["nonexistent_func(x=1)"], instances)
        assert "Error" in results[0] or "unknown" in results[0].lower()


class TestStateChecker:
    """Test state comparison between model and ground truth instances."""

    def test_identical_state_passes(self):
        """Two identically-configured instances pass state check."""
        config = {
            "GorillaFileSystem": {
                "root": {
                    "workspace": {"type": "directory", "contents": {}}
                }
            }
        }
        model = create_instances(config, ["GorillaFileSystem"])
        gt = create_instances(config, ["GorillaFileSystem"])
        result = state_checker(model, gt)
        assert result.valid

    def test_diverged_state_fails(self):
        """Instances with different state fail the check."""
        config = {
            "GorillaFileSystem": {
                "root": {
                    "workspace": {"type": "directory", "contents": {}}
                }
            }
        }
        model = create_instances(config, ["GorillaFileSystem"])
        gt = create_instances(config, ["GorillaFileSystem"])

        # Mutate model state
        model["GorillaFileSystem"].touch(file_name="extra.txt")

        result = state_checker(model, gt)
        assert not result.valid
        assert "state_mismatch" in result.error_type


class TestResponseChecker:
    """Test unordered response containment check."""

    def test_all_present_passes(self):
        """All ground truth results present in model results."""
        model_results = ['{"result": 5.0}', '{"result": 12.0}', '{"result": 3.0}']
        gt_results = ['{"result": 5.0}', '{"result": 12.0}']
        result = response_checker(model_results, gt_results, turn_index=0)
        assert result.valid

    def test_missing_result_fails(self):
        """Missing ground truth result causes failure."""
        model_results = ['{"result": 5.0}']
        gt_results = ['{"result": 5.0}', '{"result": 999.0}']
        result = response_checker(model_results, gt_results, turn_index=0)
        assert not result.valid

    def test_order_independent(self):
        """Results can be in any order."""
        model_results = ['{"result": 3.0}', '{"result": 5.0}']
        gt_results = ['{"result": 5.0}', '{"result": 3.0}']
        result = response_checker(model_results, gt_results, turn_index=0)
        assert result.valid

    def test_duplicates_handled(self):
        """Duplicate results require duplicate matches."""
        model_results = ['{"result": 5.0}']
        gt_results = ['{"result": 5.0}', '{"result": 5.0}']
        result = response_checker(model_results, gt_results, turn_index=0)
        assert not result.valid  # Only one match for two identical GT results


class TestIrrelevanceChecker:
    """Test irrelevance check for empty ground truth turns."""

    def test_no_calls_passes(self):
        """Zero model calls on empty GT turn passes."""
        result = irrelevance_checker([], turn_index=0)
        assert result.valid

    def test_calls_present_fails(self):
        """Model calls on empty GT turn fails."""
        result = irrelevance_checker(
            [{"some_func": {"arg": "val"}}],
            turn_index=2,
        )
        assert not result.valid
        assert "irrelevance" in result.error_type
