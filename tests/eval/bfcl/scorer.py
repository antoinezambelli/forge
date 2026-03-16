"""BFCL AST scorer — Python-only subset of BFCL's AST matching logic."""

from __future__ import annotations

import ast
import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests.eval.bfcl.runner import BfclMultiTurnResult, BfclRunResult
from tests.eval.bfcl.schema_adapter import _BFCL_TYPE_MAP


@dataclass
class ScoreResult:
    """Result of scoring one BFCL test case."""

    test_id: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    error_type: str = ""


# ---------------------------------------------------------------------------
# String normalization (matching BFCL's logic)
# ---------------------------------------------------------------------------

def _standardize_string(s: str) -> str:
    """Normalize a string for comparison.

    Removes spaces, punctuation (,./-_*^), lowercases, and converts
    single quotes to double quotes. Matches BFCL's standardize_string().
    """
    return re.sub(r"[ ,.\-/_*^]", "", s).lower().replace("'", '"')


# ---------------------------------------------------------------------------
# Value comparison helpers
# ---------------------------------------------------------------------------

def _check_string(param: str, value: str, acceptable: list) -> ScoreResult | None:
    """Check a string value against acceptable values (case-insensitive)."""
    normalized = _standardize_string(value)
    for ans in acceptable:
        if isinstance(ans, str) and _standardize_string(ans) == normalized:
            return None  # Match found
    return ScoreResult(
        "", False,
        [f"Parameter '{param}': got {value!r}, expected one of {acceptable}"],
        "value_error:string",
    )


def _match_element(model_val: Any, gt_val: Any) -> bool:
    """Check if a single model value matches a ground truth value.

    Handles the BFCL ground truth convention where dict values are wrapped
    in lists of acceptable values, e.g. {"field": ["age"]} means "age" is
    the one acceptable value for "field".
    """
    if isinstance(model_val, str) and isinstance(gt_val, str):
        return _standardize_string(model_val) == _standardize_string(gt_val)
    if isinstance(model_val, dict) and isinstance(gt_val, dict):
        # gt_val uses {key: [acceptable_values]} convention
        for k, v in model_val.items():
            if k not in gt_val:
                return False
            possible = gt_val[k] if isinstance(gt_val[k], list) else [gt_val[k]]
            if isinstance(v, str):
                if not any(isinstance(p, str) and _standardize_string(p) == _standardize_string(v) for p in possible):
                    return False
            elif v not in possible:
                return False
        for k, v in gt_val.items():
            if k not in model_val:
                if not (isinstance(v, list) and "" in v):
                    return False
        return True
    return model_val == gt_val


def _check_list(param: str, value: list, acceptable: list) -> ScoreResult | None:
    """Check a list value against acceptable values."""
    for ans in acceptable:
        if not isinstance(ans, list):
            continue
        if len(value) != len(ans):
            continue
        if all(_match_element(v, a) for v, a in zip(value, ans)):
            return None
    return ScoreResult(
        "", False,
        [f"Parameter '{param}': got {value!r}, expected one of {acceptable}"],
        "value_error:list",
    )


def _check_dict(param: str, value: dict, acceptable: list) -> ScoreResult | None:
    """Check a dict value against acceptable values."""
    for ans in acceptable:
        if ans == "":
            continue
        if not isinstance(ans, dict):
            continue
        # Check all keys match and values match (with string normalization)
        match = True
        for k, v in value.items():
            if k not in ans:
                match = False
                break
            possible_vals = ans[k] if isinstance(ans[k], list) else [ans[k]]
            if isinstance(v, str):
                if not any(
                    isinstance(pv, str) and _standardize_string(pv) == _standardize_string(v)
                    for pv in possible_vals
                ):
                    match = False
                    break
            elif v not in possible_vals:
                match = False
                break
        # Check no required keys missing
        if match:
            for k, v in ans.items():
                if k not in value:
                    if not (isinstance(v, list) and "" in v):
                        match = False
                        break
        if match:
            return None
    return ScoreResult(
        "", False,
        [f"Parameter '{param}': dict mismatch — got {value!r}, expected one of {acceptable!r}"],
        "value_error:dict",
    )


# ---------------------------------------------------------------------------
# Core single-function scorer
# ---------------------------------------------------------------------------

# Python JSON Schema type → Python type
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "float": float,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "object": dict,
}


def simple_function_checker(
    func_description: dict,
    model_output: dict,
    ground_truth: dict,
) -> ScoreResult:
    """Score a single function call against ground truth.

    Args:
        func_description: BFCL function schema dict
        model_output: {"func_name": {"param": value, ...}}
        ground_truth: {"func_name": {"param": [acceptable_values], ...}}

    Returns:
        ScoreResult with valid=True/False
    """
    func_name = func_description["name"]
    expected_answers = list(ground_truth.values())[0]  # {param: [values]}
    param_details = func_description["parameters"]["properties"]
    required_params = func_description["parameters"].get("required", [])

    # 1. Check function name
    if func_name not in model_output:
        return ScoreResult(
            "", False,
            [f"Wrong function: expected {func_name!r}"],
            "wrong_func_name",
        )

    model_params = model_output[func_name]

    # 2. Check required params present
    for param in required_params:
        if param not in model_params:
            return ScoreResult(
                "", False,
                [f"Missing required parameter: {param!r}"],
                "missing_required",
            )

    # 3. Check each provided param
    for param, value in model_params.items():
        if param not in param_details or param not in expected_answers:
            return ScoreResult(
                "", False,
                [f"Unexpected parameter: {param!r}"],
                "unexpected_param",
            )

        expected_type_str = param_details[param].get("type", "string")
        # Normalize BFCL language-specific types (Boolean → boolean, etc.)
        expected_type_str = _BFCL_TYPE_MAP.get(expected_type_str, expected_type_str)

        # Allow int → float auto-conversion
        if expected_type_str == "float" and isinstance(value, int):
            value = float(value)

        # Type check
        expected_type = _TYPE_MAP.get(expected_type_str, str)
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                return ScoreResult(
                    "", False,
                    [f"Parameter '{param}': expected {expected_type_str}, got {type(value).__name__}"],
                    "type_error",
                )
        elif not isinstance(value, expected_type):
            return ScoreResult(
                "", False,
                [f"Parameter '{param}': expected {expected_type_str}, got {type(value).__name__}"],
                "type_error",
            )

        # Value check (type-specific)
        if isinstance(value, str):
            err = _check_string(param, value, expected_answers[param])
            if err:
                return err
        elif isinstance(value, list):
            err = _check_list(param, value, expected_answers[param])
            if err:
                return err
        elif isinstance(value, dict):
            err = _check_dict(param, value, expected_answers[param])
            if err:
                return err
        else:
            # Numeric/bool: direct containment check
            if value not in expected_answers[param]:
                return ScoreResult(
                    "", False,
                    [f"Parameter '{param}': got {value!r}, expected one of {expected_answers[param]}"],
                    "value_error",
                )

    # 4. Check for required optional params not provided
    for param, acceptable in expected_answers.items():
        if param not in model_params:
            if "" not in acceptable:
                return ScoreResult(
                    "", False,
                    [f"Missing parameter {param!r} (not marked optional in ground truth)"],
                    "missing_optional",
                )

    return ScoreResult("", True)


# ---------------------------------------------------------------------------
# Category-level routing
# ---------------------------------------------------------------------------

def _find_description(func_descriptions: list[dict], name: str) -> dict | None:
    """Find a function description by name."""
    for fd in func_descriptions:
        if fd["name"] == name:
            return fd
    return None


def _score_irrelevance(test_id: str, model_output: list[dict]) -> ScoreResult:
    """Irrelevance: model should make zero tool calls."""
    if len(model_output) == 0:
        return ScoreResult(test_id, True)
    return ScoreResult(
        test_id, False,
        [f"Expected 0 calls, got {len(model_output)}"],
        "irrelevance:called_tool",
    )


def _score_simple(
    test_id: str,
    func_descriptions: list[dict],
    model_output: list[dict],
    ground_truth: list[dict],
) -> ScoreResult:
    """Simple/multiple: exactly one function call expected."""
    if len(model_output) != 1:
        return ScoreResult(
            test_id, False,
            [f"Expected 1 call, got {len(model_output)}"],
            "wrong_count",
        )

    # Find the matching function description
    expected_func = list(ground_truth[0].keys())[0]
    func_desc = _find_description(func_descriptions, expected_func)
    if func_desc is None:
        return ScoreResult(
            test_id, False,
            [f"No description for {expected_func!r}"],
            "internal_error",
        )

    result = simple_function_checker(func_desc, model_output[0], ground_truth[0])
    result.test_id = test_id
    return result


def _score_parallel(
    test_id: str,
    func_descriptions: list[dict],
    model_output: list[dict],
    ground_truth: list[dict],
) -> ScoreResult:
    """Parallel: N calls, order-independent matching."""
    if len(model_output) != len(ground_truth):
        return ScoreResult(
            test_id, False,
            [f"Expected {len(ground_truth)} calls, got {len(model_output)}"],
            "wrong_count",
        )

    # Try to match each ground truth entry to a model output (order-independent)
    matched = set()
    for gt in ground_truth:
        expected_func = list(gt.keys())[0]
        func_desc = _find_description(func_descriptions, expected_func)
        if func_desc is None:
            return ScoreResult(
                test_id, False,
                [f"No description for {expected_func!r}"],
                "internal_error",
            )

        found = False
        for idx, mo in enumerate(model_output):
            if idx in matched:
                continue
            result = simple_function_checker(func_desc, mo, gt)
            if result.valid:
                matched.add(idx)
                found = True
                break

        if not found:
            return ScoreResult(
                test_id, False,
                [f"No matching call for {expected_func!r}"],
                "parallel:no_match",
            )

    return ScoreResult(test_id, True)


def score_entry(
    test_id: str,
    func_descriptions: list[dict],
    model_output: list[dict],
    ground_truth: list[dict],
    category: str,
) -> ScoreResult:
    """Score one test case based on category type.

    Args:
        test_id: BFCL test ID
        func_descriptions: BFCL function schema dicts
        model_output: Extracted calls from runner [{"func": {"p": v}}, ...]
        ground_truth: From possible_answer JSONL
        category: BFCL category name
    """
    if "irrelevance" in category:
        return _score_irrelevance(test_id, model_output)
    elif "parallel" in category:
        return _score_parallel(test_id, func_descriptions, model_output, ground_truth)
    elif "multiple" in category:
        return _score_simple(test_id, func_descriptions, model_output, ground_truth)
    else:
        return _score_simple(test_id, func_descriptions, model_output, ground_truth)


# ---------------------------------------------------------------------------
# Batch scoring utility
# ---------------------------------------------------------------------------

def score_category(
    run_results: list[BfclRunResult],
    category: str,
    data_dir: Path | None = None,
) -> list[ScoreResult]:
    """Score all results for a category against ground truth.

    Args:
        run_results: Results from runner.run_category()
        category: BFCL category name
        data_dir: Path to BFCL data dir (default: tests/eval/bfcl/data/)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"

    from tests.eval.bfcl.schema_adapter import CATEGORY_FILES, load_jsonl

    test_path = data_dir / CATEGORY_FILES[category]
    gt_path = data_dir / "possible_answer" / CATEGORY_FILES[category]

    test_entries = {e["id"]: e for e in load_jsonl(test_path)}
    gt_entries = {e["id"]: e for e in load_jsonl(gt_path)}

    scores = []
    for result in run_results:
        if not result.completed:
            scores.append(ScoreResult(
                result.test_id, False,
                [f"Workflow did not complete: {result.error}"],
                "incomplete",
            ))
            continue

        test_entry = test_entries[result.test_id]
        gt_entry = gt_entries[result.test_id]

        score = score_entry(
            test_id=result.test_id,
            func_descriptions=test_entry["function"],
            model_output=result.extracted_calls,
            ground_truth=gt_entry["ground_truth"],
            category=category,
        )
        scores.append(score)

    return scores


# ---------------------------------------------------------------------------
# Ground truth execution
# ---------------------------------------------------------------------------


def _parse_call_string(call_str: str) -> tuple[str, dict[str, Any]]:
    """Parse a ground truth call string into (func_name, kwargs).

    Handles strings like:
    - ``"cd(folder='document')"``
    - ``"mv(source='a.txt', destination='b/')"``
    - ``"sort('file.txt')"`` (positional — mapped to first param by position)
    - ``"add(a=2.0, b=3.0)"``

    Uses Python's ``ast`` module for safe parsing.
    """
    tree = ast.parse(call_str, mode="eval")
    call_node = tree.body
    assert isinstance(call_node, ast.Call), f"Not a call: {call_str}"

    # Extract function name
    if isinstance(call_node.func, ast.Name):
        func_name = call_node.func.id
    elif isinstance(call_node.func, ast.Attribute):
        func_name = call_node.func.attr
    else:
        raise ValueError(f"Unexpected call target: {call_str}")

    # Extract kwargs
    kwargs: dict[str, Any] = {}
    for kw in call_node.keywords:
        kwargs[kw.arg] = ast.literal_eval(kw.value)

    # Handle positional args (e.g. sort('file.txt'))
    for i, arg in enumerate(call_node.args):
        kwargs[f"_positional_{i}"] = ast.literal_eval(arg)

    return func_name, kwargs


def _resolve_positional_args(kwargs: dict, param_names: list[str]) -> dict:
    """Convert _positional_N keys to actual parameter names."""
    resolved = {}
    positional = {}
    for k, v in kwargs.items():
        if k.startswith("_positional_"):
            idx = int(k.split("_")[-1])
            positional[idx] = v
        else:
            resolved[k] = v
    for idx, val in sorted(positional.items()):
        if idx < len(param_names):
            resolved[param_names[idx]] = val
    return resolved


def _normalize_result(result: Any) -> str:
    """Normalize a backend method result to a string.

    Matches BFCL's execute_multi_turn_func_call normalization.
    """
    if isinstance(result, str):
        return result
    elif isinstance(result, dict):
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)
    elif result is None:
        return ""
    else:
        return str(result)


def execute_ground_truth(
    call_strings: list[str],
    instances: dict[str, Any],
) -> list[str]:
    """Execute ground truth function-call strings against backend instances.

    Parses each string like ``"cd(folder='temp')"`` into a method name and
    kwargs, then calls the bound method on the matching instance.

    Args:
        call_strings: Python call strings from the ground truth entry.
        instances: Backend instances (keyed by class name).

    Returns:
        List of execution result strings (JSON-serialized dicts, or str()).
    """
    # Build method and signature lookups
    method_map: dict[str, Any] = {}
    sig_map: dict[str, list[str]] = {}
    for class_name, instance in instances.items():
        for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if not name.startswith("_"):
                method_map[name] = method
                sig = inspect.signature(method)
                sig_map[name] = [
                    p.name for p in sig.parameters.values()
                    if p.name != "self"
                ]

    results: list[str] = []
    for call_str in call_strings:
        try:
            func_name, kwargs = _parse_call_string(call_str)
            if func_name not in method_map:
                results.append(f"Error: unknown function {func_name}")
                continue
            resolved = _resolve_positional_args(kwargs, sig_map.get(func_name, []))
            result = method_map[func_name](**resolved)
            results.append(_normalize_result(result))
        except Exception as e:
            results.append(f"Error during execution: {e}")

    return results


# ---------------------------------------------------------------------------
# Multi-turn scoring
# ---------------------------------------------------------------------------


def state_checker(
    model_instances: dict[str, Any],
    gt_instances: dict[str, Any],
) -> ScoreResult:
    """Compare public attributes of model instances vs ground truth instances.

    Every public attribute (non-underscore) of each ground truth instance
    must have an equal value in the corresponding model instance.
    """
    for class_name, gt_instance in gt_instances.items():
        model_instance = model_instances[class_name]

        for attr_name in vars(gt_instance):
            if attr_name.startswith("_"):
                continue
            model_val = getattr(model_instance, attr_name)
            gt_val = getattr(gt_instance, attr_name)
            if model_val != gt_val:
                return ScoreResult(
                    "", False,
                    [f"State mismatch in {class_name}.{attr_name}: "
                     f"model={model_val!r}, gt={gt_val!r}"],
                    "multi_turn:state_mismatch",
                )

    return ScoreResult("", True)


def response_checker(
    all_model_results: list[str],
    gt_turn_results: list[str],
    turn_index: int,
) -> ScoreResult:
    """Check that all ground truth results appear in model's accumulated results.

    Unordered containment: each ground truth result must appear at least once
    in the model's cumulative results. Handles duplicates correctly (each
    model result can only match one ground truth result).
    """
    remaining = list(all_model_results)
    missing = []
    for gt_result in gt_turn_results:
        try:
            remaining.remove(gt_result)
        except ValueError:
            missing.append(gt_result)

    if missing:
        return ScoreResult(
            "", False,
            [f"Turn {turn_index}: missing ground truth results: {missing}"],
            "multi_turn:response_mismatch",
        )
    return ScoreResult("", True)


def irrelevance_checker(
    model_calls: list[dict],
    turn_index: int,
) -> ScoreResult:
    """Check that model made no calls on an empty ground truth turn."""
    if len(model_calls) > 0:
        return ScoreResult(
            "", False,
            [f"Turn {turn_index}: expected 0 calls, got {len(model_calls)}"],
            "multi_turn:irrelevance_error",
        )
    return ScoreResult("", True)


def score_multi_turn_entry(
    result: BfclMultiTurnResult,
    entry: dict,
    ground_truth: list[list[str]],
    category: str,
) -> ScoreResult:
    """Score one multi-turn test case.

    Executes ground truth calls against separate backend instances, then
    runs state_checker + irrelevance_checker per turn.
    """
    test_id = result.test_id
    long_context = "long_context" in category

    if not result.completed:
        return ScoreResult(
            test_id, False,
            [f"Workflow did not complete: {result.error}"],
            "incomplete",
        )

    initial_config = entry["initial_config"]
    involved_classes = entry["involved_classes"]

    from tests.eval.bfcl.backend_wiring import create_instances

    # Create separate ground truth instances
    gt_instances = create_instances(initial_config, involved_classes, long_context=long_context)

    for turn_index, gt_turn_calls in enumerate(ground_truth):
        # Execute ground truth calls on gt_instances
        if gt_turn_calls:
            execute_ground_truth(gt_turn_calls, gt_instances)

        # Get model's calls for this turn
        model_turn_calls = (
            result.per_turn_calls[turn_index]
            if turn_index < len(result.per_turn_calls)
            else []
        )

        # Irrelevance check
        if not gt_turn_calls:
            irr = irrelevance_checker(model_turn_calls, turn_index)
            if not irr.valid:
                irr.test_id = test_id
                return irr
            continue

        # Model had non-empty ground truth — check it produced calls
        if not model_turn_calls:
            return ScoreResult(
                test_id, False,
                [f"Turn {turn_index}: model produced no calls, "
                 f"expected {len(gt_turn_calls)} ground truth calls"],
                "multi_turn:empty_model_response",
            )

        # State check
        state = state_checker(result.instances, gt_instances)
        if not state.valid:
            state.test_id = test_id
            return state

    return ScoreResult(test_id, True)


def score_multi_turn_category(
    results: list[BfclMultiTurnResult],
    category: str,
    data_dir: Path | None = None,
) -> list[ScoreResult]:
    """Score all results for a multi-turn category."""
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"

    from tests.eval.bfcl.schema_adapter import MULTI_TURN_CATEGORIES, load_jsonl, load_multi_turn_ground_truth

    test_path = data_dir / MULTI_TURN_CATEGORIES[category]
    gt_path = data_dir / "possible_answer" / MULTI_TURN_CATEGORIES[category]

    test_entries = {e["id"]: e for e in load_jsonl(test_path)}
    gt_entries = {e["id"]: e for e in load_jsonl(gt_path)}

    scores = []
    for result in results:
        gt_entry = gt_entries.get(result.test_id)
        test_entry = test_entries.get(result.test_id)

        if gt_entry is None or test_entry is None:
            scores.append(ScoreResult(
                result.test_id, False,
                [f"Missing ground truth or test entry for {result.test_id}"],
                "internal_error",
            ))
            continue

        ground_truth = load_multi_turn_ground_truth(gt_entry)
        score = score_multi_turn_entry(result, test_entry, ground_truth, category)
        scores.append(score)

    return scores
