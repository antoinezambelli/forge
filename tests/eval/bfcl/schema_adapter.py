"""SchemaAdapter — convert BFCL function schemas into forge ToolDef objects.

Shared data-loading utilities also live here so the runner (P1-3) and
scorer (P1-4) have a single source of truth for BFCL data access.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forge import ToolDef, ToolSpec

# ---------------------------------------------------------------------------
# Data paths & category map
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"

CATEGORY_FILES = {
    "simple_python": "BFCL_v4_simple_python.json",
    "simple_java": "BFCL_v4_simple_java.json",
    "simple_javascript": "BFCL_v4_simple_javascript.json",
    "multiple": "BFCL_v4_multiple.json",
    "parallel": "BFCL_v4_parallel.json",
    "parallel_multiple": "BFCL_v4_parallel_multiple.json",
    "irrelevance": "BFCL_v4_irrelevance.json",
}


# ---------------------------------------------------------------------------
# Schema normalisation
# ---------------------------------------------------------------------------


_BFCL_TYPE_MAP: dict[str, str] = {
    "dict": "object",
    "String": "string",
    "Boolean": "boolean",
    "Array": "array",
    "ArrayList": "array",
    "HashMap": "object",
    "float": "number",
    "double": "number",
    "long": "integer",
    "char": "string",
    "tuple": "array",
    "any": "string",
}


def normalize_schema(schema: dict) -> dict:
    """Deep-copy *schema* and fix BFCL quirks.

    BFCL datasets use language-specific type names (e.g. ``String``,
    ``Boolean``, ``ArrayList``, ``dict``) instead of standard JSON Schema
    types.  This function maps them to valid JSON Schema equivalents.

    Also strips leading underscores from property names (e.g. ``_class``
    → ``class_``) because Pydantic rejects them.  Use
    ``sanitize_param_name`` to obtain the mapping for round-tripping.
    """
    schema = copy.deepcopy(schema)
    _fix_types(schema)
    _fix_param_names(schema)
    return schema


def sanitize_param_name(name: str) -> str:
    """Strip leading underscore from a parameter name.

    BFCL uses ``_class`` and ``_from`` to avoid Python keyword clashes.
    Pydantic's ``create_model`` rejects leading underscores, so we move
    the underscore to a trailing position: ``_class`` → ``class_``.
    """
    if name.startswith("_"):
        return name.lstrip("_") + "_"
    return name


def _fix_types(node: Any) -> None:
    """Recursively replace non-standard BFCL types with JSON Schema types."""
    if not isinstance(node, dict):
        return
    raw = node.get("type")
    if isinstance(raw, str) and raw in _BFCL_TYPE_MAP:
        node["type"] = _BFCL_TYPE_MAP[raw]
    for value in node.values():
        if isinstance(value, dict):
            _fix_types(value)


def _fix_param_names(schema: dict) -> None:
    """Rename underscore-prefixed property keys in-place.

    Only touches the top-level ``properties`` and ``required`` of *schema*
    (parameter names live at the top level of a function's parameter schema).
    """
    props = schema.get("properties")
    if not props:
        return
    renames = {k: sanitize_param_name(k) for k in props if k.startswith("_")}
    if not renames:
        return
    for old, new in renames.items():
        props[new] = props.pop(old)
    req = schema.get("required")
    if req:
        schema["required"] = [renames.get(r, r) for r in req]


# ---------------------------------------------------------------------------
# Tool-name sanitisation
# ---------------------------------------------------------------------------

_VALID_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def sanitize_tool_name(name: str) -> str:
    """Replace characters invalid for Anthropic/OpenAI tool names.

    BFCL uses dotted names like ``math.factorial`` or ``AclApi.add_mapping``.
    Both Anthropic and OpenAI restrict tool names to ``[a-zA-Z0-9_-]``.
    """
    if _VALID_TOOL_NAME.match(name):
        return name
    return name.replace(".", "_")


# ---------------------------------------------------------------------------
# ToolDef construction
# ---------------------------------------------------------------------------


def make_tool_def(func_schema: dict, executor: Callable) -> ToolDef:
    """Convert a single BFCL function schema into a forge ``ToolDef``.

    The tool name is sanitised (dots replaced with underscores) so it is
    accepted by all backends.  Callers that need the original BFCL name
    should compare ``func_schema["name"]`` with ``tool_def.spec.name``.
    """
    name = sanitize_tool_name(func_schema["name"])
    description = func_schema["description"]
    raw_params = func_schema.get("parameters", {"type": "object", "properties": {}})
    normalized = normalize_schema(raw_params)
    spec = ToolSpec.from_json_schema(name, description, normalized)
    return ToolDef(spec=spec, callable=executor)


# ---------------------------------------------------------------------------
# Entry parsing
# ---------------------------------------------------------------------------


def load_test_entry(entry: dict) -> tuple[str, list[dict], list[dict]]:
    """Parse a BFCL test entry.

    Returns:
        (test_id, messages, function_schemas)
        - test_id: e.g. ``"simple_python_0"``
        - messages: list of message dicts for the first turn (``question[0]``)
        - function_schemas: list of raw BFCL function schema dicts
    """
    test_id = entry["id"]
    messages = entry["question"][0]  # First (only) turn for single-turn
    function_schemas = entry["function"]
    return test_id, messages, function_schemas


def load_ground_truth(entry: dict) -> list[dict]:
    """Parse a BFCL ground truth entry.

    Returns:
        List of acceptable answer dicts.  Each dict maps
        ``function_name -> {param: [acceptable_values]}``.
    """
    return entry["ground_truth"]


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    entries: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Multi-turn constants
# ---------------------------------------------------------------------------

# Maps BFCL class name -> module name under tests.eval.bfcl.backends
CLASS_MODULE_MAP: dict[str, str] = {
    "GorillaFileSystem": "gorilla_file_system",
    "TwitterAPI": "posting_api",
    "MessageAPI": "message_api",
    "MathAPI": "math_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}

# Classes that have no _load_scenario() (no state to initialise)
STATELESS_CLASSES: set[str] = {"MathAPI"}

MULTI_TURN_CATEGORIES = {
    "multi_turn_base": "BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
}

FUNC_DOC_DIR = DATA_DIR / "multi_turn_func_doc"


# ---------------------------------------------------------------------------
# Multi-turn entry parsing
# ---------------------------------------------------------------------------


def load_multi_turn_entry(entry: dict) -> tuple[
    str,               # test_id
    list[list[dict]],  # turns: question[turn_idx] = list of message dicts
    dict,              # initial_config
    list[str],         # involved_classes
]:
    """Parse a BFCL multi-turn test entry.

    Returns:
        (test_id, turns, initial_config, involved_classes)
        - test_id: e.g. "multi_turn_base_0"
        - turns: question[turn_idx] is a list of message dicts for that turn
        - initial_config: dict keyed by class name -> backend config
        - involved_classes: list of class names needed for this test case
    """
    return (
        entry["id"],
        entry["question"],
        entry["initial_config"],
        entry["involved_classes"],
    )


def load_multi_turn_ground_truth(entry: dict) -> list[list[str]]:
    """Parse a BFCL multi-turn ground truth entry.

    Returns:
        List of turns. Each turn is a list of Python function-call strings
        like ``["cd(folder='document')", "mkdir(dir_name='temp')"]``.
        Empty list for a turn means "no calls expected" (irrelevance).
    """
    return entry["ground_truth"]


# ---------------------------------------------------------------------------
# Func doc loading
# ---------------------------------------------------------------------------


def load_func_docs(involved_classes: list[str]) -> list[dict]:
    """Load multi-turn function schemas for the given classes.

    Reads JSONL files from ``data/multi_turn_func_doc/{module_name}.json``
    for each class in *involved_classes*. Returns the combined list of
    raw BFCL function schema dicts (same format as single-turn ``function``
    field).

    Args:
        involved_classes: e.g. ["GorillaFileSystem", "TwitterAPI"]

    Returns:
        Combined list of function schema dicts from all involved classes.
    """
    schemas: list[dict] = []
    for class_name in involved_classes:
        module_name = CLASS_MODULE_MAP[class_name]
        path = FUNC_DOC_DIR / f"{module_name}.json"
        schemas.extend(load_jsonl(path))
    return schemas
