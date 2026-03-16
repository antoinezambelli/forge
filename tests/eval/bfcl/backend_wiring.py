"""Backend wiring — BFCL backend classes as forge ToolDef executors.

Unlike single-turn stubs, multi-turn executors call REAL backend methods.
State mutates across calls within a test case (e.g. GorillaFileSystem.mv()
actually moves a file in the in-memory filesystem).
"""

from __future__ import annotations

import copy
import inspect
import json
from typing import Any

from forge.core.workflow import ToolDef

from tests.eval.bfcl.schema_adapter import (
    CLASS_MODULE_MAP,
    STATELESS_CLASSES,
    load_func_docs,
    make_tool_def,
)


def _import_class(class_name: str) -> type:
    """Import a BFCL backend class by name."""
    import importlib
    module_name = CLASS_MODULE_MAP[class_name]
    module = importlib.import_module(f"tests.eval.bfcl.backends.{module_name}")
    return getattr(module, class_name)


def create_instances(
    initial_config: dict,
    involved_classes: list[str],
    long_context: bool = False,
) -> dict[str, Any]:
    """Instantiate and configure backend class instances.

    Args:
        initial_config: From test entry ``initial_config`` field.
            Keyed by class name -> config dict.
        involved_classes: List of class names to instantiate.
        long_context: If True, passed to ``_load_scenario()``.

    Returns:
        Dict mapping class name -> configured instance.
    """
    instances: dict[str, Any] = {}
    for class_name in involved_classes:
        cls = _import_class(class_name)
        instance = cls()
        if class_name not in STATELESS_CLASSES:
            config = copy.deepcopy(initial_config.get(class_name, {}))
            instance._load_scenario(config, long_context=long_context)
        instances[class_name] = instance
    return instances


def wire_tools(
    instances: dict[str, Any],
    func_docs: list[dict],
    excluded_functions: list[str] | None = None,
    missed_functions: dict[str, list[str]] | None = None,
    turn_index: int | None = None,
) -> dict[str, ToolDef]:
    """Wire backend instances into forge ToolDefs.

    Each public method on each backend instance becomes a ToolDef whose
    callable invokes the method on that instance (preserving state).

    The schema for each tool comes from the func_doc JSONL. If a method
    exists on the instance but has no matching func_doc entry, it is
    silently skipped (BFCL has some methods that aren't exposed as tools
    in certain test cases).

    Args:
        instances: From ``create_instances()``.
        func_docs: From ``load_func_docs()``. Combined schemas for all
            involved classes.
        excluded_functions: Function names to omit from tool set (from
            test entry ``excluded_function`` field).
        missed_functions: From ``missed_function`` field (miss_func category).
            Maps turn index (as string) to list of function names to withhold
            for that turn.
        turn_index: Current turn index. Used with *missed_functions* to
            determine which functions to withhold this turn.

    Returns:
        Dict mapping tool name -> ToolDef.
    """
    excluded = set(excluded_functions or [])

    # Apply missed_functions for current turn
    if missed_functions and turn_index is not None:
        turn_key = str(turn_index)
        if turn_key in missed_functions:
            excluded = excluded | set(missed_functions[turn_key])

    # Build a lookup: method_name -> (instance, bound_method)
    method_map: dict[str, tuple[Any, Any]] = {}
    for class_name, instance in instances.items():
        for method_name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if method_name.startswith("_"):
                continue
            if method_name in excluded:
                continue
            method_map[method_name] = (instance, method)

    # Build ToolDefs from func_docs that match available methods
    tools: dict[str, ToolDef] = {}
    for schema in func_docs:
        name = schema["name"]
        if name not in method_map:
            continue
        _, bound_method = method_map[name]

        def _make_executor(method):
            """Capture bound method in closure."""
            def executor(**kwargs):
                result = method(**kwargs)
                # Normalise to string (matching BFCL's execute_multi_turn_func_call)
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
            return executor

        tool_def = make_tool_def(schema, _make_executor(bound_method))
        tools[tool_def.spec.name] = tool_def

    return tools
