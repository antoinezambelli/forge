"""Runner — BFCL evaluation through forge WorkflowRunner.

Single-turn: loads BFCL JSONL test cases, builds forge Workflows, runs them,
and extracts tool calls for the scorer.

Multi-turn: loops over BFCL turns, calling WorkflowRunner.run() once per turn
with accumulated message history. Backend instances persist across turns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from forge.clients.base import LLMClient
from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact, TieredCompact
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType
from forge.core.runner import WorkflowRunner
from forge.core.workflow import ToolDef, Workflow
from forge.errors import ForgeError

from tests.eval.bfcl.backend_wiring import create_instances, wire_tools
from tests.eval.bfcl.executors import make_done_tool, make_stub_executor
from tests.eval.bfcl.schema_adapter import (
    CATEGORY_FILES,
    DATA_DIR,
    MULTI_TURN_CATEGORIES,
    load_func_docs,
    load_jsonl,
    load_multi_turn_entry,
    load_test_entry,
    make_tool_def,
    sanitize_param_name,
    sanitize_tool_name,
)


@dataclass
class BfclRunResult:
    """Result of running one BFCL test case through forge."""

    test_id: str
    # List of successful tool calls: [{"func_name": {"param": value, ...}}, ...]
    extracted_calls: list[dict[str, dict[str, Any]]]
    # Did the workflow complete (model called done)?
    completed: bool
    # Error message if workflow raised
    error: str | None = None
    # Number of LLM round-trips
    iterations: int = 0
    # Wall time in seconds
    elapsed_seconds: float = 0.0


def build_workflow(
    function_schemas: list[dict],
    category: str,
) -> tuple[Workflow, dict[str, str], dict[str, str]]:
    """Build a forge Workflow for a BFCL test case.

    Args:
        function_schemas: BFCL function schema dicts from test entry["function"]
        category: BFCL category name (e.g. "simple_python") — used for naming

    Returns:
        (workflow, name_map, param_name_map) where *name_map* maps sanitised
        tool names back to the original BFCL names, and *param_name_map* maps
        sanitised parameter names back to originals (e.g. ``class_`` → ``_class``).
    """
    tools: dict[str, ToolDef] = {}
    name_map: dict[str, str] = {}  # sanitised -> original
    param_name_map: dict[str, str] = {}  # sanitised -> original

    for func_schema in function_schemas:
        original_name = func_schema["name"]
        # Build param_name_map from raw schema before normalisation
        raw_props = func_schema.get("parameters", {}).get("properties", {})
        for pname in raw_props:
            sanitised_pname = sanitize_param_name(pname)
            if sanitised_pname != pname:
                param_name_map[sanitised_pname] = pname

        tool_def = make_tool_def(func_schema, lambda **kw: "")
        sanitised_name = tool_def.spec.name
        executor = make_stub_executor(tool_def.spec)
        tool_def = ToolDef(spec=tool_def.spec, callable=executor)
        tools[sanitised_name] = tool_def
        if sanitised_name != original_name:
            name_map[sanitised_name] = original_name

    # Add synthetic done terminal
    done = make_done_tool()
    tools["done"] = done

    workflow = Workflow(
        name=f"bfcl_{category}",
        description=f"BFCL {category} evaluation",
        tools=tools,
        required_steps=[],
        terminal_tool="done",
        system_prompt_template=(
            "You are a helpful assistant. Use the available tools to answer "
            "the user's question. When you have finished, call the `done` tool."
        ),
    )
    return workflow, name_map, param_name_map


def extract_calls(
    messages: list[Message],
    name_map: dict[str, str] | None = None,
    param_name_map: dict[str, str] | None = None,
) -> list[dict[str, dict[str, Any]]]:
    """Extract successful tool calls from a completed workflow's messages.

    Returns list of {"func_name": {"param": value}} dicts, matching
    BFCL's expected format for ast_checker.

    Args:
        messages: Collected workflow messages.
        name_map: Optional mapping of sanitised tool names back to the
            original BFCL names.  If provided, extracted call names are
            restored to originals so the scorer can match ground truth.
        param_name_map: Optional mapping of sanitised parameter names
            back to originals (e.g. ``class_`` → ``_class``).

    A tool call is "successful" if:
    - It has a TOOL_CALL message with tool_calls entries
    - The immediately following TOOL_RESULT message doesn't start with
      '[ToolError]'
    - The tool name is not 'done'
    """
    result = []
    for i, msg in enumerate(messages):
        if msg.metadata.type != MessageType.TOOL_CALL:
            continue
        if not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            if tc.name == "done":
                continue
            # Find the matching TOOL_RESULT
            tool_result = _find_tool_result(messages, i + 1, tc.call_id)
            if tool_result is not None and not tool_result.content.startswith("[ToolError]"):
                call_name = name_map.get(tc.name, tc.name) if name_map else tc.name
                args = tc.args
                if param_name_map:
                    args = {param_name_map.get(k, k): v for k, v in args.items()}
                result.append({call_name: args})
    return result


def _find_tool_result(
    messages: list[Message], start: int, call_id: str
) -> Message | None:
    """Find the TOOL_RESULT message matching a given call_id."""
    for msg in messages[start:]:
        if (
            msg.metadata.type == MessageType.TOOL_RESULT
            and msg.tool_call_id == call_id
        ):
            return msg
    return None


async def run_single_entry(
    client: LLMClient,
    entry: dict,
    category: str,
    stream: bool = False,
    max_iterations: int = 15,
    max_retries_per_step: int = 5,
    max_tool_errors: int = 2,
    budget: int | None = None,
    rescue_enabled: bool = True,
) -> BfclRunResult:
    """Run a single BFCL test case through forge.

    Args:
        client: An LLMClient instance (Ollama, Llamafile, or Anthropic)
        entry: One parsed BFCL JSONL entry
        category: e.g. "simple_python", "parallel", "multiple", "irrelevance"
        stream: Whether to use streaming
        budget: Context budget in tokens (None = no compaction)
    """
    test_id, user_messages, function_schemas = load_test_entry(entry)

    workflow, name_map, param_name_map = build_workflow(function_schemas, category)

    # The user message is the content from the first (only) message
    # Single-turn: there's exactly one user message
    user_text = user_messages[0]["content"]

    strategy = NoCompact() if budget is None else TieredCompact()
    ctx = ContextManager(
        budget_tokens=budget or 200_000,  # Large default if no budget
        strategy=strategy,
    )

    collected_messages: list[Message] = []

    runner = WorkflowRunner(
        client=client,
        context_manager=ctx,
        max_iterations=max_iterations,
        max_retries_per_step=max_retries_per_step,
        max_tool_errors=max_tool_errors,
        stream=stream,
        on_message=lambda msg: collected_messages.append(msg),
        rescue_enabled=rescue_enabled,
    )

    t0 = time.monotonic()
    completed = False
    error = None
    iterations = 0

    try:
        await runner.run(workflow, user_text)
        completed = True
    except ForgeError as e:
        error = f"{type(e).__name__}: {e}"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    elapsed = time.monotonic() - t0

    # Count iterations (number of TOOL_CALL messages = LLM round-trips that
    # produced a tool call). Also count TEXT_RESPONSE as iterations since
    # those are LLM calls that didn't produce a valid tool call.
    iterations = sum(
        1 for m in collected_messages
        if m.metadata.type in (MessageType.TOOL_CALL, MessageType.TEXT_RESPONSE)
    )

    extracted = extract_calls(collected_messages, name_map, param_name_map) if completed else []

    return BfclRunResult(
        test_id=test_id,
        extracted_calls=extracted,
        completed=completed,
        error=error,
        iterations=iterations,
        elapsed_seconds=elapsed,
    )


async def run_category(
    client: LLMClient,
    category: str,
    stream: bool = False,
    max_entries: int | None = None,
    entry_ids: list[str] | None = None,
    budget: int | None = None,
    verbose: bool = False,
) -> list[BfclRunResult]:
    """Run all test cases in a BFCL category.

    Args:
        category: Key from CATEGORY_FILES (e.g. "simple_python")
        max_entries: Limit number of entries to run (None = all)
        entry_ids: Run only entries with these IDs (overrides max_entries)
        verbose: Print progress per entry
    """
    filename = CATEGORY_FILES[category]
    data_path = DATA_DIR / filename
    entries = load_jsonl(data_path)

    if entry_ids is not None:
        id_set = set(entry_ids)
        entries = [e for e in entries if e["id"] in id_set]
    elif max_entries is not None:
        entries = entries[:max_entries]

    results = []
    for i, entry in enumerate(entries):
        if verbose:
            print(f"  [{i+1}/{len(entries)}] {entry['id']} ... ", end="", flush=True)
        result = await run_single_entry(
            client, entry, category,
            stream=stream, budget=budget,
        )
        if verbose:
            status = "OK" if result.completed else f"FAIL: {result.error}"
            calls = len(result.extracted_calls)
            print(f"{status} ({calls} calls, {result.iterations} iters, "
                  f"{result.elapsed_seconds:.1f}s)")
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Multi-turn runner
# ---------------------------------------------------------------------------


@dataclass
class BfclMultiTurnResult:
    """Result of running one multi-turn BFCL test case."""

    test_id: str
    # Per-turn extracted calls: outer list = turns, inner = calls for that turn
    per_turn_calls: list[list[dict[str, dict[str, Any]]]]
    # Per-turn completion status
    per_turn_completed: list[bool]
    # Backend instances after all turns (for state comparison by scorer)
    instances: dict[str, Any]
    # Did all turns complete?
    completed: bool
    error: str | None = None
    # Total wall time
    elapsed_seconds: float = 0.0


def build_multi_turn_workflow(
    func_docs: list[dict],
    instances: dict[str, Any],
    category: str,
    excluded_functions: list[str] | None = None,
    missed_functions: dict[str, list[str]] | None = None,
    turn_index: int = 0,
) -> tuple[Workflow, dict[str, str], dict[str, str]]:
    """Build a forge Workflow for one turn of a multi-turn test case.

    A new Workflow is built each turn because the tool set can change
    (miss_func category withholds tools on specific turns).

    Returns:
        (workflow, name_map, param_name_map) — *name_map* maps sanitised
        tool names back to original BFCL names, *param_name_map* maps
        sanitised param names back to originals.
    """
    tools = wire_tools(
        instances, func_docs,
        excluded_functions=excluded_functions,
        missed_functions=missed_functions,
        turn_index=turn_index,
    )

    # Build name_map and param_name_map from func_docs
    name_map: dict[str, str] = {}
    param_name_map: dict[str, str] = {}
    for schema in func_docs:
        original = schema["name"]
        sanitised = sanitize_tool_name(original)
        if sanitised != original:
            name_map[sanitised] = original
        for pname in schema.get("parameters", {}).get("properties", {}):
            sanitised_pname = sanitize_param_name(pname)
            if sanitised_pname != pname:
                param_name_map[sanitised_pname] = pname

    # Add done terminal
    done = make_done_tool()
    tools["done"] = done

    workflow = Workflow(
        name=f"bfcl_{category}_turn{turn_index}",
        description=f"BFCL {category} turn {turn_index}",
        tools=tools,
        required_steps=[],
        terminal_tool="done",
        system_prompt_template=(
            "You are a helpful assistant. Use the available tools to answer "
            "the user's question. When you have finished all the actions for "
            "the current request, call the `done` tool."
        ),
    )
    return workflow, name_map, param_name_map


async def run_multi_turn_entry(
    client: LLMClient,
    entry: dict,
    category: str,
    stream: bool = False,
    max_iterations: int = 20,
    max_total_iterations: int = 30,
    budget: int | None = None,
    max_retries_per_step: int = 5,
    max_tool_errors: int = 2,
    rescue_enabled: bool = True,
) -> BfclMultiTurnResult:
    """Run a single multi-turn BFCL test case through forge.

    Loops over turns. Each turn is a separate WorkflowRunner.run() call.
    Message history carries forward between turns so the model sees the
    full conversation. Backend instances persist across turns (stateful).

    Args:
        client: An LLMClient instance.
        entry: One parsed BFCL multi-turn JSONL entry.
        category: e.g. "multi_turn_base", "multi_turn_miss_func".
        stream: Whether to use streaming.
        max_iterations: Max LLM round-trips per turn.
        max_total_iterations: Max LLM round-trips across all turns combined.
        budget: Context budget in tokens (None = no compaction).
    """
    test_id, turns, initial_config, involved_classes = load_multi_turn_entry(entry)
    long_context = "long_context" in category

    # Create backend instances (persist across all turns)
    instances = create_instances(initial_config, involved_classes, long_context=long_context)

    # Load function schemas
    func_docs = load_func_docs(involved_classes)
    excluded = entry.get("excluded_function", [])
    missed = entry.get("missed_function", None)

    strategy = NoCompact() if budget is None else TieredCompact()

    per_turn_calls: list[list[dict[str, dict[str, Any]]]] = []
    per_turn_completed: list[bool] = []
    conversation: list[Message] = []  # Accumulated message history
    error: str | None = None
    total_iterations_used = 0

    t0 = time.monotonic()

    for turn_index, turn_messages in enumerate(turns):
        # Skip empty turns (miss_func can have empty question turns)
        if not turn_messages:
            per_turn_calls.append([])
            per_turn_completed.append(True)
            continue

        # Cap this turn's iterations by remaining total budget
        remaining = max_total_iterations - total_iterations_used
        if remaining <= 0:
            error = f"Turn {turn_index}: total iteration cap ({max_total_iterations}) reached"
            per_turn_calls.append([])
            per_turn_completed.append(False)
            break
        turn_max = min(max_iterations, remaining)

        # Build workflow for this turn (tool set may differ per turn)
        workflow, name_map, param_name_map = build_multi_turn_workflow(
            func_docs, instances, category,
            excluded_functions=excluded,
            missed_functions=missed,
            turn_index=turn_index,
        )

        # Get user message for this turn
        user_text = turn_messages[0]["content"]

        # Collect messages emitted during this turn
        turn_new_messages: list[Message] = []

        ctx = ContextManager(
            budget_tokens=budget or 200_000,
            strategy=strategy,
        )
        runner = WorkflowRunner(
            client=client,
            context_manager=ctx,
            max_iterations=turn_max,
            max_retries_per_step=max_retries_per_step,
            max_tool_errors=max_tool_errors,
            stream=stream,
            on_message=lambda msg: turn_new_messages.append(msg),
            rescue_enabled=rescue_enabled,
        )

        if turn_index == 0:
            # First turn: normal run (builds system prompt + user input)
            try:
                await runner.run(workflow, user_text)
                per_turn_completed.append(True)
            except (ForgeError, Exception) as e:
                error = f"Turn {turn_index}: {type(e).__name__}: {e}"
                per_turn_completed.append(False)
        else:
            # Subsequent turns: carry forward conversation, append new user message
            seed = list(conversation)
            seed.append(Message(
                MessageRole.USER, user_text,
                MessageMeta(MessageType.USER_INPUT),
            ))
            try:
                await runner.run(workflow, user_text, initial_messages=seed)
                per_turn_completed.append(True)
            except (ForgeError, Exception) as e:
                error = f"Turn {turn_index}: {type(e).__name__}: {e}"
                per_turn_completed.append(False)

        # Count iterations used this turn
        turn_iters = sum(
            1 for m in turn_new_messages
            if m.metadata.type in (MessageType.TOOL_CALL, MessageType.TEXT_RESPONSE)
        )
        total_iterations_used += turn_iters

        # Accumulate conversation history
        conversation.extend(turn_new_messages)

        # Extract calls for this turn
        turn_calls = extract_calls(turn_new_messages, name_map, param_name_map)
        per_turn_calls.append(turn_calls)

        # If turn failed, stop (state is unreliable)
        if not per_turn_completed[-1]:
            break

    elapsed = time.monotonic() - t0

    return BfclMultiTurnResult(
        test_id=test_id,
        per_turn_calls=per_turn_calls,
        per_turn_completed=per_turn_completed,
        instances=instances,
        completed=all(per_turn_completed),
        error=error,
        elapsed_seconds=elapsed,
    )


async def run_multi_turn_category(
    client: LLMClient,
    category: str,
    stream: bool = False,
    max_entries: int | None = None,
    entry_ids: list[str] | None = None,
    budget: int | None = None,
    verbose: bool = False,
) -> list[BfclMultiTurnResult]:
    """Run all test cases in a multi-turn BFCL category.

    Args:
        category: Key from MULTI_TURN_CATEGORIES (e.g. "multi_turn_base").
        max_entries: Limit number of entries (None = all).
        entry_ids: Run only entries with these IDs (overrides max_entries).
        verbose: Print progress per entry.
    """
    filename = MULTI_TURN_CATEGORIES[category]
    data_path = DATA_DIR / filename
    entries = load_jsonl(data_path)

    if entry_ids is not None:
        id_set = set(entry_ids)
        entries = [e for e in entries if e["id"] in id_set]
    elif max_entries is not None:
        entries = entries[:max_entries]

    results = []
    for i, entry in enumerate(entries):
        if verbose:
            print(f"  [{i+1}/{len(entries)}] {entry['id']} ... ", end="", flush=True)
        result = await run_multi_turn_entry(
            client, entry, category,
            stream=stream, budget=budget,
        )
        if verbose:
            ok = "OK" if result.completed else f"FAIL: {result.error}"
            n_turns = len(result.per_turn_calls)
            total_calls = sum(len(tc) for tc in result.per_turn_calls)
            print(f"{ok} ({n_turns} turns, {total_calls} calls, "
                  f"{result.elapsed_seconds:.1f}s)")
        results.append(result)

    return results
