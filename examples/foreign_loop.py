"""Using forge's guardrail middleware in your own agentic loop.

If you already have an orchestration loop (BFCL harness, openClaw pipeline,
custom agent), you don't need to adopt WorkflowRunner. Plug forge's
guardrails into your existing loop with two calls:

    result = guardrails.check(response)   # before execution
    done   = guardrails.record(["tool"])  # after execution

This file has two parts:
  Part 1 -- The simple API (Guardrails facade)
  Part 2 -- The granular API (individual middleware components)

No LLM backend required -- uses scripted responses to show each guardrail.
"""

from __future__ import annotations

from forge.core.workflow import TextResponse, ToolCall


# =====================================================================
# Part 1: Simple API
#
# Two calls: check() before execution, record() after.
# =====================================================================

from forge.guardrails import Guardrails

guardrails = Guardrails(
    tool_names=["search", "lookup", "answer"],
    required_steps=["search", "lookup"],
    terminal_tool="answer",
)


def handle_response_simple(response):
    """Process one LLM response through forge's guardrails."""

    result = guardrails.check(response)

    if result.action == "fatal":
        return f"FATAL: {result.reason}"

    if result.action in ("retry", "step_blocked"):
        # Inject result.nudge into your conversation history
        return f"{result.action}: {result.nudge.content[:80]}..."

    # result.action == "execute"
    # Run the tools yourself, then tell forge what succeeded.
    tool_calls = result.tool_calls
    executed = [tc.tool for tc in tool_calls]  # your execution here
    done = guardrails.record(executed)
    return f"executed {executed}" + (" -- DONE" if done else "")


# =====================================================================
# Part 2: Granular API
#
# Direct access to ResponseValidator, StepEnforcer, and ErrorTracker.
# Same logic as Part 1, but you control each component individually.
# Use this when you need custom behavior between checkpoints (e.g.,
# logging, metrics, conditional rescue, partial execution).
# =====================================================================

from forge.guardrails import ErrorTracker, ResponseValidator, StepEnforcer

validator = ResponseValidator(
    tool_names=["search", "lookup", "answer"],
    rescue_enabled=True,
)
enforcer = StepEnforcer(
    required_steps=["search", "lookup"],
    terminal_tool="answer",
)
errors = ErrorTracker(max_retries=3, max_tool_errors=2)


def handle_response_granular(response):
    """Same logic as handle_response_simple, with full control."""

    # Checkpoint 1: Is this response usable?
    result = validator.validate(response)

    if result.needs_retry:
        errors.record_retry()
        if errors.retries_exhausted:
            return "FATAL: too many consecutive bad responses"
        return f"retry: {result.nudge.content[:80]}..."

    errors.reset_retries()

    # Checkpoint 2: Is the model skipping required steps?
    step_check = enforcer.check(result.tool_calls)

    if step_check.needs_nudge:
        return f"step_blocked: {step_check.nudge.content[:80]}..."

    # Execute tools yourself, then sync up.
    tool_calls = result.tool_calls
    executed = [tc.tool for tc in tool_calls]  # your execution here
    for name in executed:
        enforcer.record(name)
    errors.reset_errors()
    enforcer.reset_premature()

    done = enforcer.terminal_reached(tool_calls)
    return f"executed {executed}" + (" -- DONE" if done else "")


# =====================================================================
# Demo
# =====================================================================

EXAMPLES = [
    ("Model returns plain text (no tool call)",
     TextResponse(content="I think the answer is 42.")),
    ("Model jumps to terminal tool, skipping required steps",
     [ToolCall(tool="answer", args={"text": "42"})]),
    ("Model calls first required step",
     [ToolCall(tool="search", args={"query": "meaning of life"})]),
    ("Model calls second required step",
     [ToolCall(tool="lookup", args={"id": "result-1"})]),
    ("Model calls terminal tool (steps now satisfied)",
     [ToolCall(tool="answer", args={"text": "The answer is 42."})]),
]

if __name__ == "__main__":
    print("=== Part 1: Simple API (Guardrails) ===")
    for i, (desc, response) in enumerate(EXAMPLES, 1):
        print(f"\n{i}. {desc}")
        print(f"   {handle_response_simple(response)}")

    # Reset granular state for Part 2 demo
    enforcer._tracker.completed_steps.clear()
    enforcer._premature_attempts = 0
    errors._consecutive_retries = 0

    print("\n\n=== Part 2: Granular API (individual components) ===")
    for i, (desc, response) in enumerate(EXAMPLES, 1):
        print(f"\n{i}. {desc}")
        print(f"   {handle_response_granular(response)}")
