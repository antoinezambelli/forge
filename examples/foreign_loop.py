"""Minimal example: using forge's guardrail middleware in your own loop.

This shows how to get forge's reliability stack (rescue parsing, retry nudges,
step enforcement, error tracking) without using WorkflowRunner. You own the
loop, the LLM client, and tool execution — forge just advises.

No LLM backend required — the example uses a fake model that demonstrates
each guardrail firing in sequence.
"""

from __future__ import annotations

from forge.core.workflow import TextResponse, ToolCall
from forge.guardrails import ErrorTracker, ResponseValidator, StepEnforcer


# ── Setup (once per session) ────────────────────────────────────

validator = ResponseValidator(
    tool_names=["search", "lookup", "answer"],
    rescue_enabled=True,
)
enforcer = StepEnforcer(
    required_steps=["search", "lookup"],
    terminal_tool="answer",
)
errors = ErrorTracker(max_retries=3, max_tool_errors=2)


# ── Fake model responses (simulates a misbehaving LLM) ─────────

FAKE_RESPONSES = [
    # Turn 1: model returns plain text (no tool call)
    TextResponse(content="I think the answer is 42."),
    # Turn 2: model tries to jump straight to answer (premature terminal)
    [ToolCall(tool="answer", args={"text": "42"})],
    # Turn 3: model does the right thing
    [ToolCall(tool="search", args={"query": "meaning of life"})],
    # Turn 4: model does the next step
    [ToolCall(tool="lookup", args={"id": "result-1"})],
    # Turn 5: model calls answer (steps satisfied — allowed)
    [ToolCall(tool="answer", args={"text": "The answer is 42."})],
]


# ── Fake tool execution ────────────────────────────────────────

def execute_tool(tool_call: ToolCall) -> str:
    return f"[{tool_call.tool}] executed with {tool_call.args}"


# ── The consumer's own loop ────────────────────────────────────

def run():
    messages: list[dict[str, str]] = []
    turn = 0

    for response in FAKE_RESPONSES:
        turn += 1
        print(f"\n-- Turn {turn} --")

        # Phase 1: validate response (rescue, retry, unknown tool check)
        result = validator.validate(response)

        if result.needs_retry:
            errors.record_retry()
            if errors.retries_exhausted:
                print("FATAL: retries exhausted")
                return
            nudge = result.nudge
            print(f"  RETRY ({nudge.kind}): {nudge.content[:80]}...")
            messages.append({"role": nudge.role, "content": nudge.content})
            continue

        # Valid tool calls — reset retry counter
        errors.reset_retries()
        tool_calls = result.tool_calls

        # Phase 2: check step requirements
        step_check = enforcer.check(tool_calls)

        if step_check.needs_nudge:
            nudge = step_check.nudge
            print(f"  STEP NUDGE (tier {nudge.tier}): {nudge.content[:80]}...")
            messages.append({"role": nudge.role, "content": nudge.content})
            continue

        # Phase 3: execute tools
        batch_ok = True
        for tc in tool_calls:
            try:
                result_text = execute_tool(tc)
                enforcer.record(tc.tool)
                errors.record_result(success=True)
                print(f"  OK: {result_text}")
            except Exception:
                errors.record_result(success=False)
                batch_ok = False

        if batch_ok:
            errors.reset_errors()
            enforcer.reset_premature()

        # Phase 4: check for terminal
        if errors.tool_errors_exhausted:
            print("FATAL: tool errors exhausted")
            return

        if enforcer.terminal_reached(tool_calls):
            print("\nDone - terminal tool reached after all steps satisfied.")
            return

    print("\nLoop ended without terminal.")


if __name__ == "__main__":
    run()
