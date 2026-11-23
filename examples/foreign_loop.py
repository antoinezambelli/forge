"""Using forge's guardrail middleware in your own agentic loop.

If you already have an orchestration loop (BFCL harness, openClaw pipeline,
custom agent), you don't need to adopt WorkflowRunner. Plug forge's
guardrails into your existing loop with two calls:

    result = guardrails.check(response)   # before execution
    done   = guardrails.record(["tool"])  # after execution

This file has three parts:
  Part 1 -- The simple API (Guardrails facade)
  Part 2 -- The granular API (individual middleware components)
  Part 3 -- Using the respond tool for chat in tool-equipped sessions

The middleware covers validation, rescue parsing, retry nudges, and step
enforcement. Your loop is responsible for: iteration caps, cancellation,
context management (compaction, threshold callbacks), and streaming. For
the batteries-included version, see WorkflowRunner.

No LLM backend required -- uses scripted responses to show each guardrail.
"""

from __future__ import annotations

from forge.core.messages import RichMessage, MessageRole, TextPart, ToolCallPart, ValidArguments


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

    if result.action in ("retry", "tool_error", "step_blocked"):
        # Inject result.nudge into your conversation history
        return f"{result.action}: {result.nudge.content[:80]}..."

    # result.action == "execute"
    # Run the tools yourself, then tell forge what succeeded.
    tool_calls = result.tool_calls
    executed = [tc.name for tc in tool_calls]  # your execution here
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
    terminal_tools=frozenset({"answer"}),
)
errors = ErrorTracker(max_retries=3, max_tool_errors=2)


def handle_response_granular(response):
    """Same logic as handle_response_simple, with full control."""

    # Checkpoint 1: Is this response usable?
    result = validator.validate(response)

    if result.needs_retry:
        if result.nudge.kind == "tool_arg_validation":
            errors.record_result(success=False)
            exhausted = errors.tool_errors_exhausted
        else:
            errors.record_retry()
            exhausted = errors.retries_exhausted
        if exhausted:
            return "FATAL: too many consecutive bad responses"
        return f"retry: {result.nudge.content[:80]}..."

    errors.reset_retries()

    # Checkpoint 2: Is the model skipping required steps?
    step_check = enforcer.check(result.tool_calls)

    if step_check.needs_nudge:
        return f"step_blocked: {step_check.nudge.content[:80]}..."

    # Execute tools yourself, then sync up.
    tool_calls = result.tool_calls
    executed = [tc.name for tc in tool_calls]  # your execution here
    for name in executed:
        enforcer.record(name)
    errors.reset_errors()
    enforcer.reset_premature()

    done = enforcer.terminal_reached(tool_calls)
    return f"executed {executed}" + (" -- DONE" if done else "")


# =====================================================================
# Demo
# =====================================================================

EXAMPLES_WORKFLOW = [
    ("Model returns plain text (no tool call)",
     RichMessage(MessageRole.ASSISTANT, [TextPart("I think the answer is 42.")])),
    ("Model jumps to terminal tool, skipping required steps",
     RichMessage(MessageRole.ASSISTANT, [ToolCallPart("answer", ValidArguments({"text": "42"}))])),
    ("Model calls first required step",
     RichMessage(MessageRole.ASSISTANT, [ToolCallPart("search", ValidArguments({"query": "meaning of life"}))])),
    ("Model calls second required step",
     RichMessage(MessageRole.ASSISTANT, [ToolCallPart("lookup", ValidArguments({"id": "result-1"}))])),
    ("Model calls terminal tool (steps now satisfied)",
     RichMessage(MessageRole.ASSISTANT, [ToolCallPart("answer", ValidArguments({"text": "The answer is 42."}))])),
]

# =====================================================================
# Part 3: Respond tool
#
# In tool-equipped sessions, the model must choose between calling a
# tool and responding with text.  Small models frequently get this
# wrong -- they produce text when they should call tools, or vice versa.
#
# The respond tool eliminates this ambiguity: the model calls
# respond(message="...") instead of producing bare text.  From forge's
# perspective, every response is a valid tool call -- no retries wasted
# on conversational turns, no accuracy loss on tool-calling turns.
#
# For WorkflowRunner consumers: use respond_definition() and the separate respond handler from forge.tools.
# For proxy consumers: opt in with --inject-respond-tool.
# For middleware consumers: include "respond" in tool_names and handle
# the respond tool call in your execution logic.
# =====================================================================

from forge.tools import RESPOND_TOOL_NAME

# Middleware with respond -- include it in tool_names
respond_guardrails = Guardrails(
    tool_names=["search", "lookup", "answer", RESPOND_TOOL_NAME],
    required_steps=["search", "lookup"],
    terminal_tool="answer",
)


def handle_response_with_respond(response):
    """Process a response that may include a respond() call."""

    result = respond_guardrails.check(response)

    if result.action == "fatal":
        return f"FATAL: {result.reason}"

    if result.action in ("retry", "tool_error", "step_blocked"):
        return f"{result.action}: {result.nudge.content[:80]}..."

    tool_calls = result.tool_calls

    # Check if the model called respond -- extract the message
    for tc in tool_calls:
        if tc.name == RESPOND_TOOL_NAME:
            message = tc.arguments.value.get("message", "")
            return f"MODEL SAYS: {message}"

    # Normal tool execution
    executed = [tc.name for tc in tool_calls]
    done = respond_guardrails.record(executed)
    return f"executed {executed}" + (" -- DONE" if done else "")


EXAMPLES_RESPOND = [
    ("Model produces bare text (retried -- respond not used)",
     RichMessage(MessageRole.ASSISTANT, [TextPart("Hello, how can I help?")])),
    ("Model correctly uses respond for chat",
     RichMessage(MessageRole.ASSISTANT, [ToolCallPart("respond", ValidArguments({"message": "Hello! How can I help you?"}))])),
    ("Model calls a real tool",
     RichMessage(MessageRole.ASSISTANT, [ToolCallPart("search", ValidArguments({"query": "meaning of life"}))])),
    ("Model calls respond after completing work",
     RichMessage(MessageRole.ASSISTANT, [ToolCallPart("respond", ValidArguments({"message": "I found the answer: 42."}))])),
]


if __name__ == "__main__":
    print("=== Part 1: Simple API (Guardrails) ===")
    for i, (desc, response) in enumerate(EXAMPLES_WORKFLOW, 1):
        print(f"\n{i}. {desc}")
        print(f"   {handle_response_simple(response)}")

    # Reset granular state for Part 2 demo
    enforcer._tracker.completed_steps.clear()
    enforcer._premature_attempts = 0
    errors._consecutive_retries = 0

    print("\n\n=== Part 2: Granular API (individual components) ===")
    for i, (desc, response) in enumerate(EXAMPLES_WORKFLOW, 1):
        print(f"\n{i}. {desc}")
        print(f"   {handle_response_granular(response)}")

    print("\n\n=== Part 3: Respond tool ===")
    for i, (desc, response) in enumerate(EXAMPLES_RESPOND, 1):
        print(f"\n{i}. {desc}")
        print(f"   {handle_response_with_respond(response)}")
