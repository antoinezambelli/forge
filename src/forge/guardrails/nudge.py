"""Lightweight nudge message returned by guardrail components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Nudge:
    """A message to inject into conversation history.

    Returned by guardrail components when the model needs correction.
    The consumer maps this to their framework's message format::

        # OpenAI-style
        messages.append({"role": nudge.role, "content": nudge.content})

        # LangChain
        messages.append(HumanMessage(content=nudge.content))

    Attributes:
        role: Message role for injection ("user", "system", or "tool").
        content: The nudge text.
        kind: Identifies what generated the nudge ("retry", "unknown_tool",
            "step"). Useful for logging/metrics and for WorkflowRunner to
            map back to MessageType for compaction prioritization.
        tier: Escalation level for step nudges (0 = N/A, 1-3 = escalating).
    """

    role: str
    content: str
    kind: str
    tier: int = 0


# Nudge kinds that drain the tool-error BUDGET (record_result, max_tool_errors)
# rather than the retry budget. Malformed args are conceptually "tool called
# with bad inputs" — same family as a runtime FileNotFoundError. Shared by
# run_inference (core) and the Guardrails facade so both account identically.
TOOL_ERROR_KINDS: frozenset[str] = frozenset({"tool_arg_validation"})

# Nudge kinds emitted on the tool-result CHANNEL (role="tool") — the model
# called a tool incorrectly (bad name or bad args), so the correction rides the
# canonical tool channel rather than a user nudge. Maps to the Guardrails
# facade's action="tool_error". Superset of TOOL_ERROR_KINDS: an unknown-tool
# call rides the tool channel but only drains the retry budget.
TOOL_CHANNEL_KINDS: frozenset[str] = frozenset({"unknown_tool", "tool_arg_validation"})
