"""Port of the six supplement-v3 context observations, original capture 29b1370.

Original inputs, budgets and captures are retained. Only canonical bindings and
explicit comparison assertions are added; this is not a production projection.
Run: python -m tests.context_history_baseline --output <directory>
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
from forge import (ConversationState, RichMessage, MessageRole, TextPart,
                   ReasoningPart, ToolCallPart, ToolResultPart, ValidArguments,
                   NoCompact, SlidingWindowCompact, TieredCompact)
from tests.runner_history_baseline import BASELINE, RecordingContext, seed, plain


def compaction_history():
    return ConversationState(seed() + [
        RichMessage(MessageRole.ASSISTANT, [ReasoningPart("R" * 300)], 0),
        RichMessage(MessageRole.ASSISTANT, [ToolCallPart("read", ValidArguments({"path": "a"}), "old-call")], 0),
        RichMessage(MessageRole.TOOL, [ToolResultPart("old-call", [TextPart("V" * 450)])], 0, extra_fields={"name": "read"}),
        RichMessage(MessageRole.USER, [TextPart("N" * 60)], 0, "retry"),
        RichMessage(MessageRole.ASSISTANT, [TextPart("T" * 200)], 0),
        RichMessage(MessageRole.ASSISTANT, [ToolCallPart("read", ValidArguments({}), "recent-call-a"), ToolCallPart("read", ValidArguments({}), "recent-call-b")], 1),
        RichMessage(MessageRole.TOOL, [ToolResultPart("recent-call-a", [TextPart("A" * 40)])], 1, extra_fields={"name": "read"}),
        RichMessage(MessageRole.TOOL, [ToolResultPart("recent-call-b", [TextPart("B" * 40)])], 1, extra_fields={"name": "read"}),
    ])


async def compact_case(kind):
    history = compaction_history()
    if kind == "no-compact":
        context = RecordingContext(NoCompact(), 20)
    elif kind == "sliding":
        context = RecordingContext(SlidingWindowCompact(keep_recent=1, compact_threshold=.4), 200)
    else:
        budget = {"phase1": 400, "phase2": 220, "phase3": 120, "usage-invalidation": 400}[kind]
        context = RecordingContext(TieredCompact(keep_recent=1, phase_thresholds=(.4, .65, .85)), budget)
    before = plain(history.messages)
    usage_observation = None
    if kind == "usage-invalidation":
        context.update_token_count(1)
        before_invalidation = context.maybe_compact(history)
        usage_observation = {"reported_tokens": context.estimate_tokens(history), "history": plain(before_invalidation.messages)}
        context.invalidate_usage()
    fallback = context.estimate_tokens(history)
    compacted = context.maybe_compact(history, step_index=2)
    assert compacted.messages[:2] == history.messages[:2], "protected headers changed"
    assert compacted.messages[-3:] == history.messages[-3:], "recent parallel-call turn changed"
    if kind in {"phase1", "phase2", "phase3"}:
        assert context.events[-1]["phase_reached"] == int(kind[-1]), "fixture did not exercise the named phase"
    return {"before": before, "after": plain(compacted.messages), "events": context.events,
            "fallback_tokens": fallback, "after_tokens": context.estimate_tokens(compacted),
            "usage_before_invalidation": usage_observation}



def compare_history(old, new, *, truncated=False):
    assert len(old) == len(new)
    for previous, current in zip(old, new):
        assert previous["role"] == current["role"]
        assert previous["metadata"]["step_index"] == current["step_index"]
        if previous["tool_calls"]:
            assert [(p["name"], p["args"], p["call_id"]) for p in previous["tool_calls"]] == [
                (p["name"], p["arguments"]["value"], p["call_id"]) for p in current["parts"]
            ]
            assert previous["content"] == ""
        else:
            assert len(current["parts"]) == 1
            part = current["parts"][0]
            if previous["metadata"]["type"] == "tool_result":
                assert part["content"][0]["text"] == previous["content"]
                assert part["outcome"] == "success"
                if truncated and "[Truncated" in previous["content"]:
                    # The old constructor discarded pairing/name; CP-07 fixes it.
                    assert previous["tool_call_id"] is None and previous["tool_name"] is None
                    assert part["call_id"] == "old-call"
                    assert current["extra_fields"] == {"name": "read"}
                else:
                    assert part["call_id"] == previous["tool_call_id"]
                    assert current["extra_fields"] == {"name": previous["tool_name"]}
            else:
                assert part["text"] == previous["content"]
            if previous["metadata"]["type"] == "retry_nudge":
                assert current["forge_purpose"] == "retry"


async def main():
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    before = {row["id"]: row["observation"] for row in map(json.loads, (BASELINE / "supplement-v3/observations.jsonl").read_text(encoding="utf-8").splitlines())}
    observations, comparisons = [], []
    for kind in ("no-compact", "sliding", "phase1", "phase2", "phase3", "usage-invalidation"):
        identity = "C-" + kind
        after = await compact_case(kind)
        prior = before[identity]
        compare_history(prior["before"], after["before"])
        compare_history(prior["after"], after["after"], truncated=True)
        for key in ("events", "fallback_tokens", "after_tokens"):
            assert prior[key] == after[key], (identity, key, prior[key], after[key])
        if kind == "usage-invalidation":
            assert prior["usage_before_invalidation"]["reported_tokens"] == after["usage_before_invalidation"]["reported_tokens"] == 1
            compare_history(prior["usage_before_invalidation"]["history"], after["usage_before_invalidation"]["history"])
        else:
            assert prior["usage_before_invalidation"] == after["usage_before_invalidation"] is None
        observations.append({"id": identity, "observation": after})
        comparisons.append({"id": identity,
            "classification": "unchanged policy/accounting; approved canonical API migration" + ("; approved retained correlation/name fix" if kind in {"phase1", "usage-invalidation"} else ""),
            "unchanged": ["original inputs/content", "thresholds/phases/events", "protected headers/recent parallel group", "content characters/4", "backend usage/invalidation"],
            "tokens": [after["fallback_tokens"], after["after_tokens"]],
            "phases": [e["phase_reached"] for e in after["events"]],
            "unresolved_differences": []})
    (args.output / "observations.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
    result = {"original_source_sha256": hashlib.sha256((BASELINE / "supplement_cases.py").read_bytes()).hexdigest(),
              "capture_revision": "29b1370db67e2f7ceab3b25e978913e7841009e5", "comparisons": comparisons}
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
