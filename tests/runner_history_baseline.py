"""Copied Runner observations with canonical-only harness bindings.

Source: forge-primitives-baselines/pre-2026-11-22-29b1370/supplement_cases.py
Capture revision: 29b1370db67e2f7ceab3b25e978913e7841009e5, supplement-v3.
Original 79 observations and 22 additions are never modified.
Run from repo root: python -m tests.runner_history_baseline --output <directory>
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path

from forge import (ChunkType, StreamChunk, ContextManager, NoCompact, WorkflowRunner,
                   ToolDefinition, Workflow, RichMessage, MessageRole, TextPart,
                   ReasoningPart, ToolCallPart, ToolResultPart, ValidArguments)
from forge.core.codecs import render_openai, replay_messages
from forge.clients.base import InferenceRequest, format_tool

BASELINE = Path(r"C:\Users\antoi\Documents\forge-primitives-baselines\pre-2026-11-22-29b1370")


def plain(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def seed():
    return [RichMessage(MessageRole.SYSTEM, [TextPart("Use tools.")]),
            RichMessage(MessageRole.USER, [TextPart("Do the task.")])]


class RecordingContext(ContextManager):
    def __init__(self, strategy=None, budget=10000, **kwargs):
        self.history = []
        self.passes = []
        self.events = []
        super().__init__(strategy or NoCompact(), budget, on_compact=lambda event: self.events.append(plain(event)), **kwargs)

    def maybe_compact(self, state, **kwargs):
        self.history = state.messages
        before = plain(state.messages)
        result = super().maybe_compact(state, **kwargs)
        self.passes.append({"before": before, "after": plain(result.messages)})
        return result


class ScriptedClient:
    api_format = "openai"

    def __init__(self, replies):
        self.replies = copy.deepcopy(replies)
        self.returned_replies = []
        self.requests = []

    def render_request(self, state, tools, metadata, *, stream=False):
        return InferenceRequest({"messages": render_openai(replay_messages(state, metadata.reasoning_replay)),
                                 "tools": [format_tool(t) for t in tools], "stream": stream}, "https://mock/chat")

    async def send(self, request):
        self.requests.append(plain(request.payload))
        if not self.replies:
            raise AssertionError("scripted fixture exhausted")
        reply = self.replies.pop(0)
        self.returned_replies.append(plain(reply))
        return reply

    async def send_stream(self, request):
        yield StreamChunk(ChunkType.TEXT_DELTA, content="partial-visible")
        result = await self.send(request)
        yield StreamChunk(ChunkType.FINAL, response=result)


SIMPLE_SCHEMA = {"type": "object", "properties": {"path": {"type": "string"}}, "additionalProperties": False}


async def runner_case(kind):
    executed, emitted, chunks = [], [], []
    attempts = 0

    def read(path="a"):
        nonlocal attempts
        executed.append({"tool": "read", "path": path})
        attempts += 1
        if kind == "execution-retry" and attempts == 1:
            raise ValueError("controlled tool failure")
        return {"path": path, "content": "R" * (450 if kind == "compaction" else 12)}

    def use(path="a"):
        executed.append({"tool": "use", "path": path})
        return "used:" + path

    def finish(path="a"):
        executed.append({"tool": "finish", "path": path})
        return {"finished": path}

    replies = {
        "success": [RichMessage(MessageRole.ASSISTANT, [ToolCallPart("read", ValidArguments({"path": "a"})), ToolCallPart("read", ValidArguments({"path": "b"}))]), RichMessage(MessageRole.ASSISTANT, [ToolCallPart("finish", ValidArguments({}))])],
        "seeded-history": [RichMessage(MessageRole.ASSISTANT, [ToolCallPart("read", ValidArguments({"path": "a"}))]), RichMessage(MessageRole.ASSISTANT, [ToolCallPart("finish", ValidArguments({}))])],
        "execution-retry": [RichMessage(MessageRole.ASSISTANT, [ToolCallPart("read", ValidArguments({"path": "a"}))]), RichMessage(MessageRole.ASSISTANT, [ToolCallPart("read", ValidArguments({"path": "a"}))]), RichMessage(MessageRole.ASSISTANT, [ToolCallPart("finish", ValidArguments({}))])],
        "step": [RichMessage(MessageRole.ASSISTANT, [ToolCallPart("finish", ValidArguments({}))]), RichMessage(MessageRole.ASSISTANT, [ToolCallPart("read", ValidArguments({}))]), RichMessage(MessageRole.ASSISTANT, [ToolCallPart("finish", ValidArguments({}))])],
        "prerequisite": [RichMessage(MessageRole.ASSISTANT, [ToolCallPart("read", ValidArguments({"path": "b"}))]), RichMessage(MessageRole.ASSISTANT, [ToolCallPart("use", ValidArguments({"path": "a"}))]), RichMessage(MessageRole.ASSISTANT, [ToolCallPart("read", ValidArguments({"path": "a"}))]), RichMessage(MessageRole.ASSISTANT, [ToolCallPart("use", ValidArguments({"path": "a"}))]), RichMessage(MessageRole.ASSISTANT, [ToolCallPart("finish", ValidArguments({}))])],
        "rescue": [RichMessage(MessageRole.ASSISTANT, [TextPart('{"tool":"read","args":{"path":"a"}}')]), RichMessage(MessageRole.ASSISTANT, [ToolCallPart("finish", ValidArguments({}))])],
        "stream": [RichMessage(MessageRole.ASSISTANT, [ToolCallPart("read", ValidArguments({}))]), RichMessage(MessageRole.ASSISTANT, [ToolCallPart("finish", ValidArguments({}))])],
        "compaction": [RichMessage(MessageRole.ASSISTANT, [ReasoningPart("old reasoning"), ToolCallPart("read", ValidArguments({}))]), RichMessage(MessageRole.ASSISTANT, [ToolCallPart("read", ValidArguments({"path": "b"}))]), RichMessage(MessageRole.ASSISTANT, [ToolCallPart("finish", ValidArguments({}))])],
    }[kind]
    tools = {name: ToolDefinition(name, name, SIMPLE_SCHEMA) for name in ("read", "use", "finish")}
    workflow = Workflow(
        name="baseline", description="behavior capture", tools=tools,
        required_steps=["read"], terminal_tool="finish", system_prompt_template="Use tools in order.",
        handlers={"read": read, "use": use, "finish": finish},
        prerequisites={"use": [{"tool": "read", "match_arg": "path"}]},
    )
    if kind == "compaction":
        from forge.context.strategies import TieredCompact
        context = RecordingContext(TieredCompact(keep_recent=1, compact_threshold=.4), 160)
    else:
        context = RecordingContext()
    client = ScriptedClient(replies)

    async def on_chunk(chunk):
        chunks.append({"chunk": plain(chunk), "executed_so_far": plain(executed)})

    runner = WorkflowRunner(client, context, max_iterations=12, stream=kind == "stream",
                            on_chunk=on_chunk, on_message=lambda m: emitted.append(plain(m)), reasoning_replay="full")
    initial = None
    if kind == "seeded-history":
        initial = seed() + [
            RichMessage(MessageRole.ASSISTANT, [ToolCallPart("read", ValidArguments({"path": "previous"}), "caller-history-id")], step_index=0),
            RichMessage(MessageRole.TOOL, [ToolResultPart("caller-history-id", [TextPart("previous result")])], step_index=0, extra_fields={"name": "read"}),
            RichMessage(MessageRole.USER, [TextPart("new user turn")]),
        ]
    result = await runner.run(workflow, "Perform the workflow.", initial_messages=initial)
    assert result == {"finished": "a"}, "fixture did not reach its terminal result"
    return {"return_value": result, "initial_messages": plain(initial), "executions": executed, "emitted": emitted,
            "history_after": plain(context.history), "requests": client.requests,
            "returned_replies": client.returned_replies,
            "compaction_passes": context.passes, "compaction_events": context.events, "stream_chunks": chunks}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    before = {row["id"]: row for row in (json.loads(line) for line in (BASELINE / "supplement-v3/observations.jsonl").read_text(encoding="utf-8").splitlines())}
    observations, comparisons = [], []
    for kind in ("success", "seeded-history", "execution-retry", "step", "prerequisite", "rescue", "stream", "compaction"):
        identity = "R-" + kind
        after = await runner_case(kind)
        observations.append({"id": identity, "observation": after})
        prior = before[identity]["observation"]
        stable = [key for key in ("return_value", "executions") if prior[key] == after[key]]
        assert len(stable) == 2, identity
        assert len(prior["requests"]) == len(after["requests"]), identity
        expected_messages = copy.deepcopy([r["messages"] for r in prior["requests"]])
        if kind == "compaction":
            # CP-06: reasoning is a typed sibling field, no longer visible text.
            assert expected_messages[1][2]["content"] == "old reasoning"
            expected_messages[1][2]["reasoning_content"] = expected_messages[1][2]["content"]
            expected_messages[1][2]["content"] = ""
            # One assistant reply now holds reasoning and calls. The policies
            # and accounting still reach the same phases/token observations.
            expected_events = copy.deepcopy(prior["compaction_events"])
            assert [(e["messages_before"], e["messages_after"]) for e in expected_events] == [(5, 5), (7, 5)]
            expected_events[0].update(messages_before=4, messages_after=4)
            expected_events[1].update(messages_before=6)
            assert after["compaction_events"] == expected_events
        assert expected_messages == [r["messages"] for r in after["requests"]], identity
        assert [row["executed_so_far"] for row in prior["stream_chunks"]] == [row["executed_so_far"] for row in after["stream_chunks"]], identity
        old_ids = [m["tool_call_id"] for m in prior["history_after"] if m.get("tool_call_id")]
        new_ids = [p["call_id"] for m in after["history_after"] for p in m["parts"] if "outcome" in p]
        assert old_ids == new_ids, (identity, old_ids, new_ids)
        old_steps = [m["metadata"]["step_index"] for m in prior["history_after"]]
        assert old_steps == [m["step_index"] for m in after["history_after"]], identity
        comparisons.append({"id": identity, "classification": "unchanged behavior; approved rich API/history migration", "unchanged": stable + ["request count", "wire messages except typed reasoning sibling" if kind == "compaction" else "wire messages", "stream execution ordering", "result call IDs", "step grouping"],
                            **({"approved_changes": ["typed reasoning sibling rendering (CP-06)", "one assistant aggregate changes event message counts"], "unchanged_context": ["phase 3", "tokens 132->132 and 252->129", "protected recent iteration", "call skeletons"]} if kind == "compaction" else {})})
    (args.output / "observations.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
    result = {"original_source_sha256": hashlib.sha256((BASELINE / "supplement_cases.py").read_bytes()).hexdigest(), "capture_revision": "29b1370db67e2f7ceab3b25e978913e7841009e5", "comparisons": comparisons}
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
