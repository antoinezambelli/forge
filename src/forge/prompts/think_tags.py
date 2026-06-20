"""Thinking/reasoning tag parsing shared across client adapters.

Reasoning models wrap their chain-of-thought in delimiter tags. When the
backend's reasoning parser is absent — or doesn't split a given model's output
into a dedicated field — that thinking arrives inline in the message
``content`` instead. This module is the single source of truth for detecting
and extracting those blocks, used by the client adapters (to populate
``ToolCall.reasoning`` and to clean ``TextResponse`` content) and by the
prompt-rescue path in ``templates`` (to strip thinking before parsing a
rehearsed tool call).

Supported delimiters:
  - ``[THINK]...[/THINK]``  — Mistral (Ministral Reasoning)
  - ``<think>...</think>``  — Qwen3, DeepSeek

Extend ``THINK_TAG_RE`` when adding a new model family. If a model
library/registry is added later, move these patterns into per-model profiles
instead of hard-coding here.
"""

from __future__ import annotations

import re

THINK_TAG_RE = re.compile(
    r"\[THINK\](.*?)\[/THINK\]|<think>(.*?)</think>", re.DOTALL
)


def extract_think_tags(text: str) -> tuple[str, str]:
    """Split thinking blocks out of ``text``.

    Returns ``(reasoning, remaining_content)``: the concatenated thinking
    blocks (joined by blank lines) and the text with those blocks removed and
    stripped. When no tags are present, ``reasoning`` is the empty string and
    ``remaining_content`` is the original text unchanged.
    """
    reasoning_parts: list[str] = []
    remaining = text
    for m in THINK_TAG_RE.finditer(text):
        # group(1) is the [THINK] body, group(2) is the <think> body.
        content = (m.group(1) or m.group(2) or "").strip()
        reasoning_parts.append(content)
    if reasoning_parts:
        remaining = THINK_TAG_RE.sub("", text).strip()
    return "\n\n".join(reasoning_parts), remaining
