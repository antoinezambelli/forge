"""Tests for budget_override in eval runner."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from forge.clients.base import ChunkType, StreamChunk
from forge.context.strategies import TieredCompact
from forge.core.workflow import LLMResponse, ToolCall, ToolSpec, TextResponse

from tests.eval.eval_runner import EvalConfig, RunResult, run_scenario
from tests.eval.scenarios import compaction_stress, basic_2step


class _MockClient:
    """Minimal client that returns a tool call on each send."""

    api_format: str = "ollama"

    def __init__(self, calls: list[ToolCall]) -> None:
        self._calls = list(calls)
        self._idx = 0

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        if self._idx < len(self._calls):
            tc = self._calls[self._idx]
            self._idx += 1
            return [tc]
        return TextResponse(content="stuck")

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        resp = await self.send(messages, tools)
        yield StreamChunk(type=ChunkType.FINAL, response=resp)

    async def get_context_length(self) -> int | None:
        return None


class TestBudgetOverride:
    """Verify that EvalConfig.budget_override overrides scenario.budget_tokens."""

    async def test_override_changes_budget(self) -> None:
        """budget_override should replace scenario.budget_tokens in ContextManager."""
        # compaction_stress has budget_tokens=2048
        assert compaction_stress.budget_tokens == 2048

        # Build a client that completes the scenario
        client = _MockClient([
            ToolCall(tool="fetch_sales_data", args={"quarter": 4, "year": 2024}),
            ToolCall(tool="analyze_sales", args={}),
            ToolCall(tool="report", args={"findings": "test"}),
        ])

        config = EvalConfig(
            runs_per_scenario=1,
            budget_override=16384,
        )

        result = await run_scenario(client, compaction_stress, config)
        assert result.completeness
        # With 16384 budget, no compaction should fire on short mock responses
        assert len(result.compaction_events) == 0

    async def test_no_override_uses_scenario_default(self) -> None:
        """Without budget_override, scenario.budget_tokens is used."""
        assert basic_2step.budget_tokens == 8192

        client = _MockClient([
            ToolCall(tool="get_country_info", args={"country": "France"}),
            ToolCall(tool="summarize", args={"content": "test"}),
        ])

        config = EvalConfig(
            runs_per_scenario=1,
            budget_override=None,
        )

        result = await run_scenario(client, basic_2step, config)
        assert result.completeness

    async def test_tight_budget_triggers_compaction(self) -> None:
        """A very tight budget with long responses should trigger compaction."""
        client = _MockClient([
            ToolCall(tool="fetch_sales_data", args={"quarter": 4, "year": 2024}),
            ToolCall(tool="analyze_sales", args={}),
            ToolCall(tool="report", args={"findings": "test"}),
        ])

        config = EvalConfig(
            runs_per_scenario=1,
            budget_override=64,  # Extremely tight — system prompt alone exceeds this
            strategy_overrides={"compaction": TieredCompact(keep_recent=2)},
        )

        result = await run_scenario(client, compaction_stress, config)
        assert result.completeness
        assert len(result.compaction_events) > 0
