"""Semantic regression tests for the public eval metric vocabulary."""

from __future__ import annotations

import pytest

from tests.eval import report, significance


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "model": "model-q4",
        "backend": "llamaserver",
        "mode": "native",
        "scenario": "basic_2step",
        "run": 0,
        "correct": None,
        "completed": False,
        "validation_error": None,
        "iterations": 2,
        "ideal_iterations": 2,
        "wasted_calls": None,
        "elapsed_s": 1.0,
    }
    row.update(overrides)
    return row


def test_report_keeps_all_three_denominators_independent() -> None:
    scenario_rows = [
        _row(run=0, correct=True, completed=False),
        {
            **_row(run=1),
            "accuracy": False,
            "completeness": True,
            "validate_error": None,
        },
        _row(run=2, correct=None, completed=True),
        {
            **_row(run=3),
            "accuracy": False,
            "completeness": False,
            "validate_error": "ValueError",
        },
    ]
    for row in scenario_rows[1::2]:
        row.pop("correct")
        row.pop("completed")
        row.pop("validation_error")

    metrics = report.compute_config_metrics(
        report.ConfigKey("model-q4", "llamaserver", "native"),
        {"basic_2step": scenario_rows},
        scenarios=["basic_2step"],
    )

    assert metrics.attempted_count == 4
    assert metrics.correct_count == 1
    assert metrics.validated_count == 2
    assert metrics.completed_count == 2
    assert metrics.validation_error_count == 1
    assert metrics.score == pytest.approx(1 / 4)
    assert metrics.validated_accuracy == pytest.approx(1 / 2)
    assert metrics.completion_rate == pytest.approx(1 / 2)
    assert metrics.efficiency == 0.0  # correct-but-incomplete is not efficient work

    dashboard = report._metrics_to_json_row(metrics, ["basic_2step"])
    assert dashboard["score"] == 25.0
    assert dashboard["validatedAccuracy"] == 50.0
    assert dashboard["completionRate"] == 50.0
    assert dashboard["attemptedCount"] == 4
    assert dashboard["correctCount"] == 1
    assert dashboard["validatedCount"] == 2
    assert dashboard["completedCount"] == 2
    assert dashboard["scenarioAttempted"] == {"basic_2step": 4}
    assert "accuracy" not in dashboard
    assert "completeness" not in dashboard
    assert "scenarioRuns" not in dashboard


def test_report_labels_only_explicit_reasoning_levels() -> None:
    legacy = report.ConfigKey("model-q4", "llamaserver", "native")
    explicit_default = report.ConfigKey(
        "model-q4", "llamaserver", "native", reasoning_level="default"
    )

    assert legacy.short_label == explicit_default.short_label
    assert "@" not in legacy.short_label
    for level in ("low", "medium", "high", "xhigh"):
        key = report.ConfigKey(
            "model-q4", "llamaserver", "native", reasoning_level=level
        )
        assert f"@{level}]" in key.short_label


def test_report_extracts_known_quantizations() -> None:
    assert report.extract_quant("model-Q4_K_M") == "q4_K_M"
    assert report.extract_quant("model-Q4_K_XL") == "q4_K_XL"
    assert report.extract_quant("model-IQ3_XXS") == "iq3_XXS"
    assert report.extract_quant("model-Q8_0") == "q8_0"
    assert report.extract_quant("unquantized-model") == "n/a"


def test_significance_identity_does_not_pool_replay_or_reasoning() -> None:
    base = _row()
    none = significance.configuration_identity(
        {**base, "reasoning_replay": "none"}
    )
    full = significance.configuration_identity(
        {**base, "reasoning_replay": "full"}
    )
    high = significance.configuration_identity(
        {**base, "reasoning_replay": "none", "reasoning_level": "high"}
    )

    assert none != full
    assert none != high


def test_significance_reads_both_outcome_dialects() -> None:
    rows = [
        {**_row(run=0, correct=True, completed=True), "ablation": "reforged"},
        {**_row(run=1, correct=False, completed=True), "ablation": "reforged"},
        {
            **_row(run=0),
            "ablation": "bare",
            "accuracy": False,
            "completeness": True,
        },
        {
            **_row(run=1),
            "ablation": "bare",
            "accuracy": False,
            "completeness": True,
        },
    ]
    for row in rows[2:]:
        row.pop("correct")
        row.pop("completed")
        row.pop("validation_error")

    analysis = significance.analyze_config(rows)

    assert analysis[0]["ablation"] == "reforged"
    assert analysis[0]["score"] == 0.5
    assert analysis[1]["ablation"] == "bare"
    assert analysis[1]["score"] == 0.0
    assert analysis[1]["paired_n"] == 2
    assert analysis[1]["discordant_b"] == 1
    assert analysis[1]["discordant_c"] == 0


def test_rich_history_counts_parallel_calls_all_reasoning_and_distinct_errors(capsys):
    from forge import MessageRole, RichMessage, TextPart, ReasoningPart, ToolResultPart
    from tests.canonical import assistant, call
    from tests.eval.eval_runner import _verbose_printer
    from tests.eval.metrics import analyze_history

    history = [
        assistant(ReasoningPart("earlier"), call("fetch", {})),
        assistant(TextPart("visible"), ReasoningPart("one"), call("fetch", {}),
                  ReasoningPart("two"), call("submit", {})),
        RichMessage(MessageRole.USER, [TextPart("retry")], forge_purpose="retry"),
        RichMessage(MessageRole.TOOL, [ToolResultPart("a", [TextPart("[StepEnforcementError] blocked")], "error")], forge_purpose="step"),
        RichMessage(MessageRole.TOOL, [ToolResultPart("b", [TextPart("[PrerequisiteError] blocked")], "error")], forge_purpose="prerequisite"),
        RichMessage(MessageRole.TOOL, [ToolResultPart("c", [TextPart("[ToolArgValidationError] malformed")], "error")], forge_purpose="tool_arg_validation"),
        RichMessage(MessageRole.TOOL, [ToolResultPart("unknown", [TextPart("[UnknownTool] absent")], "error")], forge_purpose="unknown_tool"),
        RichMessage(MessageRole.TOOL, [ToolResultPart("d", [TextPart("[ToolResolutionError] absent")], "error")]),
        RichMessage(MessageRole.TOOL, [ToolResultPart("e", [TextPart("[ToolError] ValueError: broken")], "error")]),
        RichMessage(MessageRole.TOOL, [ToolResultPart("f", [TextPart("[ToolError] literal successful output")])]),
    ]
    stats = analyze_history(history)
    assert stats.total_tool_calls == 3
    assert stats.unique_tools_called == {"fetch", "submit"}
    assert stats.reasoning_messages == 3
    assert stats.retry_nudges == 3
    assert stats.step_nudges == stats.tool_errors == 1
    _verbose_printer(history[1])
    output = capsys.readouterr().out
    assert "PARALLEL 2" in output
    assert "fetch, submit" in output
    assert "[text]      visible" in output
    assert "[thinking]  one" in output
    assert "[thinking]  two" in output


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_counting_wrapper_delegates_exact_rendered_request_and_usage(stream):
    from forge import ConversationState, RequestMetadata, TokenUsage, ChunkType
    from tests.canonical import assistant, call
    from tests.conftest import MockClient
    from tests.eval.eval_runner import CountingClientWrapper

    class Client(MockClient):
        def render_request(self, *args, **kwargs):
            self.rendered = super().render_request(*args, **kwargs)
            return self.rendered

        def _next(self):
            self.last_usage = {0: TokenUsage(10, 2, 12, cache_creation_input_tokens=3, cache_read_input_tokens=4)}
            return super()._next()

    reply = assistant(call("submit", {}))
    client = Client([reply])
    wrapper = CountingClientWrapper(client)
    request = wrapper.render_request(ConversationState(), [], RequestMetadata(), stream=stream)
    assert request is client.rendered
    if stream:
        chunks = [chunk async for chunk in wrapper.send_stream(request)]
        assert chunks[-1].type == ChunkType.FINAL
        assert chunks[-1].response is reply
        assert client.send_stream_calls == [request]
    else:
        assert await wrapper.send(request) is reply
        assert client.send_calls == [request]
    assert wrapper.call_count == 1
    assert (wrapper.total_input_tokens, wrapper.total_output_tokens) == (10, 2)
    assert (wrapper.total_cache_creation_tokens, wrapper.total_cache_read_tokens) == (3, 4)
