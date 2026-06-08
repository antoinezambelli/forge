"""Resume-key behavior for the reasoning_replay eval axis (batch_eval).

reasoning_replay is part of the canonical run key: distinct policies
(none / keep-last / full) on the same model+scenario are independent runs
and must not collide in resume counting, or a multi-policy sweep would
under-count and skip work it never actually ran.
"""

from __future__ import annotations

import json

import pytest

from forge.core.reasoning import DEFAULT_REASONING_REPLAY

import tests.eval.batch_eval as batch_eval
from tests.eval.batch_eval import (
    BatchConfig,
    _compute_cost,
    _count_completed_runs,
    _run_key,
    _run_result_to_row,
    run_batch,
)
from tests.eval.eval_runner import RunResult
from tests.eval.scenarios import basic_2step


def _row(model: str, scenario: str, reasoning_replay: str) -> dict:
    """Build a JSONL row via the production path for a given policy."""
    cfg = BatchConfig(model=model, backend="llamaserver", mode="native", think=None)
    res = RunResult(
        scenario_name=scenario,
        completeness=True,
        iterations_used=3,
        accuracy=True,
        messages=None,
    )
    return _run_result_to_row(
        res, cfg, basic_2step, run_idx=1,
        ablation_name="reforged", reasoning_replay=reasoning_replay,
    )


def test_run_key_distinguishes_reasoning_replay() -> None:
    base = dict(
        model="m", backend="llamaserver", mode="native",
        ablation_name="reforged", tool_choice="auto", scenario="s",
    )
    k_none = _run_key(reasoning_replay="none", **base)
    k_keep = _run_key(reasoning_replay="keep-last", **base)
    k_full = _run_key(reasoning_replay="full", **base)

    # All three policies yield distinct keys...
    assert len({k_none, k_keep, k_full}) == 3
    # ...and the key is stable for the same inputs.
    assert _run_key(reasoning_replay="none", **base) == k_none
    assert "none" in k_none


def test_run_result_to_row_records_reasoning_replay() -> None:
    row = _row("M", "sc", "none")
    assert row["reasoning_replay"] == "none"

    # Default when the caller doesn't pass one (legacy callers / inert axis).
    cfg = BatchConfig(model="M", backend="llamaserver", mode="native", think=None)
    res = RunResult(scenario_name="sc", completeness=True, iterations_used=2, messages=None)
    default_row = _run_result_to_row(res, cfg, basic_2step, run_idx=1)
    assert default_row["reasoning_replay"] == DEFAULT_REASONING_REPLAY


@pytest.mark.asyncio
async def test_anthropic_batch_rows_record_selected_reasoning_replay(
    tmp_path, monkeypatch,
) -> None:
    """Anthropic rows must use the runtime policy, not the module default."""
    cfg = BatchConfig(
        model="claude-sonnet-4-6",
        backend="anthropic",
        mode="native",
        think=True,
    )
    output = tmp_path / "results.jsonl"

    monkeypatch.setattr(batch_eval, "ALL_SCENARIOS", [basic_2step])
    monkeypatch.setattr(batch_eval, "_build_client", lambda config, models_dir: object())

    async def fake_run_with_timeout(client, scenario, eval_config, ablation):
        assert eval_config.reasoning_replay == "none"
        return RunResult(
            scenario_name=scenario.name,
            completeness=True,
            iterations_used=3,
            accuracy=True,
            messages=None,
        )

    monkeypatch.setattr(batch_eval, "_run_with_timeout", fake_run_with_timeout)

    await run_batch(
        configs=[cfg],
        runs_per_scenario=1,
        output_path=output,
        tags=["plumbing"],
        reasoning_replay="none",
    )

    row = json.loads(output.read_text().strip())
    assert row["model"] == "claude-sonnet-4-6"
    assert row["backend"] == "anthropic"
    assert row["reasoning_replay"] == "none"


def test_count_completed_runs_separates_policies(tmp_path) -> None:
    rows = [
        _row("M", "sc", "none"),
        _row("M", "sc", "none"),
        _row("M", "sc", "full"),
        _row("M", "sc", "keep-last"),
    ]
    # A pre-knob row (no reasoning_replay field) must fold into the default
    # policy, so a default-policy resume skips it and a different policy re-runs.
    legacy = _row("M", "sc", "keep-last")
    del legacy["reasoning_replay"]
    rows.append(legacy)

    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    counts = _count_completed_runs(path, ablation_name="reforged")

    def key(rr: str) -> str:
        return _run_key("M", "llamaserver", "native", "reforged", "auto", rr, "sc")

    assert counts[key("none")] == 2
    assert counts[key("full")] == 1
    # explicit keep-last + the legacy row defaulting to keep-last
    assert counts[key("keep-last")] == 2
    assert counts[key("none")] + counts[key("full")] + counts[key("keep-last")] == 5


def test_compute_cost_prices_cache_tokens() -> None:
    """Cache writes bill 1.25× and reads 0.1× of the input rate; uncached input
    and output keep their base rates. (sonnet: $3 input / $15 output per Mtok.)"""
    cost = _compute_cost(
        "claude-sonnet-4-6",
        input_tokens=1_000,
        output_tokens=500,
        cache_creation_tokens=2_000,
        cache_read_tokens=4_000,
    )
    expected = (
        1_000 * 3.0
        + 2_000 * 3.0 * 1.25
        + 4_000 * 3.0 * 0.1
        + 500 * 15.0
    ) / 1_000_000
    assert cost == expected

    # Back-compat: omitting cache args matches the old input+output formula.
    assert _compute_cost("claude-sonnet-4-6", 1_000, 500) == (
        1_000 * 3.0 + 500 * 15.0
    ) / 1_000_000

    # Opus 4.8 is priced (placeholder rate), not an unknown-model 0.0.
    assert _compute_cost("claude-opus-4-8", 1_000, 0) > 0


def test_run_result_to_row_emits_cache_tokens() -> None:
    cfg = BatchConfig(model="claude-sonnet-4-6", backend="anthropic", mode="native", think=None)
    res = RunResult(
        scenario_name="sc",
        completeness=True,
        iterations_used=3,
        accuracy=True,
        messages=None,
        input_tokens=1_000,
        output_tokens=500,
        cache_creation_tokens=2_000,
        cache_read_tokens=4_000,
    )
    row = _run_result_to_row(res, cfg, basic_2step, run_idx=1)

    assert row["cache_creation_input_tokens"] == 2_000
    assert row["cache_read_input_tokens"] == 4_000
    assert row["cost_usd"] == round(
        _compute_cost("claude-sonnet-4-6", 1_000, 500, 2_000, 4_000), 6
    )
