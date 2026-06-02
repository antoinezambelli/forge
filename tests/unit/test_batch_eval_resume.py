"""Resume-key behavior for the reasoning_replay eval axis (batch_eval).

reasoning_replay is part of the canonical run key: distinct policies
(none / keep-last / full) on the same model+scenario are independent runs
and must not collide in resume counting, or a multi-policy sweep would
under-count and skip work it never actually ran.
"""

from __future__ import annotations

import json

from forge.core.reasoning import DEFAULT_REASONING_REPLAY

from tests.eval.batch_eval import (
    BatchConfig,
    _count_completed_runs,
    _run_key,
    _run_result_to_row,
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
