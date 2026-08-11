"""Generation-aware row capture and resume behavior for batch evals."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tests.eval.batch_eval as batch_eval
from tests.eval.batch_eval import (
    BatchConfig,
    _compute_cost,
    _preflight_recorded_runs,
    _run_key,
    _run_result_to_row,
    run_batch,
)
from tests.eval.eval_runner import RunResult
from tests.eval.outcomes import CANONICAL_DIALECT, LEGACY_DIALECT, OutcomeDialect
from tests.eval.scenarios import basic_2step


def _result(scenario: str = "basic_2step") -> RunResult:
    return RunResult(
        scenario_name=scenario,
        completed=True,
        iterations_used=3,
        correct=True,
        messages=None,
    )


def _row(
    model: str = "M",
    scenario: str = "basic_2step",
    reasoning_replay: str = "none",
    generation: int = 0,
    run_idx: int = 1,
    outcome_dialect: OutcomeDialect = CANONICAL_DIALECT,
) -> dict[str, Any]:
    """Build a JSONL row via the production serializer."""
    cfg = BatchConfig(model=model, backend="llamaserver", mode="native", think=None)
    return _run_result_to_row(
        _result(scenario), cfg, basic_2step, run_idx,
        generation=generation,
        ablation_name="reforged",
        reasoning_replay=reasoning_replay,
        outcome_dialect=outcome_dialect,
    )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _managed_config() -> BatchConfig:
    return BatchConfig(model="M", backend="ollama", mode="native", think=None)


def _anthropic_config() -> BatchConfig:
    return BatchConfig(
        model="claude-sonnet-4-6", backend="anthropic", mode="native", think=True
    )


def _install_inert_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeServer:
        client_base_url = "http://localhost:11434"

        async def restart(self):
            pass

        async def resolve_budget(self, *args, **kwargs):
            return 4096

        async def stop(self):
            pass

    async def fake_setup_backend(**kwargs):
        return FakeServer(), SimpleNamespace(budget_tokens=4096)

    monkeypatch.setattr(batch_eval, "ALL_SCENARIOS", [basic_2step])
    monkeypatch.setattr(batch_eval, "_check_model_available", lambda *args: None)
    monkeypatch.setattr(batch_eval, "setup_backend", fake_setup_backend)
    monkeypatch.setattr(batch_eval, "_build_client", lambda *args: object())

    async def fake_run_with_timeout(client, scenario, eval_config, ablation, run_timeout):
        return _result(scenario.name)

    monkeypatch.setattr(batch_eval, "_run_with_timeout", fake_run_with_timeout)


def _fail_if_backend_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        pytest.fail("backend construction was reached")

    monkeypatch.setattr(batch_eval, "setup_backend", fail)
    monkeypatch.setattr(batch_eval, "_build_client", fail)


def test_run_key_distinguishes_reasoning_replay() -> None:
    base = dict(
        model="m", backend="llamaserver", mode="native",
        ablation_name="reforged", tool_choice="auto",
        reasoning_level="default", scenario="s",
    )
    keys = {
        _run_key(reasoning_replay=policy, **base)
        for policy in ("none", "keep-last", "full")
    }
    assert len(keys) == 3


def test_run_key_distinguishes_reasoning_level() -> None:
    base = dict(
        model="m", backend="llamaserver", mode="native",
        ablation_name="reforged", tool_choice="auto",
        reasoning_replay="none", scenario="s",
    )
    assert _run_key(reasoning_level="default", **base) != _run_key(
        reasoning_level="high", **base
    )


def test_run_result_to_row_records_generation_and_replay() -> None:
    scratch = _row(reasoning_replay="none", generation=0)
    released = _row(reasoning_replay="full", generation=7)

    assert scratch["gen"] == 0
    assert scratch["reasoning_replay"] == "none"
    assert released["gen"] == 7
    assert released["reasoning_replay"] == "full"
    assert released["reasoning_level"] == "default"
    assert scratch["correct"] is True
    assert scratch["completed"] is True
    assert "accuracy" not in scratch
    assert "completeness" not in scratch


def test_run_result_to_row_requires_generation_keyword() -> None:
    cfg = BatchConfig(model="M", backend="llamaserver", mode="native", think=None)
    with pytest.raises(TypeError, match="generation"):
        _run_result_to_row(_result(), cfg, basic_2step, 1)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_mixed_managed_and_anthropic_configs_fail_before_preflight_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "results.jsonl"
    _fail_if_backend_reached(monkeypatch)
    monkeypatch.setattr(
        batch_eval,
        "_preflight_recorded_runs",
        lambda *args, **kwargs: pytest.fail("result preflight was reached"),
    )

    with pytest.raises(ValueError, match="managed backends.*anthropic"):
        await run_batch(
            configs=[_managed_config(), _anthropic_config()],
            runs_per_scenario=1,
            output_path=output,
            reasoning_replay="none",
            generation=4,
        )

    assert not output.exists()


@pytest.mark.asyncio
async def test_new_output_defaults_to_generation_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "results.jsonl"
    _install_inert_backend(monkeypatch)

    await run_batch(
        configs=[_managed_config()], runs_per_scenario=1, output_path=output
    )

    assert json.loads(output.read_text(encoding="utf-8"))["gen"] == 0


@pytest.mark.asyncio
async def test_managed_server_append_records_nonzero_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "results.jsonl"
    _install_inert_backend(monkeypatch)

    await run_batch(
        configs=[_managed_config()],
        runs_per_scenario=1,
        output_path=output,
        generation=5,
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["backend"] == "ollama"
    assert row["gen"] == 5


@pytest.mark.asyncio
async def test_empty_output_accepts_requested_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "results.jsonl"
    output.write_bytes(b"")
    _install_inert_backend(monkeypatch)

    await run_batch(
        configs=[_managed_config()], runs_per_scenario=1,
        output_path=output, generation=6,
    )

    assert json.loads(output.read_text(encoding="utf-8"))["gen"] == 6


@pytest.mark.asyncio
async def test_public_setup_receives_the_configuration_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_flags = ("--reasoning-format", "auto", "--no-mmap")
    setups: list[dict[str, Any]] = []

    _install_inert_backend(monkeypatch)

    class Server:
        client_base_url = "http://localhost:8080/v1"

        async def stop(self):
            pass

    async def setup(**kwargs):
        setups.append(kwargs)
        return Server(), SimpleNamespace(budget_tokens=4096)

    monkeypatch.setattr(batch_eval, "setup_backend", setup)
    config = BatchConfig(
        model="M",
        backend="llamaserver",
        mode="native",
        think=None,
        gguf_filename="M.gguf",
        server_recipe=batch_eval._BatchServerRecipe(recipe_flags),
    )

    await run_batch(
        configs=[config],
        runs_per_scenario=1,
        output_path=tmp_path / "results.jsonl",
    )

    assert setups[0]["extra_flags"] == list(recipe_flags)


@pytest.mark.asyncio
async def test_same_generation_resume_uses_exact_next_run_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "results.jsonl"
    first = _row(generation=3, run_idx=1)
    first["backend"] = "ollama"
    _write_rows(output, [first])
    _install_inert_backend(monkeypatch)

    await run_batch(
        configs=[_managed_config()], runs_per_scenario=2,
        output_path=output, reasoning_replay="none", generation=3,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["run"] for row in rows] == [1, 2]
    assert {row["gen"] for row in rows} == {3}


@pytest.mark.asyncio
async def test_resume_safely_separates_unterminated_final_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "results.jsonl"
    config = _managed_config()
    first = _row(model=config.model, generation=4, run_idx=1)
    first["backend"] = config.backend
    first["mode"] = config.mode
    original = json.dumps(first).encode("utf-8")
    output.write_bytes(original)
    _install_inert_backend(monkeypatch)

    await run_batch(
        configs=[config], runs_per_scenario=2,
        output_path=output, reasoning_replay="none", generation=4,
    )

    appended = output.read_bytes()
    assert appended.startswith(original + b"\n")
    rows = [json.loads(line) for line in appended.splitlines()]
    assert [row["run"] for row in rows] == [1, 2]
    assert {row["gen"] for row in rows} == {4}


def test_legacy_missing_replay_counts_as_full_not_none(tmp_path: Path) -> None:
    legacy = _row(
        reasoning_replay="none", outcome_dialect=LEGACY_DIALECT
    )
    del legacy["reasoning_replay"]
    del legacy["gen"]
    output = tmp_path / "legacy.jsonl"
    _write_rows(output, [legacy])

    counts = _preflight_recorded_runs(
        output, requested_generation=0
    ).recorded_counts

    def key(policy: str) -> str:
        return _run_key(
            "M", "llamaserver", "native", "reforged", "auto",
            policy, "default", "basic_2step",
        )

    assert counts[key("full")] == 1
    assert counts.get(key("none"), 0) == 0


@pytest.mark.asyncio
async def test_legacy_full_resume_skips_existing_generation_zero_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "legacy.jsonl"
    legacy = _row(
        reasoning_replay="full",
        outcome_dialect=LEGACY_DIALECT,
    )
    legacy["backend"] = "ollama"
    del legacy["reasoning_replay"]
    del legacy["gen"]
    _write_rows(output, [legacy])
    before = output.read_bytes()
    _install_inert_backend(monkeypatch)

    await run_batch(
        configs=[_managed_config()], runs_per_scenario=1,
        output_path=output, reasoning_replay="full", generation=0,
    )

    assert output.read_bytes() == before


@pytest.mark.asyncio
async def test_explicit_none_does_not_collide_with_legacy_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "legacy.jsonl"
    legacy = _row(outcome_dialect=LEGACY_DIALECT)
    legacy["backend"] = "ollama"
    del legacy["reasoning_replay"]
    del legacy["gen"]
    _write_rows(output, [legacy])
    _install_inert_backend(monkeypatch)

    await run_batch(
        configs=[_managed_config()], runs_per_scenario=1,
        output_path=output, reasoning_replay="none", generation=0,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert "reasoning_replay" not in rows[0]
    assert rows[1]["reasoning_replay"] == "none"
    assert rows[1]["gen"] == 0
    assert rows[1]["accuracy"] is True
    assert rows[1]["completeness"] is True
    assert "correct" not in rows[1]
    assert "completed" not in rows[1]


@pytest.mark.parametrize(
    "row",
    [
        {**_row(), "accuracy": True, "completeness": True},
        {key: value for key, value in _row().items() if key != "completed"},
        {
            **{
                key: value
                for key, value in _row().items()
                if key not in {"correct", "completed"}
            },
            "accuracy": True,
            "completed": True,
        },
        {**_row(), "correct": "yes"},
        {**_row(), "completed": 1},
        {**_row(), "validation_error": 123},
    ],
    ids=[
        "dual",
        "partial",
        "hybrid",
        "wrong-correct-type",
        "wrong-completed-type",
        "wrong-validation-error-type",
    ],
)
@pytest.mark.asyncio
async def test_outcome_dialect_errors_fail_before_backend_and_preserve_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    row: dict[str, Any],
) -> None:
    output = tmp_path / "invalid-outcome.jsonl"
    _write_rows(output, [row])
    before = output.read_bytes()
    _fail_if_backend_reached(monkeypatch)

    with pytest.raises(ValueError, match="outcome"):
        await run_batch(
            configs=[_managed_config()],
            runs_per_scenario=2,
            output_path=output,
            generation=0,
        )

    assert output.read_bytes() == before


@pytest.mark.parametrize(
    ("payload", "requested_generation", "message"),
    [
        (json.dumps(_row(generation=1)).encode() + b"\n", 2, "does not match"),
        (json.dumps({k: v for k, v in _row().items() if k != "gen"}).encode() + b"\n", 1, "does not match"),
        ((json.dumps(_row(generation=1)) + "\n" + json.dumps(_row(generation=2)) + "\n").encode(), 1, "mixed effective generations"),
        (b"{not-json}\n", 0, "malformed JSON"),
        (b"[]\n", 0, "must be an object"),
        (b"\xff\n", 0, "invalid UTF-8"),
        (
            json.dumps({k: v for k, v in _row().items() if k != "scenario"}).encode()
            + b"\n",
            0,
            "cannot build resume key",
        ),
        (json.dumps({**_row(), "gen": True}).encode() + b"\n", 0, "non-negative integer"),
        (json.dumps({**_row(), "gen": -1}).encode() + b"\n", 0, "non-negative integer"),
        (json.dumps({**_row(), "gen": 1.0}).encode() + b"\n", 0, "non-negative integer"),
        (json.dumps({**_row(), "gen": "1"}).encode() + b"\n", 0, "non-negative integer"),
        (json.dumps({**_row(), "gen": None}).encode() + b"\n", 0, "non-negative integer"),
    ],
    ids=[
        "requested-mismatch", "genless-nonzero", "mixed", "malformed-json",
        "non-object", "invalid-utf8", "missing-resume-field", "bool", "negative",
        "float", "string", "null",
    ],
)
@pytest.mark.asyncio
async def test_rejected_output_preflight_is_before_backend_and_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    requested_generation: int,
    message: str,
) -> None:
    output = tmp_path / "results.jsonl"
    output.write_bytes(payload)
    before = output.read_bytes()
    _fail_if_backend_reached(monkeypatch)

    with pytest.raises(ValueError, match=message):
        await run_batch(
            configs=[_managed_config()], runs_per_scenario=1,
            output_path=output, generation=requested_generation,
        )

    assert output.read_bytes() == before


@pytest.mark.asyncio
async def test_dry_run_still_rejects_generation_mismatch_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "results.jsonl"
    _write_rows(output, [_row(generation=2)])
    before = output.read_bytes()
    _fail_if_backend_reached(monkeypatch)

    with pytest.raises(ValueError, match="does not match"):
        await run_batch(
            configs=[_managed_config()], runs_per_scenario=1,
            output_path=output, generation=3, dry_run=True,
        )

    assert output.read_bytes() == before


@pytest.mark.parametrize("generation", [True, -1, 1.0, "1", None])
@pytest.mark.asyncio
async def test_invalid_programmatic_generation_creates_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation: Any,
) -> None:
    output = tmp_path / "not-created.jsonl"
    _fail_if_backend_reached(monkeypatch)

    with pytest.raises(ValueError, match="non-negative integer"):
        await run_batch(
            configs=[_managed_config()], runs_per_scenario=1,
            output_path=output, generation=generation,
        )

    assert not output.exists()


def test_compute_cost_prices_cache_tokens() -> None:
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
    assert _compute_cost("claude-sonnet-4-6", 1_000, 500) == (
        1_000 * 3.0 + 500 * 15.0
    ) / 1_000_000
    assert _compute_cost("claude-opus-4-8", 1_000, 0) > 0


def test_run_result_to_row_emits_cache_tokens() -> None:
    cfg = BatchConfig(
        model="claude-sonnet-4-6", backend="anthropic", mode="native", think=None
    )
    result = RunResult(
        scenario_name="basic_2step",
        completed=True,
        iterations_used=3,
        correct=True,
        messages=None,
        input_tokens=1_000,
        output_tokens=500,
        cache_creation_tokens=2_000,
        cache_read_tokens=4_000,
    )
    row = _run_result_to_row(result, cfg, basic_2step, 1, generation=0)

    assert row["cache_creation_input_tokens"] == 2_000
    assert row["cache_read_input_tokens"] == 4_000
    assert row["cost_usd"] == round(
        _compute_cost("claude-sonnet-4-6", 1_000, 500, 2_000, 4_000), 6
    )
