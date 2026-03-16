"""Tests for BFCL report generation."""

import json
from pathlib import Path

import pytest


def _make_rows(
    model: str = "test-model",
    backend: str = "ollama",
    mode: str = "native",
    ablation: str = "reforged",
    categories: dict[str, list[bool]] | None = None,
) -> list[dict]:
    """Build mock JSONL rows.

    Args:
        categories: {category_name: [valid_1, valid_2, ...]}
    """
    if categories is None:
        categories = {
            "simple_python": [True, True, False],
            "multiple": [True, False],
        }
    rows = []
    for cat, valids in categories.items():
        for i, valid in enumerate(valids):
            rows.append({
                "model": model,
                "backend": backend,
                "mode": mode,
                "ablation": ablation,
                "category": cat,
                "test_id": f"{cat}_{i}",
                "valid": valid,
                "completed": True,
                "error_type": "" if valid else "test_error",
                "errors": [] if valid else ["test"],
                "elapsed_s": 1.0,
                "iterations": 2,
            })
    return rows


class TestBuildReport:
    """Test ASCII report generation."""

    def test_empty_rows(self):
        from tests.eval.bfcl.bfcl_report import build_report
        assert build_report([]) == "No data."

    def test_single_config_single_category(self):
        from tests.eval.bfcl.bfcl_report import build_report
        rows = _make_rows(categories={"simple_python": [True, True, False]})
        report = build_report(rows)
        assert "66.7%" in report  # 2/3
        assert "spy" in report  # column header

    def test_overall_accuracy_is_entry_level(self):
        """Overall Acc% = total valid / total entries, NOT average of categories."""
        from tests.eval.bfcl.bfcl_report import build_report
        # Category A: 1/1 = 100%, Category B: 0/9 = 0%
        # Average of averages would be 50%, entry-level is 1/10 = 10%
        rows = _make_rows(categories={
            "simple_python": [True],
            "multiple": [False] * 9,
        })
        report = build_report(rows)
        assert "10.0%" in report

    def test_two_configs_sorted_by_accuracy(self):
        """Higher accuracy config appears first."""
        from tests.eval.bfcl.bfcl_report import build_report
        rows_good = _make_rows(
            ablation="reforged",
            categories={"simple_python": [True, True, True]},
        )
        rows_bad = _make_rows(
            ablation="bare",
            categories={"simple_python": [True, False, False]},
        )
        report = build_report(rows_good + rows_bad)
        lines = report.strip().split("\n")
        # Find data rows (skip header/separator)
        data_lines = [l for l in lines if "[reforged]" in l or "[bare]" in l]
        assert len(data_lines) == 2
        assert "[reforged]" in data_lines[0]  # Higher accuracy first
        assert "[bare]" in data_lines[1]

    def test_missing_category_shows_dash(self):
        """Config missing a category shows dash in that column."""
        from tests.eval.bfcl.bfcl_report import build_report
        rows_a = _make_rows(
            ablation="reforged",
            categories={"simple_python": [True], "multiple": [True]},
        )
        rows_b = _make_rows(
            ablation="bare",
            categories={"simple_python": [True]},  # no multiple
        )
        report = build_report(rows_a + rows_b)
        assert "\u2014" in report

    def test_categories_derived_from_data(self):
        """Unknown categories still appear as columns."""
        from tests.eval.bfcl.bfcl_report import build_report
        rows = _make_rows(categories={"new_custom_category": [True, False]})
        report = build_report(rows)
        assert "new_" in report  # fallback: first 4 chars


class TestShortenModel:
    """Test model name shortening."""

    def test_strips_quant(self):
        from tests.eval.bfcl.bfcl_report import _shorten_model
        assert "q4_K_M" not in _shorten_model("ministral-3:8b-instruct-2512-q4_K_M")

    def test_strips_instruct_suffix(self):
        from tests.eval.bfcl.bfcl_report import _shorten_model
        result = _shorten_model("llama3.1:8b-instruct-q4_K_M")
        assert "instruct" not in result

    def test_strips_date_suffix(self):
        from tests.eval.bfcl.bfcl_report import _shorten_model
        result = _shorten_model("claude-haiku-4-5-20251001")
        assert "20251001" not in result


class TestBuildProgress:
    """Test progress summary."""

    def test_empty_rows(self):
        from tests.eval.bfcl.bfcl_report import build_progress
        assert build_progress([]) == "No data."

    def test_shows_category_counts(self):
        from tests.eval.bfcl.bfcl_report import build_progress
        rows = _make_rows(categories={
            "simple_python": [True, True, False],
            "multiple": [True, False],
        })
        progress = build_progress(rows)
        assert "simple_python: 3 done" in progress
        assert "multiple: 2 done" in progress


class TestConfigLabel:
    """Test row label formatting."""

    def test_ollama_native(self):
        from tests.eval.bfcl.bfcl_report import _config_label
        label = _config_label({
            "model": "ministral-3:8b-instruct-2512-q4_K_M",
            "backend": "ollama", "mode": "native", "ablation": "reforged",
        })
        assert "OL/N" in label
        assert "[reforged]" in label

    def test_anthropic_bare(self):
        from tests.eval.bfcl.bfcl_report import _config_label
        label = _config_label({
            "model": "claude-haiku-4-5-20251001",
            "backend": "anthropic", "mode": "native", "ablation": "bare",
        })
        assert "AN/N" in label
        assert "[bare]" in label


class TestEndToEndReport:
    """Test report with realistic multi-config mock data."""

    def test_reforged_vs_bare_comparison(self):
        """Two configs (reforged/bare) for the same model show in one table."""
        from tests.eval.bfcl.bfcl_report import build_report

        categories = {
            "simple_python": ([True]*8 + [False]*2),   # 80%
            "multiple": ([True]*7 + [False]*3),         # 70%
            "parallel": ([True]*9 + [False]*1),         # 90%
            "irrelevance": ([True]*10),                 # 100%
        }
        categories_bare = {
            "simple_python": ([True]*6 + [False]*4),   # 60%
            "multiple": ([True]*4 + [False]*6),         # 40%
            "parallel": ([True]*5 + [False]*5),         # 50%
            "irrelevance": ([True]*3 + [False]*7),     # 30%
        }

        rows_reforged = _make_rows(
            model="ministral-3:8b-instruct-2512-q4_K_M",
            ablation="reforged",
            categories=categories,
        )
        rows_bare = _make_rows(
            model="ministral-3:8b-instruct-2512-q4_K_M",
            ablation="bare",
            categories=categories_bare,
        )

        report = build_report(rows_reforged + rows_bare)

        # Both configs appear
        assert "[reforged]" in report
        assert "[bare]" in report

        # Reforged should be first (higher accuracy)
        reforged_pos = report.index("[reforged]")
        bare_pos = report.index("[bare]")
        assert reforged_pos < bare_pos

        # All 4 category columns present
        assert "spy" in report
        assert "mul" in report
        assert "par" in report
        assert "irr" in report

    def test_multi_model_comparison(self):
        """Three different models sort correctly by accuracy."""
        from tests.eval.bfcl.bfcl_report import build_report

        rows = []
        for model, pct in [
            ("claude-haiku-4-5-20251001", 1.0),
            ("ministral-3:8b-instruct-2512-q4_K_M", 0.8),
            ("llama3.1:8b-instruct-q4_K_M", 0.5),
        ]:
            n = 20
            valid_count = int(n * pct)
            rows.extend(_make_rows(
                model=model,
                categories={"simple_python": [True]*valid_count + [False]*(n - valid_count)},
            ))

        report = build_report(rows)
        lines = [l for l in report.split("\n") if "[reforged]" in l]
        assert len(lines) == 3
        # First line should be haiku (100%)
        assert "claude-haiku" in lines[0]
        # Last line should be llama (50%)
        assert "llama3.1" in lines[2]

    def test_write_and_read_jsonl(self, tmp_path):
        """Write mock JSONL, read with report, verify output."""
        from tests.eval.bfcl.bfcl_report import _load_rows, build_report

        rows = _make_rows(categories={
            "simple_python": [True, True, False],
            "irrelevance": [True, True, True],
        })

        jsonl_path = tmp_path / "test.jsonl"
        with jsonl_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        loaded = _load_rows(jsonl_path)
        assert len(loaded) == 6

        report = build_report(loaded)
        assert "83.3%" in report  # 5/6
