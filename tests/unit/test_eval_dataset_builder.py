"""Focused coverage for the offline Forge Parquet dataset builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.eval import dataset_builder, publication
from tests.eval.outcomes import LEGACY_DIALECT


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ablation": "reforged",
        "accuracy": True,
        "backend": "backend-a",
        "budget_tokens": 100,
        "compaction_events": 0,
        "completeness": True,
        "elapsed_s": 1.25,
        "error_message": None,
        "error_type": None,
        "gen": 1,
        "ideal_iterations": 2,
        "iterations": 2,
        "mode": "native",
        "model": "model-a",
        "reasoning_msgs": 1,
        "retry_nudges": 0,
        "run": 1,
        "scenario": "repeated-scenario",
        "step_nudges": 0,
        "tool_choice": "auto",
        "tool_errors": 0,
        "wasted_calls": 0,
    }
    row.update(overrides)
    return row


def _write_source(
    root: Path, name: str, rows: list[dict[str, Any]]
) -> publication.SourceSpec:
    payload = b"".join(
        json.dumps(
            row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
    (root / name).write_bytes(payload)
    generations = {row["gen"] for row in rows}
    assert len(generations) == 1
    return publication.SourceSpec(
        name,
        "synthetic",
        generations.pop(),
        LEGACY_DIALECT,
        len(rows),
        hashlib.sha256(payload).hexdigest(),
    )


def _fixture_plan(root: Path) -> publication.PublicationPlan:
    root.mkdir()
    gen1 = [
        _row(model="shared", run=1, cost_usd=0.125, reasoning_wire=None),
        _row(model="carried", run=1, accuracy=None, elapsed_s=2.5),
        _row(
            model="shared",
            run=1,
            reasoning_replay="none",
            elapsed_s=3.75,
        ),
    ]
    gen2 = [
        _row(model="shared", run=1, gen=2, reasoning_replay="none"),
        _row(model="new", run=1, gen=2, cost_usd=0.5),
        _row(model="new", run=1, gen=2, cost_usd=0.75),
        _row(model="another", run=1, gen=2, elapsed_s=4.0),
    ]
    specs = [
        _write_source(root, "gen1.jsonl", gen1),
        _write_source(root, "gen2.jsonl", gen2),
    ]
    return publication.build_publication_plan(root, source_specs=specs)


@pytest.fixture
def metadata() -> dataset_builder.ReleaseMetadata:
    return dataset_builder.validate_release_metadata(
        "mit",
        dataset_builder.FORGE_PAPER_DOI_URL,
        "http://example.test/reproduce?version=1",
    )


def test_physical_schema_is_exact_and_pyarrow_is_lazy() -> None:
    pa, _ = dataset_builder._require_pyarrow()
    schema = dataset_builder._physical_schema(pa)
    assert len(schema) == 52
    assert [field.name for field in schema] == [
        field.name for field in publication.NORMALIZED_SCHEMA
    ]
    assert [field.nullable for field in schema] == [
        field.nullable for field in publication.NORMALIZED_SCHEMA
    ]
    assert schema.field("elapsed_s").type == pa.float64()
    assert schema.field("source_line").type == pa.int64()


def test_card_documents_replay_reasoning_and_carried_evidence_semantics(
    tmp_path: Path,
    metadata: dataset_builder.ReleaseMetadata,
) -> None:
    plan = _fixture_plan(tmp_path / "source")
    body = dataset_builder._card_bytes(metadata, plan).decode("utf-8")
    assert "Rows predating the\nknob have no raw value and resolve to effective `full`" in body
    assert (
        "legacy-inferred arm\nremains distinct from an explicitly recorded `full` arm"
    ) in body
    assert "raw `reasoning_replay` field preserves the source value" in body
    assert "Missing raw\n`reasoning_level` resolves to effective `default`" in body
    assert "Each accepted source revision is\npinned by its exact hash" in body
    assert (
        "Carried evidence is an older\nselected cohort retained only because no "
        "newer matching cohort exists."
    ) in body
    assert "score = correct_count / attempted_count" in body
    assert "validated_accuracy = correct_count / validated_count" in body
    assert "completion_rate = completed_count / attempted_count" in body
    assert "outcome corpus" in body
    assert "not a collection of complete agent traces" in body
    assert "comparability epoch" in body
    assert dataset_builder.HUGGING_FACE_DATASET_URL in body
    assert body.count(f'load_dataset("{dataset_builder.HUGGING_FACE_DATASET_ID}"') == 3
    assert "does not include a packaged Parquet-to-report or dashboard command" in body
    assert "Recompute aggregate metrics and perform independent analyses" in body
    assert "Reproduce Forge's aggregate reports" not in body
    assert (
        "Forge: Closing the Agentic Reliability Gap Between\n"
        "Self-Hosted and Frontier Language Models"
    ) in body
    assert metadata.citation_url in body


@pytest.mark.parametrize("license_id", ["", "other", "custom", "MIT", " mit"])
def test_license_rejects_unsupported_identifiers(license_id: str) -> None:
    with pytest.raises(dataset_builder.DatasetBuilderError, match="license"):
        dataset_builder.validate_release_metadata(
            license_id, "https://example.test/cite", "https://example.test/repro"
        )


@pytest.mark.parametrize(
    "url",
    ["", "relative/path", "ftp://example.test/x", "https:///missing-host", " https://example.test"],
)
def test_metadata_rejects_malformed_urls(url: str) -> None:
    with pytest.raises(dataset_builder.DatasetBuilderError, match="URL"):
        dataset_builder.validate_release_metadata(
            "mit", url, "https://example.test/repro"
        )


@pytest.mark.parametrize(
    "citation_url",
    [
        "https://example.test/citation",
        "http://doi.org/10.1145/3786335.3813193",
        "https://doi.org/10.1145/3786335.3813193/",
        "https://doi.org/10.1145/3786335.3813193?source=forge",
    ],
)
def test_metadata_requires_canonical_forge_paper_doi(citation_url: str) -> None:
    with pytest.raises(
        dataset_builder.DatasetBuilderError, match="canonical Forge paper DOI"
    ):
        dataset_builder.validate_release_metadata(
            "mit", citation_url, "https://example.test/repro"
        )


def test_build_is_deterministic_structured_and_fully_verified(
    tmp_path: Path,
    metadata: dataset_builder.ReleaseMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _fixture_plan(tmp_path / "source")
    monkeypatch.setattr(dataset_builder, "SHARD_ROWS", 3)
    monkeypatch.setattr(dataset_builder, "RECORD_BATCH_ROWS", 2)
    observed: dict[str, int] = {name: 0 for name in dataset_builder.CONFIGURATION_NAMES}

    def observe(name: str, rows: int) -> None:
        observed[name] = max(observed[name], rows)

    first = tmp_path / "first"
    second = tmp_path / "second"
    summary = dataset_builder._build_from_plan(
        plan, first, metadata, observe_buffer=observe
    )
    dataset_builder._build_from_plan(plan, second, metadata)

    assert summary["verified"] is True
    assert summary["counts"] == {
        name: plan.views[name].row_count
        for name in dataset_builder.CONFIGURATION_NAMES
    }
    assert all(size <= 2 for size in observed.values())
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert {
        path.relative_to(first).as_posix()
        for path in first.rglob("*.parquet")
    } == {
        "data/latest/part-00001-of-00002.parquet",
        "data/latest/part-00002-of-00002.parquet",
        "data/snapshot/part-00001-of-00002.parquet",
        "data/snapshot/part-00002-of-00002.parquet",
        "data/history/part-00001-of-00003.parquet",
        "data/history/part-00002-of-00003.parquet",
        "data/history/part-00003-of-00003.parquet",
    }

    manifest = json.loads((first / "manifest.json").read_bytes())
    assert manifest["publication_plan"] == plan.to_dict()
    assert manifest["format"] == {
        "configuration_order": ["latest", "snapshot", "history"],
        "compression": "zstd",
        "record_batch_rows": 2,
        "shard_rows": 3,
        "shard_template": "part-{index:05d}-of-{total:05d}.parquet",
    }
    assert "builder_source_sha256" in manifest["builder"]
    assert len(manifest["builder"]["source_files"]) == 5
    assert all("sha256" in fact for fact in manifest["files"])
    assert all(
        "rows" in fact
        for fact in manifest["files"]
        if fact["path"].endswith(".parquet")
    )
    assert "manifest.json" not in {fact["path"] for fact in manifest["files"]}

    card = dataset_builder._parse_card(first / "README.md")
    assert card["license"] == "mit"
    assert [config["config_name"] for config in card["configs"]] == [
        "latest",
        "snapshot",
        "history",
    ]
    assert [config["data_files"][0]["path"] for config in card["configs"]] == [
        "data/latest/*.parquet",
        "data/snapshot/*.parquet",
        "data/history/*.parquet",
    ]
    assert [config["config_name"] for config in card["configs"] if config.get("default")] == ["latest"]
    assert dataset_builder._verify_against_plan(plan, first)["verified"] is True


@pytest.mark.parametrize("nonempty", [False, True])
def test_preexisting_destination_is_never_modified(
    tmp_path: Path,
    metadata: dataset_builder.ReleaseMetadata,
    nonempty: bool,
) -> None:
    plan = _fixture_plan(tmp_path / "source")
    output = tmp_path / "existing"
    output.mkdir()
    if nonempty:
        (output / "sentinel").write_text("keep", encoding="utf-8")
    before = list(output.iterdir())
    with pytest.raises(dataset_builder.DatasetBuilderError, match="already exists"):
        dataset_builder._build_from_plan(plan, output, metadata)
    assert list(output.iterdir()) == before


def test_missing_output_parents_are_created_after_plan(
    tmp_path: Path, metadata: dataset_builder.ReleaseMetadata
) -> None:
    plan = _fixture_plan(tmp_path / "source")
    output = tmp_path / "new" / "nested" / "bundle"
    dataset_builder._build_from_plan(plan, output, metadata)
    assert (output / "manifest.json").is_file()


def test_injected_publish_failure_retains_staging_and_not_destination(
    tmp_path: Path,
    metadata: dataset_builder.ReleaseMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _fixture_plan(tmp_path / "source")
    output = tmp_path / "requested"

    def fail_publish(_staging: Path, _output: Path) -> None:
        raise OSError("injected after manifest")

    monkeypatch.setattr(dataset_builder, "_publish_staging", fail_publish)
    with pytest.raises(dataset_builder.DatasetBuilderError, match="incomplete staging retained") as exc_info:
        dataset_builder._build_from_plan(plan, output, metadata)
    assert "requested destination remained absent" in str(exc_info.value)
    assert not output.exists()
    staging = list(tmp_path.glob(".requested.incomplete-*"))
    assert len(staging) == 1
    assert (staging[0] / "manifest.json").is_file()


@pytest.mark.parametrize(
    "mutation",
    ["extra", "missing", "corrupt_parquet", "corrupt_card", "corrupt_provenance"],
)
def test_verifier_rejects_tree_and_content_corruption(
    tmp_path: Path,
    metadata: dataset_builder.ReleaseMetadata,
    mutation: str,
) -> None:
    plan = _fixture_plan(tmp_path / "source")
    output = tmp_path / "bundle"
    dataset_builder._build_from_plan(plan, output, metadata)
    if mutation == "extra":
        (output / "extra.txt").write_text("extra", encoding="utf-8")
    elif mutation == "missing":
        (output / "provenance" / "schema.json").unlink()
    elif mutation == "corrupt_parquet":
        shard = next(output.rglob("*.parquet"))
        shard.write_bytes(shard.read_bytes() + b"corrupt")
    elif mutation == "corrupt_card":
        (output / "README.md").write_text("---\n{}\n---\n", encoding="utf-8")
    else:
        (output / "provenance" / "sources.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises((dataset_builder.DatasetBuilderError, OSError)):
        dataset_builder._verify_against_plan(plan, output)


@pytest.mark.parametrize("mutation", ["early", "extra", "order", "schema"])
def test_readback_rejects_row_and_schema_invariants_after_rehash(
    tmp_path: Path,
    metadata: dataset_builder.ReleaseMetadata,
    mutation: str,
) -> None:
    """Reach read-back checks rather than stopping at the physical file hash."""
    pa, pq = dataset_builder._require_pyarrow()
    plan = _fixture_plan(tmp_path / "source")
    output = tmp_path / "bundle"
    dataset_builder._build_from_plan(plan, output, metadata)
    relative = "data/history/part-00001-of-00001.parquet"
    path = output / relative
    table = pq.read_table(path)
    if mutation == "early":
        changed = table.slice(0, table.num_rows - 1)
    elif mutation == "extra":
        changed = pa.concat_tables([table, table.slice(0, 1)])
    elif mutation == "order":
        indices = pa.array([1, 0, *range(2, table.num_rows)])
        changed = table.take(indices)
    else:
        changed = pa.Table.from_pylist(table.to_pylist())
    pq.write_table(
        changed,
        path,
        compression=dataset_builder.COMPRESSION,
        version="2.6",
        use_dictionary=False,
        write_statistics=True,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    fact = next(item for item in manifest["files"] if item["path"] == relative)
    fact["sha256"] = dataset_builder._sha256_file(path)
    fact["size_bytes"] = path.stat().st_size
    manifest_path.write_bytes(dataset_builder._canonical_bytes(manifest))
    with pytest.raises(dataset_builder.DatasetBuilderError):
        dataset_builder._verify_against_plan(plan, output)


def test_missing_pyarrow_message_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = dataset_builder.importlib.import_module

    def missing(name: str) -> Any:
        if name == "pyarrow":
            raise ImportError("not installed")
        return real_import(name)

    monkeypatch.setattr(dataset_builder.importlib, "import_module", missing)
    with pytest.raises(dataset_builder.DatasetBuilderError, match=r"\.\[dataset-builder\]"):
        dataset_builder._require_pyarrow()
