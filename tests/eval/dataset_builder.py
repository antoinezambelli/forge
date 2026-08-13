"""Offline, deterministic Parquet materializer for the Forge eval corpus.

This checkout-local command intentionally consumes only the verified publication
contract in :mod:`tests.eval.publication`.  It does not upload data, contact the
Hugging Face Hub, or require credentials.

Build and verify the pinned corpus with::

    python -m tests.eval.dataset_builder build --source-root . --output build/forge-eval-dataset-v2 --license mit --citation-url https://doi.org/10.1145/3786335.3813193 --reproduction-url https://example.test/reproduce
    python -m tests.eval.dataset_builder verify --source-root . --bundle build/forge-eval-dataset-v2
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tests.eval import publication


BUNDLE_VERSION = "forge-eval-parquet-v2"
CONFIGURATION_NAMES = ("latest", "snapshot", "history")
COMPRESSION = "zstd"
RECORD_BATCH_ROWS = 8_192
SHARD_ROWS = 100_000
SHARD_TEMPLATE = "part-{index:05d}-of-{total:05d}.parquet"
FORGE_PAPER_DOI_URL = "https://doi.org/10.1145/3786335.3813193"
HUGGING_FACE_DATASET_ID = "antoinezambelli/forge-evals"
HUGGING_FACE_DATASET_URL = (
    f"https://huggingface.co/datasets/{HUGGING_FACE_DATASET_ID}"
)

# Deliberately frozen for the fixed v2 card layout.  Identifiers that need a
# license_name or a shipped LICENSE file (including "other") are unsupported.
ACCEPTED_LICENSES = frozenset(
    {
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc-by-4.0",
        "cc-by-sa-4.0",
        "cc0-1.0",
        "gpl-3.0",
        "lgpl-3.0",
        "mit",
        "mpl-2.0",
        "odc-by",
        "odbl",
    }
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BUILDER_SOURCE_PATHS = (
    "tests/eval/dataset_builder.py",
    "tests/eval/publication.py",
    "tests/eval/outcomes.py",
    "tests/eval/generation.py",
    "tests/eval/provenance.py",
)


class DatasetBuilderError(RuntimeError):
    """A compact bundle materialization or verification failure."""


@dataclass(frozen=True)
class ReleaseMetadata:
    license: str
    citation_url: str
    reproduction_url: str

    def to_dict(self) -> dict[str, str]:
        return {
            "license": self.license,
            "citation_url": self.citation_url,
            "reproduction_url": self.reproduction_url,
        }


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        pa = importlib.import_module("pyarrow")
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise DatasetBuilderError(
            'Parquet materialization requires PyArrow; install it locally with '
            '`python -m pip install -e ".[dataset-builder]"`.'
        ) from exc
    return pa, pq


def _validate_url(value: str, label: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise DatasetBuilderError(f"{label} must be an absolute HTTP(S) URL")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DatasetBuilderError(f"{label} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise DatasetBuilderError(f"{label} must not contain URL credentials")
    return value


def validate_release_metadata(
    license_id: str, citation_url: str, reproduction_url: str
) -> ReleaseMetadata:
    if not isinstance(license_id, str) or license_id != license_id.strip():
        raise DatasetBuilderError("license must be a recognized Hugging Face identifier")
    if license_id not in ACCEPTED_LICENSES:
        accepted = ", ".join(sorted(ACCEPTED_LICENSES))
        raise DatasetBuilderError(
            f"unsupported Hugging Face license identifier {license_id!r}; "
            f"accepted v2 identifiers: {accepted}"
        )
    validated_citation_url = _validate_url(citation_url, "citation URL")
    if validated_citation_url != FORGE_PAPER_DOI_URL:
        raise DatasetBuilderError(
            "citation URL must be the canonical Forge paper DOI "
            f"{FORGE_PAPER_DOI_URL!r}"
        )
    return ReleaseMetadata(
        license=license_id,
        citation_url=validated_citation_url,
        reproduction_url=_validate_url(reproduction_url, "reproduction URL"),
    )


def _canonical_bytes(value: Any) -> bytes:
    return publication.canonical_json_bytes(value, sort_keys=True) + b"\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _physical_schema(pa: Any) -> Any:
    types = {
        "string": pa.string(),
        "bool": pa.bool_(),
        "int64": pa.int64(),
        "float64": pa.float64(),
    }
    return pa.schema(
        [
            pa.field(field.name, types[field.logical_type], nullable=field.nullable)
            for field in publication.NORMALIZED_SCHEMA
        ]
    )


def _source_provenance(plan: publication.PublicationPlan) -> dict[str, Any]:
    return {
        "contract_version": publication.CONTRACT_VERSION,
        "sources": [source.to_dict() for source in plan.sources],
    }


def _config_metadata(name: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "config_name": name,
        "data_files": [{"split": "train", "path": f"data/{name}/*.parquet"}],
    }
    if name == "latest":
        config["default"] = True
    return config


def _card_frontmatter(metadata: ReleaseMetadata) -> dict[str, Any]:
    return {
        "pretty_name": "Forge Agentic Workflow Evaluation Corpus",
        "license": metadata.license,
        "tags": [
            "agent-evaluation",
            "tool-use",
            "function-calling",
            "local-inference",
            "benchmark",
            "tabular",
        ],
        "configs": [_config_metadata(name) for name in CONFIGURATION_NAMES],
    }


def _format_rate(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator * 100:.2f}%" if denominator else "n/a"


def _card_bytes(
    metadata: ReleaseMetadata, plan: publication.PublicationPlan
) -> bytes:
    frontmatter = publication.canonical_json(_card_frontmatter(metadata), sort_keys=False)
    view_rows = []
    for name in CONFIGURATION_NAMES:
        stats = plan.views[name]
        view_rows.append(
            f"| `{name}`{' (default)' if name == 'latest' else ''} "
            f"| {stats.attempted_count:,} | {stats.correct_count:,} "
            f"| {stats.validated_count:,} | {stats.completed_count:,} "
            f"| {_format_rate(stats.correct_count, stats.attempted_count)} "
            f"| {_format_rate(stats.correct_count, stats.validated_count)} "
            f"| {_format_rate(stats.completed_count, stats.attempted_count)} |"
        )
    view_table = "\n".join(view_rows)
    column_count = len(publication.NORMALIZED_SCHEMA)
    body = f"""---
{frontmatter}
---

# Forge Agentic Workflow Evaluation Corpus

Forge evaluates multi-step agent workflows across models, quantizations,
inference backends, function-calling modes, scenarios, guardrail ablations, and
reasoning-replay policies. This dataset publishes the released run-level outcome
records behind Forge's reports and dashboard. One row represents one attempted
evaluation run.

This is an **outcome corpus**, not a collection of complete agent traces. It does
not contain full prompts, conversations, tool-result transcripts, or hidden model
reasoning and cannot reconstruct an end-to-end trajectory.

## Configurations and counts

| Configuration | Attempted | Correct | Validated | Completed | Score | Validated accuracy | Completion rate |
|---|---:|---:|---:|---:|---:|---:|---:|
{view_table}

- `latest` is the default and contains selected rows at the corpus-wide maximum
  evaluation generation.
- `snapshot` retains the selected arm for each comparable configuration,
  including older carried evidence when no newer matching arm exists.
- `history` contains every released row, including superseded generations, in
  pinned source-file and source-line order.

Dataset repository: {HUGGING_FACE_DATASET_URL}

## Quickstart

```python
from datasets import load_dataset

latest = load_dataset("{HUGGING_FACE_DATASET_ID}", "latest", split="train")
snapshot = load_dataset("{HUGGING_FACE_DATASET_ID}", "snapshot", split="train")
history = load_dataset("{HUGGING_FACE_DATASET_ID}", "history", split="train")
```

## Metric contract

The canonical headline metric is **Score**:

- `score = correct_count / attempted_count`
- `validated_accuracy = correct_count / validated_count`
- `completion_rate = completed_count / attempted_count`

`attempted_count` means rows present in the selected cohort, not a theoretical
schedule. `correct` is `true`, `false`, or null when no usable correctness
judgment exists. A null judgment remains in the Score denominator and is
excluded from the validated-accuracy denominator. `completed` records whether
the workflow returned normally and is independent of correctness. Exact integer
components are included in the publication plan so every displayed rate is
reproducible without reverse-engineering rounded percentages.

## Schema overview

All {column_count} columns use one explicit v2 schema:

- **Identity and condition:** model, backend, mode, ablation, tool choice,
  reasoning replay/level, scenario, run index, and evaluation generation.
- **Outcome:** `correct`, `completed`, `validation_error`, execution error type,
  and execution error message.
- **Efficiency:** iterations, ideal iterations, wasted calls, elapsed seconds,
  context budget, and stream retries.
- **Guardrail and reasoning telemetry:** nudges, tool errors, compaction events,
  captured reasoning counts, and on-wire reasoning counts.
- **Hosted accounting:** input, output, cache-creation, cache-read tokens, and
  recorded cost when the source backend supplied them.
- **Provenance and selection:** release, source file and line hashes, generation
  metadata, canonical configuration/arm identifiers, selection status, and view
  membership.

The builder never modifies source JSONLs. Each accepted source revision is
pinned by its exact hash, so intentional provenance corrections require an
explicit pin update. Legacy `accuracy`/`completeness` spelling is normalized to
published `correct`/`completed`; the legacy aliases do not appear in the
Parquet schema. Sparse source fields become null.

## Generations, replay, and carried evidence

An evaluation `generation` is a **comparability epoch**, not generated model
text and not necessarily a Forge release. A generation changes when collection
semantics materially change. Several releases may share one generation.

The raw `reasoning_replay` field preserves the source value. Rows predating the
knob have no raw value and resolve to effective `full`; this legacy-inferred arm
remains distinct from an explicitly recorded `full` arm. Missing raw
`reasoning_level` resolves to effective `default`. Carried evidence is an older
selected cohort retained only because no newer matching cohort exists.

## Methodology and statistical use

Forge scenarios exercise tool selection, argument fidelity, multi-step
sequencing, error recovery, stateful interactions, and context-pressure paths.
Repeated runs are stored individually. Cohort comparisons should control for
model, quantization, backend, mode, scenario set, replay policy, reasoning
level, generation, and collection environment. For paired arms, Forge uses
paired McNemar tests on matching `(scenario, run)` observations and Wilson
intervals for Score. The run rows contain the paired observations and exact
outcome components needed to recompute those analyses independently. This
bundle does not include a packaged Parquet-to-report or dashboard command.

## Intended uses

- Recompute aggregate metrics and perform independent analyses from run-level
  outcomes.
- Compare controlled model/backend/mode or ablation cohorts.
- Study completion, validation, efficiency, replay, and guardrail behavior.
- Build new statistical views while retaining source-level provenance.

This dataset should not be used as a general model-quality leaderboard, as
training text, or as evidence that one backend/model is universally superior.
It also does not measure subjective answer quality beyond each scenario's
deterministic validator.

## Limitations

Many cells are intentionally absent or inapplicable, and a missing scenario is
not scored as an attempted failure. Historical evidence spans different rigs,
backends, quantizations, serving versions, context budgets, and model-specific
reasoning behavior. Timing, token, and cost fields are operational measurements
and are only comparable under compatible collection conditions. Repeated runs
within one scenario may be more correlated than independent scenario-level
effects.

## Provenance and integrity

`provenance/sources.json` pins every released source, generation, dialect, row
count, and SHA-256. `provenance/schema.json` contains the complete source-dialect
and normalized-schema contract. `manifest.json` records every configuration,
logical digest, shard row count, file hash, builder-source hash, Git revision,
Python version, and Parquet writer version. Verify a downloaded bundle with:

```bash
python -m tests.eval.dataset_builder verify --source-root . --bundle <bundle>
```

Project and methodology: {metadata.reproduction_url}

Paper citation: Zambelli, A. *Forge: Closing the Agentic Reliability Gap Between
Self-Hosted and Frontier Language Models.* {metadata.citation_url}

License: `{metadata.license}`
"""
    return body.encode("utf-8")


def _parse_card(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DatasetBuilderError("README.md is not readable UTF-8") from exc
    lines = text.splitlines()
    if len(lines) < 3 or lines[0] != "---":
        raise DatasetBuilderError("README.md has invalid metadata frontmatter")
    try:
        closing = lines.index("---", 1)
        metadata = json.loads("\n".join(lines[1:closing]))
    except (ValueError, json.JSONDecodeError) as exc:
        raise DatasetBuilderError("README.md has invalid metadata frontmatter") from exc
    if not isinstance(metadata, dict):
        raise DatasetBuilderError("README.md metadata must be an object")
    return metadata


def _builder_identity(pa: Any) -> dict[str, Any]:
    files: list[dict[str, str]] = []
    combined = hashlib.sha256()
    for relative in BUILDER_SOURCE_PATHS:
        path = _REPOSITORY_ROOT / relative
        if not path.is_file():
            raise DatasetBuilderError(f"builder source is missing: {relative}")
        file_hash = _sha256_file(path)
        files.append({"path": relative, "sha256": file_hash})
        combined.update(
            publication.canonical_json_bytes([relative, file_hash], sort_keys=False)
            + b"\n"
        )
    try:
        revision = subprocess.run(
            ["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "builder_source_sha256": combined.hexdigest(),
        "source_files": files,
        "git_revision": revision,
        "python_version": platform.python_version(),
        "parquet_writer": {"name": "pyarrow", "version": pa.__version__},
    }


def _shard_count(row_count: int) -> int:
    return (row_count + SHARD_ROWS - 1) // SHARD_ROWS


def _shard_relpaths(plan: publication.PublicationPlan, name: str) -> list[str]:
    total = _shard_count(plan.views[name].row_count)
    return [
        f"data/{name}/" + SHARD_TEMPLATE.format(index=index, total=total)
        for index in range(1, total + 1)
    ]


class _ViewWriter:
    def __init__(
        self,
        *,
        root: Path,
        name: str,
        row_count: int,
        schema: Any,
        pa: Any,
        pq: Any,
        observe_buffer: Callable[[str, int], None] | None,
    ) -> None:
        self.root = root
        self.name = name
        self.row_count = row_count
        self.schema = schema
        self.pa = pa
        self.pq = pq
        self.observe_buffer = observe_buffer
        self.total_shards = _shard_count(row_count)
        self.buffer: list[dict[str, Any]] = []
        self.emitted = 0
        self.shard_index = 1
        self.shard_emitted = 0
        self.writer: Any | None = None
        self.shards: list[dict[str, Any]] = []

    def _relative_path(self) -> str:
        return f"data/{self.name}/" + SHARD_TEMPLATE.format(
            index=self.shard_index, total=self.total_shards
        )

    def add(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if self.observe_buffer is not None:
            self.observe_buffer(self.name, len(self.buffer))
        shard_target = min(SHARD_ROWS, self.row_count - self.emitted + self.shard_emitted)
        if len(self.buffer) >= RECORD_BATCH_ROWS or self.shard_emitted + len(self.buffer) == shard_target:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        if self.writer is None:
            relative = self._relative_path()
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = self.pq.ParquetWriter(
                path,
                self.schema,
                compression=COMPRESSION,
                version="2.6",
                use_dictionary=False,
                write_statistics=True,
            )
        table = self.pa.Table.from_pylist(self.buffer, schema=self.schema)
        batch_rows = len(self.buffer)
        self.writer.write_table(table, row_group_size=RECORD_BATCH_ROWS)
        self.buffer.clear()
        self.emitted += batch_rows
        self.shard_emitted += batch_rows
        if self.shard_emitted == SHARD_ROWS or self.emitted == self.row_count:
            relative = self._relative_path()
            self.writer.close()
            self.writer = None
            self.shards.append({"path": relative, "rows": self.shard_emitted})
            self.shard_index += 1
            self.shard_emitted = 0

    def finish(self) -> list[dict[str, Any]]:
        self._flush()
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.emitted != self.row_count:
            raise DatasetBuilderError(f"{self.name} emitted row count does not match plan")
        if len(self.shards) != self.total_shards:
            raise DatasetBuilderError(f"{self.name} shard count does not match plan")
        return self.shards

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def _write_rows(
    plan: publication.PublicationPlan,
    root: Path,
    pa: Any,
    pq: Any,
    *,
    observe_buffer: Callable[[str, int], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    schema = _physical_schema(pa)
    writers = {
        name: _ViewWriter(
            root=root,
            name=name,
            row_count=plan.views[name].row_count,
            schema=schema,
            pa=pa,
            pq=pq,
            observe_buffer=observe_buffer,
        )
        for name in CONFIGURATION_NAMES
    }
    try:
        for row in publication.iter_classified_history(plan):
            writers["history"].add(row)
            if publication.row_in_view(row, "snapshot"):
                writers["snapshot"].add(row)
            if publication.row_in_view(row, "latest"):
                writers["latest"].add(row)
        return {name: writers[name].finish() for name in CONFIGURATION_NAMES}
    finally:
        for writer in writers.values():
            writer.close()


def _expected_non_manifest_paths(
    plan: publication.PublicationPlan,
) -> list[str]:
    paths = ["README.md", "provenance/sources.json", "provenance/schema.json"]
    for name in CONFIGURATION_NAMES:
        paths.extend(_shard_relpaths(plan, name))
    return paths


def _actual_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _file_facts(
    root: Path,
    expected: Sequence[str],
    shard_rows: Mapping[str, int],
    *,
    permitted_extra: Sequence[str] = (),
) -> list[dict[str, Any]]:
    actual = _actual_files(root)
    expected_tree = set(expected) | set(permitted_extra)
    if actual != expected_tree:
        missing = sorted(expected_tree - actual)
        extra = sorted(actual - expected_tree)
        raise DatasetBuilderError(f"bundle tree mismatch; missing={missing}, extra={extra}")
    facts = []
    for relative in expected:
        path = root / relative
        fact: dict[str, Any] = {
            "path": relative,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if relative in shard_rows:
            fact["rows"] = shard_rows[relative]
        facts.append(fact)
    return facts


def _format_facts() -> dict[str, Any]:
    return {
        "configuration_order": list(CONFIGURATION_NAMES),
        "compression": COMPRESSION,
        "record_batch_rows": RECORD_BATCH_ROWS,
        "shard_rows": SHARD_ROWS,
        "shard_template": SHARD_TEMPLATE,
    }


def _configuration_facts(
    plan: publication.PublicationPlan,
    shards: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "default": name == "latest",
            "data_files": f"data/{name}/*.parquet",
            "row_count": plan.views[name].row_count,
            "shards": shards[name],
        }
        for name in CONFIGURATION_NAMES
    ]


def _write_support_files(
    root: Path, plan: publication.PublicationPlan, metadata: ReleaseMetadata
) -> None:
    (root / "provenance").mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_bytes(_card_bytes(metadata, plan))
    (root / "provenance" / "sources.json").write_bytes(
        _canonical_bytes(_source_provenance(plan))
    )
    (root / "provenance" / "schema.json").write_bytes(
        _canonical_bytes(publication.schema_manifest())
    )


def _manifest(
    *,
    plan: publication.PublicationPlan,
    metadata: ReleaseMetadata,
    builder: dict[str, Any],
    configurations: list[dict[str, Any]],
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "bundle_version": BUNDLE_VERSION,
        "release_metadata": metadata.to_dict(),
        "format": _format_facts(),
        "builder": builder,
        "publication_plan": plan.to_dict(),
        "configurations": configurations,
        "files": files,
    }


def _load_manifest(root: Path) -> dict[str, Any]:
    try:
        raw = (root / "manifest.json").read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetBuilderError("manifest.json is missing or invalid") from exc
    if not isinstance(value, dict):
        raise DatasetBuilderError("manifest.json must contain an object")
    if raw != _canonical_bytes(value):
        raise DatasetBuilderError("manifest.json is not canonical JSON with LF")
    return value


class _DigestAccumulator:
    def __init__(self) -> None:
        self.rows = 0
        self.identity = hashlib.sha256()
        self.logical = hashlib.sha256()

    def update(self, row: dict[str, Any]) -> None:
        self.rows += 1
        self.identity.update(
            publication.canonical_json_bytes(
                [row["source_file"], row["source_line"]], sort_keys=False
            )
            + b"\n"
        )
        self.logical.update(
            publication.canonical_json_bytes(row, sort_keys=True) + b"\n"
        )


def _iter_parquet_rows(
    root: Path,
    shard_facts: Sequence[Mapping[str, Any]],
    schema: Any,
    pa: Any,
    pq: Any,
) -> Iterator[dict[str, Any]]:
    for shard in shard_facts:
        path = root / str(shard["path"])
        try:
            with pa.OSFile(str(path), "rb") as source:
                parquet_file = pq.ParquetFile(source)
                if not parquet_file.schema_arrow.equals(schema, check_metadata=True):
                    raise DatasetBuilderError(
                        f"Parquet schema mismatch: {shard['path']}"
                    )
                if parquet_file.metadata.num_rows != shard["rows"]:
                    raise DatasetBuilderError(
                        f"Parquet row count mismatch: {shard['path']}"
                    )
                for batch in parquet_file.iter_batches(batch_size=RECORD_BATCH_ROWS):
                    for row in batch.to_pylist():
                        yield row
        except DatasetBuilderError:
            raise
        except Exception as exc:
            raise DatasetBuilderError(
                f"Parquet shard is unreadable: {shard['path']}"
            ) from exc


def _next_or_error(iterator: Iterator[dict[str, Any]], view: str) -> dict[str, Any]:
    try:
        return next(iterator)
    except StopIteration as exc:
        raise DatasetBuilderError(f"{view} materialized rows ended early") from exc


def _assert_exhausted(iterator: Iterator[dict[str, Any]], view: str) -> None:
    try:
        next(iterator)
    except StopIteration:
        return
    raise DatasetBuilderError(f"{view} materialized rows contain extras")


def _metadata_from_manifest(manifest: Mapping[str, Any]) -> ReleaseMetadata:
    try:
        value = manifest["release_metadata"]
        if not isinstance(value, dict) or set(value) != {
            "license",
            "citation_url",
            "reproduction_url",
        }:
            raise KeyError
        return validate_release_metadata(
            value["license"], value["citation_url"], value["reproduction_url"]
        )
    except (KeyError, TypeError) as exc:
        raise DatasetBuilderError("manifest release metadata is invalid") from exc


def _verify_against_plan(
    plan: publication.PublicationPlan,
    bundle: str | Path,
    *,
    pa: Any | None = None,
    pq: Any | None = None,
) -> dict[str, Any]:
    if pa is None or pq is None:
        pa, pq = _require_pyarrow()
    root = Path(bundle).resolve()
    if not root.is_dir():
        raise DatasetBuilderError(f"bundle is not a directory: {root}")
    manifest = _load_manifest(root)
    metadata = _metadata_from_manifest(manifest)

    if manifest.get("bundle_version") != BUNDLE_VERSION:
        raise DatasetBuilderError("bundle version mismatch")
    if manifest.get("publication_plan") != plan.to_dict():
        raise DatasetBuilderError("publication plan facts mismatch")
    if manifest.get("format") != _format_facts():
        raise DatasetBuilderError("physical format facts mismatch")

    expected_builder = _builder_identity(pa)
    actual_builder = manifest.get("builder")
    if not isinstance(actual_builder, dict):
        raise DatasetBuilderError("builder identity is missing")
    for key in ("builder_source_sha256", "source_files"):
        if actual_builder.get(key) != expected_builder[key]:
            raise DatasetBuilderError(f"builder identity mismatch: {key}")
    python_version = actual_builder.get("python_version")
    writer = actual_builder.get("parquet_writer")
    if not isinstance(python_version, str) or not python_version:
        raise DatasetBuilderError("builder Python version is invalid")
    if (
        not isinstance(writer, dict)
        or writer.get("name") != "pyarrow"
        or not isinstance(writer.get("version"), str)
        or not writer["version"]
    ):
        raise DatasetBuilderError("builder Parquet-writer version is invalid")

    expected_configs = []
    shard_rows: dict[str, int] = {}
    try:
        actual_configs = manifest["configurations"]
        if not isinstance(actual_configs, list):
            raise TypeError
        for name in CONFIGURATION_NAMES:
            shards = []
            row_count = plan.views[name].row_count
            for index, relative in enumerate(_shard_relpaths(plan, name)):
                rows = min(SHARD_ROWS, row_count - index * SHARD_ROWS)
                shards.append({"path": relative, "rows": rows})
                shard_rows[relative] = rows
            expected_configs.append(
                {
                    "name": name,
                    "default": name == "latest",
                    "data_files": f"data/{name}/*.parquet",
                    "row_count": row_count,
                    "shards": shards,
                }
            )
    except (KeyError, TypeError) as exc:
        raise DatasetBuilderError("configuration manifest is invalid") from exc
    if actual_configs != expected_configs:
        raise DatasetBuilderError("configuration or shard facts mismatch")

    expected_non_manifest = _expected_non_manifest_paths(plan)
    expected_tree = set(expected_non_manifest) | {"manifest.json"}
    actual_tree = _actual_files(root)
    if actual_tree != expected_tree:
        missing = sorted(expected_tree - actual_tree)
        extra = sorted(actual_tree - expected_tree)
        raise DatasetBuilderError(f"finished bundle tree mismatch; missing={missing}, extra={extra}")
    if (root / "provenance" / "sources.json").read_bytes() != _canonical_bytes(
        _source_provenance(plan)
    ):
        raise DatasetBuilderError("source provenance mismatch")
    if (root / "provenance" / "schema.json").read_bytes() != _canonical_bytes(
        publication.schema_manifest()
    ):
        raise DatasetBuilderError("schema provenance mismatch")
    if (root / "README.md").read_bytes() != _card_bytes(metadata, plan):
        raise DatasetBuilderError("Dataset Card content mismatch")
    card = _parse_card(root / "README.md")
    if card != _card_frontmatter(metadata):
        raise DatasetBuilderError("Dataset Card metadata mismatch")
    actual_files = manifest.get("files")
    if not isinstance(actual_files, list):
        raise DatasetBuilderError("manifest file facts are invalid")
    expected_files = _file_facts(
        root, expected_non_manifest, shard_rows, permitted_extra=("manifest.json",)
    )
    if actual_files != expected_files:
        raise DatasetBuilderError("output file hash, size, or row facts mismatch")

    schema = _physical_schema(pa)
    readers = {
        config["name"]: iter(
            _iter_parquet_rows(root, config["shards"], schema, pa, pq)
        )
        for config in actual_configs
    }
    accumulators = {name: _DigestAccumulator() for name in CONFIGURATION_NAMES}
    for expected in publication.iter_classified_history(plan):
        actual = _next_or_error(readers["history"], "history")
        if actual != expected:
            raise DatasetBuilderError("history row content, provenance, or order mismatch")
        accumulators["history"].update(actual)
        for name in ("snapshot", "latest"):
            if publication.row_in_view(expected, name):
                actual = _next_or_error(readers[name], name)
                if actual != expected:
                    raise DatasetBuilderError(
                        f"{name} row content, membership, provenance, or order mismatch"
                    )
                accumulators[name].update(actual)
    for name in CONFIGURATION_NAMES:
        _assert_exhausted(readers[name], name)
        stats = plan.views[name]
        accumulator = accumulators[name]
        if accumulator.rows != stats.row_count:
            raise DatasetBuilderError(f"{name} read-back row count mismatch")
        if accumulator.identity.hexdigest() != stats.source_identity_sha256:
            raise DatasetBuilderError(f"{name} source identity digest mismatch")
        if accumulator.logical.hexdigest() != stats.normalized_logical_sha256:
            raise DatasetBuilderError(f"{name} normalized logical digest mismatch")

    return {
        "bundle": str(root),
        "counts": {name: plan.views[name].row_count for name in CONFIGURATION_NAMES},
        "logical_digests": {
            name: plan.views[name].normalized_logical_sha256
            for name in CONFIGURATION_NAMES
        },
        "builder_source_sha256": actual_builder["builder_source_sha256"],
        "pyarrow_version": actual_builder["parquet_writer"]["version"],
        "verified": True,
    }


def _publish_staging(staging: Path, output: Path) -> None:
    if output.exists():
        raise DatasetBuilderError(f"output path appeared before publication: {output}")
    staging.rename(output)


def _build_from_plan(
    plan: publication.PublicationPlan,
    output: str | Path,
    metadata: ReleaseMetadata,
    *,
    pa: Any | None = None,
    pq: Any | None = None,
    observe_buffer: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    if pa is None or pq is None:
        pa, pq = _require_pyarrow()
    destination = Path(output).resolve()
    if destination.exists():
        raise DatasetBuilderError(f"output path already exists: {destination}")
    parent = destination.parent
    if parent.exists() and not parent.is_dir():
        raise DatasetBuilderError(f"output parent is not a directory: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{destination.name}.incomplete-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        _write_support_files(staging, plan, metadata)
        shards = _write_rows(
            plan, staging, pa, pq, observe_buffer=observe_buffer
        )
        shard_rows = {
            shard["path"]: shard["rows"]
            for config_shards in shards.values()
            for shard in config_shards
        }
        expected = _expected_non_manifest_paths(plan)
        files = _file_facts(staging, expected, shard_rows)
        builder = _builder_identity(pa)
        manifest = _manifest(
            plan=plan,
            metadata=metadata,
            builder=builder,
            configurations=_configuration_facts(plan, shards),
            files=files,
        )
        manifest_bytes = _canonical_bytes(manifest)
        (staging / "manifest.json").write_bytes(manifest_bytes)
        verified = _verify_against_plan(plan, staging, pa=pa, pq=pq)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        output_size = sum(
            path.stat().st_size for path in staging.rglob("*") if path.is_file()
        )
        summary = {
            **verified,
            "bundle": str(destination),
            "manifest_sha256": manifest_sha256,
            "size_bytes": output_size,
            "staging": str(staging),
        }
        _publish_staging(staging, destination)
        return summary
    except Exception as exc:
        destination_absent = not destination.exists()
        if staging.exists():
            message = (
                f"{exc}; incomplete staging retained at {staging}; requested "
                f"destination {'remained absent' if destination_absent else 'exists'}: "
                f"{destination}"
            )
            raise DatasetBuilderError(message) from exc
        if isinstance(exc, DatasetBuilderError):
            raise
        raise DatasetBuilderError(str(exc)) from exc


def build(
    source_root: str | Path,
    output: str | Path,
    license_id: str,
    citation_url: str,
    reproduction_url: str,
) -> dict[str, Any]:
    """Build the pinned corpus into a fresh local bundle."""
    metadata = validate_release_metadata(license_id, citation_url, reproduction_url)
    pa, pq = _require_pyarrow()
    destination = Path(output).resolve()
    if destination.exists():
        raise DatasetBuilderError(f"output path already exists: {destination}")
    plan = publication.build_publication_plan(source_root)
    return _build_from_plan(plan, destination, metadata, pa=pa, pq=pq)


def verify(source_root: str | Path, bundle: str | Path) -> dict[str, Any]:
    """Read-only verification of a finished bundle against the pinned corpus."""
    pa, pq = _require_pyarrow()
    plan = publication.build_publication_plan(source_root)
    try:
        return _verify_against_plan(plan, bundle, pa=pa, pq=pq)
    except DatasetBuilderError:
        raise
    except Exception as exc:
        raise DatasetBuilderError(f"bundle verification failed: {exc}") from exc


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="build a fresh pinned bundle")
    build_parser.add_argument("--source-root", required=True, type=Path)
    build_parser.add_argument("--output", required=True, type=Path)
    build_parser.add_argument(
        "--license",
        required=True,
        dest="license_id",
        help="recognized v2 identifier: " + ", ".join(sorted(ACCEPTED_LICENSES)),
    )
    build_parser.add_argument(
        "--citation-url",
        required=True,
        help=f"canonical Forge paper DOI (must be {FORGE_PAPER_DOI_URL})",
    )
    build_parser.add_argument("--reproduction-url", required=True)
    verify_parser = subparsers.add_parser("verify", help="verify a finished pinned bundle")
    verify_parser.add_argument("--source-root", required=True, type=Path)
    verify_parser.add_argument("--bundle", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "build":
            summary = build(
                args.source_root,
                args.output,
                args.license_id,
                args.citation_url,
                args.reproduction_url,
            )
        else:
            summary = verify(args.source_root, args.bundle)
    except (DatasetBuilderError, publication.PublicationError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.write(publication.canonical_json(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a command
    raise SystemExit(main())
