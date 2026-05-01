"""One-shot migration: rewrite llamaserver/llamafile rows to GGUF-stem identity.

Pre-v0.6.0 schema: every row's `model` field stored an Ollama-style string
regardless of backend. Post-refactor (Step 5 of the GGUF-as-identity work),
llamaserver and llamafile rows store the *file stem* as the canonical
identity. This script translates the historical rows.

Usage:
    python scripts/migrate_eval_jsonl_gguf_identity.py \
        --input eval_results.jsonl \
        --output eval_results_migrated.jsonl

Translates rows where backend is "llamaserver" or "llamafile". Rows for
"ollama" and "anthropic" are passed through unchanged. Aborts on unknown
identities (data we don't recognize is data we eyeball, not silently drop).

The translation table below is a one-shot copy of the deleted GGUF_MAP /
LLAMAFILE_MAP from tests/eval/batch_eval.py — pinned in time so this
script keeps working after the maps were deleted.

After running, manually inspect the output, then:
    mv eval_results_migrated.jsonl eval_results.jsonl

The source is never modified — safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Pinned-in-time translation tables. These are the GGUF_MAP and LLAMAFILE_MAP
# that lived in tests/eval/batch_eval.py before the GGUF-as-identity refactor.
# DO NOT import from batch_eval — the maps no longer exist there.

_GGUF_TRANSLATION: dict[str, str] = {
    # ollama-style key -> GGUF stem (filename minus .gguf)
    "llama3.1:8b-instruct-q4_K_M":           "Meta-Llama-3.1-8B-Instruct-Q4_K_M",
    "llama3.1:8b-instruct-q8_0":             "Meta-Llama-3.1-8B-Instruct-Q8_0",
    "mistral-nemo:12b-instruct-2407-q4_K_M": "Mistral-Nemo-Instruct-2407-Q4_K_M",
    "mistral:7b-instruct-v0.3-q4_K_M":       "Mistral-7B-Instruct-v0.3-Q4_K_M",
    "mistral:7b-instruct-v0.3-q8_0":         "Mistral-7B-Instruct-v0.3-Q8_0",
    "qwen3:8b-q4_K_M":                       "Qwen3-8B-Q4_K_M",
    "qwen3:8b-q8_0":                         "Qwen3-8B-Q8_0",
    "qwen3:14b-q4_K_M":                      "Qwen3-14B-Q4_K_M",
    "ministral-3:8b-instruct-2512-q4_K_M":   "Ministral-3-8B-Instruct-2512-Q4_K_M",
    "ministral-3:8b-instruct-2512-q8_0":     "Ministral-3-8B-Instruct-2512-Q8_0",
    "ministral-3:14b-instruct-2512-q4_K_M":  "Ministral-3-14B-Instruct-2512-Q4_K_M",
    "ministral-3:8b-reasoning-2512-q4_K_M":  "Ministral-3-8B-Reasoning-2512-Q4_K_M",
    "ministral-3:8b-reasoning-2512-q8_0":    "Ministral-3-8B-Reasoning-2512-Q8_0",
    "ministral-3:14b-reasoning-2512-q4_K_M": "Ministral-3-14B-Reasoning-2512-Q4_K_M",
    "gemma4:31b-it-q4_K_M":                  "gemma-4-31B-it-Q4_K_M",
    "gemma4:26b-a4b-it-q4_K_M":              "gemma-4-26B-A4B-it-UD-Q4_K_M",
    "gemma4:26b-a4b-it-q8_0":                "gemma-4-26B-A4B-it-Q8_0",
    "gemma4:e4b-it-q4_K_M":                  "gemma-4-E4B-it-Q4_K_M",
    "gemma4:e4b-it-q8_0":                    "gemma-4-E4B-it-Q8_0",
    "mistral-small-3.2:24b-instruct-2506-q4_K_M": "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M",
    "mistral-small-3.2:24b-instruct-2506-q8_0":   "Mistral-Small-3.2-24B-Instruct-2506-Q8_0",
    "devstral-small-2:24b-instruct-2512-q4_K_M":  "Devstral-Small-2-24B-Instruct-2512-Q4_K_M",
    "devstral-small-2:24b-instruct-2512-q8_0":    "Devstral-Small-2-24B-Instruct-2512-Q8_0",
    "qwen3.5:27b-q4_K_M":                    "Qwen3.5-27B-Q4_K_M",
    "qwen3.5:35b-a3b-q4_K_M":                "Qwen3.5-35B-A3B-Q4_K_M",
    "granite-4.0:h-micro-q4_K_M":            "granite-4.0-h-micro-Q4_K_M",
    "granite-4.0:h-micro-q8_0":              "granite-4.0-h-micro-Q8_0",
    "granite-4.0:h-tiny-q4_K_M":             "granite-4.0-h-tiny-Q4_K_M",
    "granite-4.0:h-tiny-q8_0":               "granite-4.0-h-tiny-Q8_0",
}

_LLAMAFILE_TRANSLATION: dict[str, str] = {
    # ollama-style key -> llamafile stem (filename minus .llamafile)
    "llama3.1:8b-instruct-q4_K_M":           "Meta-Llama-3.1-8B-Instruct.Q4_K_M",
    "llama3.1:8b-instruct-q8_0":             "Meta-Llama-3.1-8B-Instruct.Q8_0",
    "mistral-nemo:12b-instruct-2407-q4_K_M": "Mistral-Nemo-Instruct-2407.Q4_K_M",
    "mistral:7b-instruct-v0.3-q4_K_M":       "Mistral-7B-Instruct-v0.3.Q4_K_M",
    "mistral:7b-instruct-v0.3-q8_0":         "Mistral-7B-Instruct-v0.3.Q8_0",
}


def translate_model(backend: str, model: str) -> str | None:
    """Return the new model identity, or None if the row should pass through unchanged.

    Raises KeyError if backend is local-server but model is not in the
    translation table — caller should treat as fatal.
    """
    if backend == "llamaserver":
        if model not in _GGUF_TRANSLATION:
            raise KeyError(f"unknown llamaserver model: {model!r}")
        return _GGUF_TRANSLATION[model]
    if backend == "llamafile":
        if model not in _LLAMAFILE_TRANSLATION:
            raise KeyError(f"unknown llamafile model: {model!r}")
        return _LLAMAFILE_TRANSLATION[model]
    # ollama / anthropic / anything else: pass-through
    return None


def migrate(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(2)
    if output_path.exists():
        print(f"ERROR: output exists, refusing to overwrite: {output_path}", file=sys.stderr)
        print(f"  delete it manually if you want to re-run.", file=sys.stderr)
        sys.exit(2)

    in_count = 0
    out_count = 0
    by_backend: Counter[str] = Counter()
    translated: Counter[str] = Counter()
    passthrough: Counter[str] = Counter()
    sample_translations: dict[tuple[str, str], str] = {}
    unknown: dict[str, list[int]] = defaultdict(list)

    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            in_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"ERROR: malformed JSON at line {line_no}: {e}", file=sys.stderr)
                sys.exit(3)

            backend = row.get("backend", "?")
            model = row.get("model", "?")
            by_backend[backend] += 1

            try:
                new_model = translate_model(backend, model)
            except KeyError as e:
                unknown[f"{backend}:{model}"].append(line_no)
                continue  # collect all unknowns before erroring

            if new_model is not None:
                # Sample for sanity-check display
                key = (backend, model)
                if key not in sample_translations:
                    sample_translations[key] = new_model
                row["model"] = new_model
                translated[backend] += 1
            else:
                passthrough[backend] += 1

            fout.write(json.dumps(row) + "\n")
            out_count += 1

    if unknown:
        # Cleanup: don't leave a partial output file around when aborting
        output_path.unlink(missing_ok=True)
        print("\nERROR: unknown (backend, model) combinations encountered:", file=sys.stderr)
        for key, lines in sorted(unknown.items()):
            print(f"  {key}: {len(lines)} rows (first at line {lines[0]})", file=sys.stderr)
        print(
            "\nMigration aborted. Either add the model to the translation table "
            "or investigate the data.",
            file=sys.stderr,
        )
        sys.exit(4)

    # Summary
    print(f"\n{'='*60}")
    print(f"Migration complete")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"\nRows in:    {in_count:,}")
    print(f"Rows out:   {out_count:,}")
    if in_count != out_count:
        print(f"WARNING: row count mismatch ({in_count - out_count} dropped)")

    print(f"\nBy backend (input):")
    for backend, count in sorted(by_backend.items(), key=lambda x: -x[1]):
        t = translated.get(backend, 0)
        p = passthrough.get(backend, 0)
        print(f"  {backend:14s} {count:>7,}   (translated={t:>7,}, passthrough={p:>7,})")

    print(f"\nSample translations (one per source identity):")
    for (backend, old_model), new_model in sorted(sample_translations.items()):
        print(f"  [{backend:11s}] {old_model:48s} -> {new_model}")

    print(f"\nNext step: eyeball the output, then:")
    print(f"  mv {output_path.name} {input_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("eval_results.jsonl"),
        help="Input JSONL (default: eval_results.jsonl)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("eval_results_migrated.jsonl"),
        help="Output JSONL (default: eval_results_migrated.jsonl)",
    )
    args = parser.parse_args()
    migrate(args.input, args.output)


if __name__ == "__main__":
    main()
