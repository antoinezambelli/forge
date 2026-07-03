"""Reproducer for forge issue #121: Path.stem truncates dotted model names.

Demonstrates the bug when bare model names (no file extension) are passed
as gguf_path, e.g. in proxy mode: LlamafileClient(gguf_path="mimo-v2.5").

The bug: Path("mimo-v2.5").stem returns "mimo-v2" because Python sees ".5"
as a file extension.
"""

from __future__ import annotations

import re
from pathlib import Path

# Try to import the actual fixed function from the installed package
try:
    from forge.clients.llamafile import LlamafileClient
    HAS_PKG = True
except ImportError:
    HAS_PKG = False

# Standalone copy of the fixed logic (for environments where forge isn't installed)
_SHARD_SUFFIX_RE = re.compile(r"-\d{5}-of-\d{5}$")
_KNOWN_GGUF_EXTENSIONS: tuple[str, ...] = (".gguf", ".llamafile")


def _model_name_fixed(path: str | Path) -> str:
    name = Path(path).name
    for ext in _KNOWN_GGUF_EXTENSIONS:
        name = name.removesuffix(ext)
    return _SHARD_SUFFIX_RE.sub("", name)


def _model_name_buggy(path: str | Path) -> str:
    return _SHARD_SUFFIX_RE.sub("", Path(path).stem)


CASES = [
    ("mimo-v2.5", "mimo-v2.5"),
    ("Model.Q4_K_M", "Model.Q4_K_M"),
    ("qwen3:8b-q4_K_M", "qwen3:8b-q4_K_M"),
    ("custom-model", "custom-model"),
    ("mimo-v2.5.gguf", "mimo-v2.5"),
    ("Model.Q4_K_M.llamafile", "Model.Q4_K_M"),
    ("model-00001-of-00003.gguf", "model"),
    ("llama3.gguf", "llama3"),
]

print("=" * 70)
print("Issue #121: Path.stem truncates dotted model names")
if HAS_PKG:
    print("(using installed forge package for verification)")
print("=" * 70)

all_pass = True
for filename, expected in CASES:
    buggy = _model_name_buggy(filename)
    fixed = _model_name_fixed(filename)

    if buggy == expected:
        bug_status = "OK"
    else:
        bug_status = f"BUG: got {buggy!r}"
        # Buggy behavior is expected/demonstrated; don't fail the script for it.

    if fixed == expected:
        fix_status = "FIXED"
    else:
        fix_status = f"FAIL: got {fixed!r}"
        all_pass = False

    print(f"\n  {filename}")
    print(f"    buggy    → {buggy!r:25s}  [{bug_status}]")
    print(f"    fixed    → {fixed!r:25s}  [{fix_status}]")
    print(f"    expected → {expected!r}")

    # Also verify against installed package if available
    if HAS_PKG:
        pkg_result = LlamafileClient._derive_sampling_key(filename)
        pkg_ok = "OK" if pkg_result == expected else f"MISMATCH: {pkg_result!r}"
        print(f"    pkg      → {pkg_result!r:25s}  [{pkg_ok}]")
        if pkg_result != expected:
            all_pass = False

print("\n" + "=" * 70)
if all_pass:
    print("All cases pass.")
else:
    print("Some cases failed!")
    raise SystemExit(1)
