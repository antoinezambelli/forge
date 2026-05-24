#!/usr/bin/env python3
"""Pre-commit hook: scan staged files for PII and sensitive data.

Exit 0 if clean, exit 1 if suspicious content found.
"""

import re
import subprocess
import sys

# Patterns that should never appear in committed code
PATTERNS = [
    ("macOS user path", r"/Users/\w+[^/]*(?:/\.ssh|/\.gnupg|/\.local|/\.zsh|/\.cache|/\.npm|/\.config)"),
    ("zsh history", r"\.zsh_history"),
    ("SSH private key", r"ssh-(rsa|ed25519|ecdsa|dsa)\s"),
    ("PEM private key", r"-----BEGIN.*PRIVATE KEY-----"),
    ("netrc file", r"machine\s+\S+\s+login"),
    ("email address (non-noreply)", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?<!noreply)"),
    ("phone number", r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    ("password/secret assignment", r"(?:password|secret|api_key|token)\s*[:=]\s*['\"]?\w{8,}['\"]?"),
    ("AWS credentials", r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    ("private IP range", r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)\d{1,3}\.\d{1,3}\b"),
]


def main() -> int:
    # Get list of staged files (added/modified only)
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACDMR"],
        capture_output=True,
        text=True,
    )
    files = [f for f in result.stdout.strip().split("\n") if f]
    if not files:
        return 0

    # Skip ourselves and other PII checkers to avoid false positives
    skip_files = {"scripts/check_pii.py"}
    files = [f for f in files if f not in skip_files]
    if not files:
        return 0

    # Get the staged content for non-skipped files only
    result = subprocess.run(
        ["git", "diff", "--cached", "--diff-filter=ACDM", "-U0", "--", *files],
        capture_output=True,
        text=True,
    )
    diff = result.stdout

    # Only scan added lines (start with '+'), skip diff headers ('---'/'+++')
    added_lines = [
        line[1:]  # strip leading '+'
        for line in diff.split("\n")
        if line.startswith("+") and not line.startswith("+++")
    ]
    added_text = "\n".join(added_lines)

    found = []
    for name, pattern in PATTERNS:
        for match in re.finditer(pattern, added_text, re.IGNORECASE):
            # Skip lines that are clearly regex definitions (contain r' or r")
            snippet = match.group()[:60]
            # Skip if the matched text looks like it's inside a string literal definition
            found.append((name, snippet))

    if not found:
        print("PII check: clean")
        return 0

    print("PII check: FAILED — suspicious content in staged changes:")
    for name, snippet in found:
        print(f"  [{name}] {snippet}")
    print("\nReview and remove sensitive data before committing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
