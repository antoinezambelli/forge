"""Shared human-readable provenance for Forge eval generations."""

from __future__ import annotations


# An eval generation is a comparability epoch, not a release version.  Keep
# this object shared: report rendering and dataset publication both expose the
# exact same provenance strings.
GEN_INFO: dict[int, dict[str, str]] = {
    1: {
        "commit": "2b05dc4",
        "date": "2026-05-08",
        "note": "v0.6.0 suite — incl. Anthropic ablation",
    },
    2: {
        "commit": "655e1f6",
        "date": "2026-05-22",
        "note": "v0.7.0 lineup refresh (8–14B) + 32GB tier debut (v0.7.4)",
    },
    # Tag ref, not a commit SHA: gen 3 landed via a branch whose squash-merge
    # SHA didn't exist when this entry was written; the v0.7.5 tag resolves to it.
    3: {
        "commit": "v0.7.5",
        "date": "2026-06-11",
        "note": (
            "reasoning-replay grid (8–14B × none/keep-last/full) "
            "+ Claude thinking-on baseline"
        ),
    },
}
