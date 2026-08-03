"""Shared eval-generation and historical replay semantics.

This module is deliberately independent of Forge runtime imports so reporting
and collection can use one small contract for interpreting stored eval rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BaseConfigurationIdentity = tuple[Any, Any, Any, Any, Any, Any]
ExplicitPolicyIdentity = tuple[Any, Any, Any, Any, Any, Any, Any]


@dataclass
class GenerationMaxima:
    """Bounded selection state keyed by configuration and explicit policy."""

    base: dict[BaseConfigurationIdentity, int] = field(default_factory=dict)
    explicit_policy: dict[ExplicitPolicyIdentity, int] = field(
        default_factory=dict
    )


def accumulate_generation_maxima(
    maxima: GenerationMaxima, row: dict[str, Any]
) -> None:
    """Include one row in the bounded maxima used by generation selection."""
    generation = effective_generation(row)
    base_identity = base_configuration_identity(row)
    if generation > maxima.base.get(base_identity, -1):
        maxima.base[base_identity] = generation
    if "reasoning_replay" in row:
        policy_identity = explicit_policy_identity(row)
        if generation > maxima.explicit_policy.get(policy_identity, -1):
            maxima.explicit_policy[policy_identity] = generation


def selection_maximum_generation(
    maxima: GenerationMaxima, row: dict[str, Any]
) -> int:
    """Return the maximum generation governing one row's selection."""
    if "reasoning_replay" in row:
        return maxima.explicit_policy[explicit_policy_identity(row)]
    return maxima.base[base_configuration_identity(row)]


def is_selected_generation(maxima: GenerationMaxima, row: dict[str, Any]) -> bool:
    """Apply the report-compatible selection predicate to one row."""
    return effective_generation(row) == selection_maximum_generation(maxima, row)


def effective_generation(row: dict[str, Any]) -> int:
    """Return a row's generation, treating a missing field as generation 0."""
    return row.get("gen", 0)


def effective_reasoning_replay(row: dict[str, Any]) -> str:
    """Return replay policy, treating a legacy missing field as ``full``."""
    return row.get("reasoning_replay", "full")


def base_configuration_identity(row: dict[str, Any]) -> BaseConfigurationIdentity:
    """Return the whole-configuration identity used for legacy supersession."""
    return (
        row["model"],
        row["backend"],
        row["mode"],
        row.get("ablation", "reforged"),
        row.get("tool_choice", "auto"),
        row.get("reasoning_level", "default"),
    )


def explicit_policy_identity(row: dict[str, Any]) -> ExplicitPolicyIdentity:
    """Return base identity plus the row's stored replay-policy value."""
    return base_configuration_identity(row) + (row.get("reasoning_replay"),)


def select_latest_generation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select report-compatible newest-generation rows in input order.

    Legacy rows without an explicit replay field are superseded by the newest
    generation of their base configuration. Rows with an explicit replay field
    are superseded only by newer rows of the same explicit policy. Every row at
    the relevant maximum generation is retained.

    Selection is whole-configuration rather than per-scenario. The returned
    list contains the original row objects and preserves their input order.
    """
    maxima = GenerationMaxima()
    for row in rows:
        accumulate_generation_maxima(maxima, row)

    return [row for row in rows if is_selected_generation(maxima, row)]
