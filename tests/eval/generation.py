"""Shared eval-generation and historical replay semantics.

This module is deliberately independent of Forge runtime imports so reporting
and collection can use one small contract for interpreting stored eval rows.
"""

from __future__ import annotations

from typing import Any


BaseConfigurationIdentity = tuple[Any, Any, Any, Any, Any, Any]
ExplicitPolicyIdentity = tuple[Any, Any, Any, Any, Any, Any, Any]


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
    base_max: dict[BaseConfigurationIdentity, int] = {}
    policy_max: dict[ExplicitPolicyIdentity, int] = {}
    for row in rows:
        generation = effective_generation(row)
        base_identity = base_configuration_identity(row)
        policy_identity = explicit_policy_identity(row)
        if generation > base_max.get(base_identity, -1):
            base_max[base_identity] = generation
        if generation > policy_max.get(policy_identity, -1):
            policy_max[policy_identity] = generation

    selected: list[dict[str, Any]] = []
    for row in rows:
        generation = effective_generation(row)
        if "reasoning_replay" in row:
            if generation == policy_max[explicit_policy_identity(row)]:
                selected.append(row)
        elif generation == base_max[base_configuration_identity(row)]:
            selected.append(row)
    return selected
