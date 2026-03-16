"""Eval scenario definitions — re-exports from submodules."""

from ._base import EvalScenario, _check
from ._compaction import phase2_compaction, relevance_detection
from ._model_quality import (
    argument_fidelity,
    conditional_routing,
    data_gap_recovery,
    sequential_reasoning,
    tool_selection,
)
from ._plumbing import basic_2step, compaction_stress, error_recovery, sequential_3step
from ._compaction_chain import (
    compaction_chain_baseline,
    compaction_chain_p1,
    compaction_chain_p2,
    compaction_chain_p3,
)
from ._stateful_compaction import (
    compaction_stress_stateful,
    inventory_audit,
    phase2_compaction_stateful,
    relevance_detection_stateful,
    supplier_deep_dive,
)
from ._stateful_model_quality import (
    argument_fidelity_stateful,
    conditional_routing_stateful,
    data_gap_recovery_stateful,
    sequential_reasoning_stateful,
    tool_selection_stateful,
)
from ._stateful_plumbing import (
    basic_2step_stateful,
    basic_2step_stateful_tre,
    error_recovery_stateful,
    sequential_3step_stateful,
)

ALL_SCENARIOS: list[EvalScenario] = [
    basic_2step,
    sequential_3step,
    compaction_stress,
    error_recovery,
    tool_selection,
    argument_fidelity,
    sequential_reasoning,
    conditional_routing,
    data_gap_recovery,
    phase2_compaction,
    relevance_detection,
    # Stateful scenarios
    basic_2step_stateful,
    basic_2step_stateful_tre,
    sequential_3step_stateful,
    error_recovery_stateful,
    tool_selection_stateful,
    argument_fidelity_stateful,
    sequential_reasoning_stateful,
    conditional_routing_stateful,
    data_gap_recovery_stateful,
    compaction_stress_stateful,
    phase2_compaction_stateful,
    inventory_audit,
    supplier_deep_dive,
    relevance_detection_stateful,
    compaction_chain_baseline,
    compaction_chain_p1,
    compaction_chain_p2,
    compaction_chain_p3,
]

__all__ = [
    "EvalScenario",
    "_check",
    "ALL_SCENARIOS",
    "basic_2step",
    "sequential_3step",
    "compaction_stress",
    "error_recovery",
    "tool_selection",
    "argument_fidelity",
    "sequential_reasoning",
    "conditional_routing",
    "data_gap_recovery",
    "phase2_compaction",
    "relevance_detection",
    "basic_2step_stateful",
    "basic_2step_stateful_tre",
    "sequential_3step_stateful",
    "error_recovery_stateful",
    "tool_selection_stateful",
    "argument_fidelity_stateful",
    "sequential_reasoning_stateful",
    "conditional_routing_stateful",
    "data_gap_recovery_stateful",
    "compaction_stress_stateful",
    "phase2_compaction_stateful",
    "inventory_audit",
    "supplier_deep_dive",
    "relevance_detection_stateful",
    "compaction_chain_baseline",
    "compaction_chain_p1",
    "compaction_chain_p2",
    "compaction_chain_p3",
]
