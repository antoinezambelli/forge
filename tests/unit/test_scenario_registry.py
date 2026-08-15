"""Behavioral registry contracts for eval scenarios."""

from forge.core.workflow import Workflow
from tests.eval.scenarios import ALL_SCENARIOS


MODEL_REASONING_NAMES = {
    "argument_transformation",
    "argument_transformation_stateful",
    "data_gap_recovery_extended",
    "data_gap_recovery_extended_stateful",
    "grounded_synthesis",
    "grounded_synthesis_stateful",
    "inconsistent_api_recovery",
    "inconsistent_api_recovery_stateful",
}


def _assert_valid_workflow(workflow: Workflow) -> None:
    tool_names = set(workflow.tools)
    assert workflow.terminal_tools
    assert workflow.terminal_tools <= tool_names
    assert set(workflow.required_steps) <= tool_names
    assert workflow.terminal_tools.isdisjoint(workflow.required_steps)


def test_scenario_registry_is_unique_and_well_formed() -> None:
    names = [scenario.name for scenario in ALL_SCENARIOS]

    assert len(names) == len(set(names))
    assert MODEL_REASONING_NAMES <= set(names)

    for scenario in ALL_SCENARIOS:
        assert scenario.name
        assert scenario.description
        assert scenario.user_message
        assert scenario.max_iterations > 0
        if scenario.ideal_iterations is not None:
            assert 0 < scenario.ideal_iterations <= scenario.max_iterations
        _assert_valid_workflow(scenario.workflow)


def test_registered_stateful_model_reasoning_builders_are_valid() -> None:
    registered = {scenario.name: scenario for scenario in ALL_SCENARIOS}

    for name in MODEL_REASONING_NAMES:
        scenario = registered[name]
        if not name.endswith("_stateful"):
            continue

        assert scenario.build_workflow is not None
        workflow, validate_state = scenario.build_workflow()
        _assert_valid_workflow(workflow)
        assert callable(validate_state)
