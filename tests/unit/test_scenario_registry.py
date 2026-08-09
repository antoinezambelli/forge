"""Registry and shape contracts for model-reasoning eval scenarios."""

from tests.eval.scenarios import (
    ALL_SCENARIOS,
    argument_transformation,
    argument_transformation_stateful,
    data_gap_recovery_extended,
    data_gap_recovery_extended_stateful,
    grounded_synthesis,
    grounded_synthesis_stateful,
    inconsistent_api_recovery,
    inconsistent_api_recovery_stateful,
)


def test_model_reasoning_scenarios_match_registry_contract() -> None:
    registered = {scenario.name: scenario for scenario in ALL_SCENARIOS}
    expected_entries = {
        "argument_transformation": argument_transformation,
        "argument_transformation_stateful": argument_transformation_stateful,
        "data_gap_recovery_extended": data_gap_recovery_extended,
        "data_gap_recovery_extended_stateful": data_gap_recovery_extended_stateful,
        "grounded_synthesis": grounded_synthesis,
        "grounded_synthesis_stateful": grounded_synthesis_stateful,
        "inconsistent_api_recovery": inconsistent_api_recovery,
        "inconsistent_api_recovery_stateful": inconsistent_api_recovery_stateful,
    }
    for name, scenario in expected_entries.items():
        assert scenario.name == name
        assert registered[name] is scenario

    argument_workflow = argument_transformation.workflow
    assert argument_workflow is not None
    assert len(argument_workflow.tools) == 7
    assert argument_workflow.terminal_tool == "submit_audit_report"
    assert argument_workflow.required_steps == [
        "list_transactions",
        "get_approved_vendors",
    ]
    assert argument_transformation.ideal_iterations == 5
    assert argument_transformation.max_iterations == 15
    assert {"model_quality", "reasoning"} <= set(argument_transformation.tags)

    data_gap_workflow = data_gap_recovery_extended.workflow
    assert data_gap_workflow is not None
    assert len(data_gap_workflow.tools) == 12
    assert data_gap_workflow.terminal_tool == "submit_report"
    assert data_gap_workflow.required_steps == ["get_employee"]
    assert data_gap_recovery_extended.ideal_iterations == 8
    assert data_gap_recovery_extended.max_iterations == 20

    grounded_workflow = grounded_synthesis.workflow
    assert grounded_workflow is not None
    assert set(grounded_workflow.tools) == {
        "get_open_role",
        "get_candidate_pool",
        "get_skill_summary",
        "get_compatibility_check",
        "get_team_dynamics",
        "submit_hiring_decision",
    }
    assert grounded_workflow.terminal_tool == "submit_hiring_decision"
    assert {"get_open_role", "get_candidate_pool"} <= set(
        grounded_workflow.required_steps
    )
    assert grounded_synthesis.ideal_iterations == 10
    assert grounded_synthesis.max_iterations == 20
    assert "advanced_reasoning" in grounded_synthesis.tags

    recovery_workflow = inconsistent_api_recovery.workflow
    assert recovery_workflow is not None
    assert set(recovery_workflow.tools) == {
        "legacy_list_accounts",
        "legacy_get_balance",
        "legacy_get_transactions",
        "legacy_categorize_spend",
        "legacy_check_compliance",
        "legacy_aggregate_subtotal",
        "legacy_submit_audit",
    }
    assert recovery_workflow.terminal_tool == "legacy_submit_audit"
    assert "legacy_list_accounts" in recovery_workflow.required_steps
    assert inconsistent_api_recovery.ideal_iterations == 8
    assert inconsistent_api_recovery.max_iterations == 20
    assert {"advanced_reasoning", "error_recovery"} <= set(
        inconsistent_api_recovery.tags
    )

    assert "stateful" in grounded_synthesis_stateful.tags
    assert grounded_synthesis_stateful.build_workflow is not None
    assert {"stateful", "advanced_reasoning", "error_recovery"} <= set(
        inconsistent_api_recovery_stateful.tags
    )
    assert inconsistent_api_recovery_stateful.build_workflow is not None
