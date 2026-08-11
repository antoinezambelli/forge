"""Tests for forge.prompts.nudges — retry, step, and prerequisite nudge templates."""

from forge.prompts.nudges import (
    prerequisite_nudge,
    retry_nudge,
    step_nudge,
    tool_arg_validation_nudge,
)


class TestRetryNudge:
    def test_describes_retry_without_echoing_raw_response(self) -> None:
        raw = "I think the answer is 42"
        result = retry_nudge(raw)
        assert isinstance(result, str)
        assert result
        assert raw not in result
        assert "tool call" in result.lower()


class TestStepNudge:
    def test_default_names_terminal_and_pending_steps(self) -> None:
        result = step_nudge("submit", ["fetch", "analyze"])
        assert isinstance(result, str)
        assert result
        assert "cannot call submit yet" in result.lower()
        assert "fetch" in result
        assert "analyze" in result

    def test_escalation_tiers_change_urgency(self) -> None:
        direct = step_nudge("submit", ["fetch", "analyze"], tier=2)
        aggressive = step_nudge("submit", ["fetch"], tier=3)
        assert "must call one of these tools now" in direct.lower()
        assert "fetch" in direct
        assert "analyze" in direct
        assert "STOP" in aggressive
        assert "Do NOT call submit" in aggressive
        assert "fetch" in aggressive

    def test_tier_is_clamped_to_supported_range(self) -> None:
        assert step_nudge("submit", ["fetch"], tier=0) == step_nudge(
            "submit", ["fetch"], tier=1
        )
        assert step_nudge("submit", ["fetch"], tier=5) == step_nudge(
            "submit", ["fetch"], tier=3
        )


class TestPrerequisiteNudge:
    def test_names_tool_and_every_missing_prerequisite(self) -> None:
        result = prerequisite_nudge("edit_file", ["read_file", "authenticate"])
        assert isinstance(result, str)
        assert result
        assert "edit_file" in result
        assert "read_file" in result
        assert "authenticate" in result


class TestToolArgValidationNudge:
    def test_names_tool_and_required_shape(self) -> None:
        result = tool_arg_validation_nudge("edit_file", "")
        assert isinstance(result, str)
        assert result
        assert "edit_file" in result
        assert "JSON object" in result or "dict" in result

    def test_reports_received_argument_shape(self) -> None:
        cases = [
            ("string", "", "str"),
            ("none", None, "NoneType"),
            ("list", [1, 2, 3], "list"),
        ]
        for label, args, expected_type in cases:
            result = tool_arg_validation_nudge("edit", args)
            assert expected_type in result, label
