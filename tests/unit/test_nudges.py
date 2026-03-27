"""Tests for forge.prompts.nudges — retry, step, and prerequisite nudge templates."""

from forge.prompts.nudges import prerequisite_nudge, retry_nudge, step_nudge


class TestRetryNudge:
    def test_returns_non_empty_string(self) -> None:
        result = retry_nudge("some raw output")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_does_not_echo_raw_response(self) -> None:
        raw = "I think the answer is 42"
        result = retry_nudge(raw)
        assert raw not in result

    def test_mentions_tool_call(self) -> None:
        result = retry_nudge("whatever")
        assert "tool call" in result.lower()


class TestStepNudge:
    def test_returns_non_empty_string(self) -> None:
        result = step_nudge("submit_answer", ["get_data", "analyze"])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_terminal_tool_name(self) -> None:
        result = step_nudge("submit_answer", ["get_data"])
        assert "submit_answer" in result

    def test_contains_pending_steps(self) -> None:
        result = step_nudge("submit", ["get_pricing", "get_history"])
        assert "get_pricing" in result
        assert "get_history" in result

    def test_single_pending_step(self) -> None:
        result = step_nudge("done", ["fetch_data"])
        assert "fetch_data" in result

    def test_tier1_is_default(self) -> None:
        result = step_nudge("submit", ["fetch"])
        assert "cannot call submit yet" in result.lower()

    def test_tier2_direct(self) -> None:
        result = step_nudge("submit", ["fetch", "analyze"], tier=2)
        assert "must call one of these tools now" in result.lower()
        assert "fetch" in result
        assert "analyze" in result

    def test_tier3_aggressive(self) -> None:
        result = step_nudge("submit", ["fetch"], tier=3)
        assert "STOP" in result
        assert "Do NOT call submit" in result
        assert "fetch" in result

    def test_tier_clamped_below(self) -> None:
        result_t0 = step_nudge("submit", ["fetch"], tier=0)
        result_t1 = step_nudge("submit", ["fetch"], tier=1)
        assert result_t0 == result_t1

    def test_tier_clamped_above(self) -> None:
        result_t5 = step_nudge("submit", ["fetch"], tier=5)
        result_t3 = step_nudge("submit", ["fetch"], tier=3)
        assert result_t5 == result_t3


class TestPrerequisiteNudge:
    def test_returns_non_empty_string(self) -> None:
        result = prerequisite_nudge("edit_file", ["read_file"])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_tool_name(self) -> None:
        result = prerequisite_nudge("edit_file", ["read_file"])
        assert "edit_file" in result

    def test_contains_missing_prereqs(self) -> None:
        result = prerequisite_nudge("edit_file", ["read_file", "authenticate"])
        assert "read_file" in result
        assert "authenticate" in result

    def test_single_missing_prereq(self) -> None:
        result = prerequisite_nudge("edit_file", ["read_file"])
        assert "read_file" in result
