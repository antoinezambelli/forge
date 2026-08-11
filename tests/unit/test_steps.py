"""Unit tests for forge.core.steps.StepTracker."""

from forge.core.steps import StepTracker


class TestStepTracker:
    def test_empty_required_steps_is_satisfied(self):
        tracker = StepTracker(required_steps=[])
        assert tracker.is_satisfied() is True

    def test_partial_completion_not_satisfied(self):
        tracker = StepTracker(required_steps=["get_pricing", "get_history"])
        tracker.record("get_pricing")
        assert tracker.is_satisfied() is False

    def test_full_completion_is_satisfied(self):
        tracker = StepTracker(required_steps=["get_pricing", "get_history"])
        tracker.record("get_pricing")
        tracker.record("get_history")
        assert tracker.is_satisfied() is True

    def test_pending_preserves_original_order(self):
        tracker = StepTracker(
            required_steps=["z_step", "a_step", "m_step", "b_step"]
        )
        tracker.record("a_step")
        tracker.record("b_step")
        assert tracker.pending() == ["z_step", "m_step"]

    def test_record_is_idempotent(self):
        tracker = StepTracker(required_steps=["step_a"])
        tracker.record("step_a")
        tracker.record("step_a")
        assert tracker.is_satisfied() is True
        assert len(tracker.completed_steps) == 1

    def test_summary_hint_no_completions(self):
        tracker = StepTracker(required_steps=["step_a", "step_b"])
        assert tracker.summary_hint() == "[No steps completed yet]"

    def test_summary_hint_execution_order_not_alphabetical(self):
        tracker = StepTracker(required_steps=["c_step", "a_step", "b_step"])
        tracker.record("b_step")
        tracker.record("c_step")
        tracker.record("a_step")
        assert tracker.summary_hint() == "[Steps completed: b_step, c_step, a_step]"

    def test_recording_non_required_step(self):
        tracker = StepTracker(required_steps=["step_a"])
        tracker.record("unrelated_tool")
        assert tracker.is_satisfied() is False
        assert "unrelated_tool" in tracker.completed_steps
