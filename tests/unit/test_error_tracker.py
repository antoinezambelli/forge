"""Tests for ErrorTracker."""

from forge.guardrails import ErrorTracker


class TestErrorTrackerRetries:
    """Consecutive retry tracking."""

    def test_initial_state(self):
        tracker = ErrorTracker(max_retries=3)
        assert tracker.consecutive_retries == 0
        assert tracker.retries_exhausted is False

    def test_record_and_exhaust(self):
        tracker = ErrorTracker(max_retries=2)
        tracker.record_retry()
        assert tracker.retries_exhausted is False
        tracker.record_retry()
        assert tracker.retries_exhausted is False
        tracker.record_retry()
        assert tracker.retries_exhausted is True

    def test_reset_retries(self):
        tracker = ErrorTracker(max_retries=1)
        tracker.record_retry()
        tracker.record_retry()
        assert tracker.retries_exhausted is True
        tracker.reset_retries()
        assert tracker.retries_exhausted is False
        assert tracker.consecutive_retries == 0


class TestErrorTrackerToolErrors:
    """Consecutive tool error tracking."""

    def test_initial_state(self):
        tracker = ErrorTracker(max_tool_errors=2)
        assert tracker.consecutive_tool_errors == 0
        assert tracker.tool_errors_exhausted is False

    def test_record_and_exhaust(self):
        tracker = ErrorTracker(max_tool_errors=1)
        tracker.record_result(success=False)
        assert tracker.tool_errors_exhausted is False
        tracker.record_result(success=False)
        assert tracker.tool_errors_exhausted is True

    def test_soft_error_does_not_count(self):
        tracker = ErrorTracker(max_tool_errors=1)
        tracker.record_result(success=False, is_soft_error=True)
        tracker.record_result(success=False, is_soft_error=True)
        tracker.record_result(success=False, is_soft_error=True)
        assert tracker.tool_errors_exhausted is False
        assert tracker.consecutive_tool_errors == 0

    def test_success_does_not_reset_counter(self):
        """Individual success doesn't reset — only reset_errors() does."""
        tracker = ErrorTracker(max_tool_errors=2)
        tracker.record_result(success=False)
        tracker.record_result(success=True)
        assert tracker.consecutive_tool_errors == 1

    def test_reset_errors(self):
        tracker = ErrorTracker(max_tool_errors=1)
        tracker.record_result(success=False)
        tracker.record_result(success=False)
        assert tracker.tool_errors_exhausted is True
        tracker.reset_errors()
        assert tracker.tool_errors_exhausted is False
        assert tracker.consecutive_tool_errors == 0


class TestErrorTrackerIndependence:
    """Retry and tool error counters are independent."""

    def test_retry_and_tool_error_budgets_are_independent(self):
        cases = [
            ("retry", lambda tracker: tracker.record_retry()),
            ("tool error", lambda tracker: tracker.record_result(success=False)),
        ]
        for label, record in cases:
            tracker = ErrorTracker(max_retries=1, max_tool_errors=1)
            record(tracker)
            record(tracker)
            if label == "retry":
                assert tracker.retries_exhausted is True, label
                assert tracker.tool_errors_exhausted is False, label
            else:
                assert tracker.tool_errors_exhausted is True, label
                assert tracker.retries_exhausted is False, label
