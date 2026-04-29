"""Tests for forge.clients.sampling_defaults — per-model sampling lookup."""

import logging

import pytest

from forge.clients.sampling_defaults import (
    MODEL_SAMPLING_DEFAULTS,
    _WARNED_UNKNOWN,
    get_sampling_defaults,
)


class TestGetSamplingDefaults:
    """Tests for the ``get_sampling_defaults`` helper."""

    def test_known_model_returns_copy(self) -> None:
        """Known model returns its params; mutating the result doesn't affect the map."""
        model = "qwen3:8b-q4_K_M"
        assert model in MODEL_SAMPLING_DEFAULTS
        result = get_sampling_defaults(model)
        assert result["temperature"] == 0.6
        assert result["top_p"] == 0.95
        assert result["top_k"] == 20
        assert result["min_p"] == 0.0
        result["temperature"] = 999.0
        # Original map unchanged
        assert MODEL_SAMPLING_DEFAULTS[model]["temperature"] == 0.6

    def test_unknown_model_returns_empty(self) -> None:
        """Unknown model returns empty dict (no recommendation)."""
        result = get_sampling_defaults("some-model-we-dont-know:7b")
        assert result == {}

    def test_qwen3_5_uses_general_tasks_profile(self) -> None:
        """Qwen3.5/3.6 card gives a general-tasks profile with presence_penalty=1.5."""
        result = get_sampling_defaults("qwen3.5:27b-q4_K_M")
        assert result["temperature"] == 1.0
        assert result["top_p"] == 0.95
        assert result["top_k"] == 20
        assert result["min_p"] == 0.0
        assert result["presence_penalty"] == 1.5

    def test_qwen3_coder_uses_repeat_penalty(self) -> None:
        """Qwen3-Coder card specifies repeat_penalty=1.05, no min_p."""
        result = get_sampling_defaults("qwen3-coder:30b-a3b-instruct-q4_K_M")
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.8
        assert result["top_k"] == 20
        assert result["repeat_penalty"] == 1.05
        assert "min_p" not in result
        assert "presence_penalty" not in result

    def test_unknown_model_logs_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown model warns once per session, not every call."""
        model = "unique-model-for-log-test:1b"
        # Make sure it's not in the "already warned" set from another test
        _WARNED_UNKNOWN.discard(model)

        with caplog.at_level(logging.WARNING, logger="forge.clients.sampling_defaults"):
            get_sampling_defaults(model)
            get_sampling_defaults(model)
            get_sampling_defaults(model)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert model in warnings[0].getMessage()
