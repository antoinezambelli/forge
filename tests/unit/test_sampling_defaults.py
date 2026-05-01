"""Tests for forge.clients.sampling_defaults — per-model sampling lookup."""

import logging

import pytest

from forge.clients.sampling_defaults import (
    MODEL_SAMPLING_DEFAULTS,
    _INFO_LOGGED,
    apply_sampling_defaults,
    get_sampling_defaults,
)
from forge.errors import UnsupportedModelError


class TestGetSamplingDefaults:
    """Tests for the pure-lookup ``get_sampling_defaults`` helper."""

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

    def test_get_does_not_log_for_unknown_model(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``get_sampling_defaults`` is pure — no logs on unknown models.
        Logging policy lives in ``apply_sampling_defaults``.
        """
        with caplog.at_level(logging.DEBUG, logger="forge.clients.sampling_defaults"):
            get_sampling_defaults("some-model-we-dont-know:7b")
        assert caplog.records == []


class TestApplySamplingDefaults:
    """Tests for the policy-layer ``apply_sampling_defaults`` helper."""

    def test_strict_known_model_returns_dict(self) -> None:
        """strict=True + known model: returns a copy of the map entry."""
        model = "qwen3:8b-q4_K_M"
        result = apply_sampling_defaults(model, strict=True)
        assert result["temperature"] == 0.6
        # Mutating result doesn't affect the map.
        result["temperature"] = 999.0
        assert MODEL_SAMPLING_DEFAULTS[model]["temperature"] == 0.6

    def test_strict_unknown_model_raises(self) -> None:
        """strict=True + unknown model: raises ``UnsupportedModelError``."""
        with pytest.raises(UnsupportedModelError) as excinfo:
            apply_sampling_defaults("nonexistent-model:1b", strict=True)
        assert excinfo.value.model == "nonexistent-model:1b"
        assert "nonexistent-model:1b" in str(excinfo.value)

    def test_non_strict_known_model_returns_empty_and_logs_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """strict=False + known model: returns {}, fires INFO log once per session."""
        model = "qwen3:14b-q4_K_M"
        # Reset the logged-once set so this test runs cleanly regardless of order.
        _INFO_LOGGED.discard(model)

        with caplog.at_level(logging.INFO, logger="forge.clients.sampling_defaults"):
            r1 = apply_sampling_defaults(model, strict=False)
            r2 = apply_sampling_defaults(model, strict=False)
            r3 = apply_sampling_defaults(model, strict=False)

        assert r1 == {} and r2 == {} and r3 == {}
        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(infos) == 1
        assert model in infos[0].getMessage()
        assert "recommended_sampling=True" in infos[0].getMessage()

    def test_non_strict_unknown_model_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """strict=False + unknown model: returns {}, no log."""
        with caplog.at_level(logging.DEBUG, logger="forge.clients.sampling_defaults"):
            result = apply_sampling_defaults("some-model-we-dont-know:7b", strict=False)
        assert result == {}
        assert caplog.records == []
