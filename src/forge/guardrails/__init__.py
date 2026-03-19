"""Composable guardrail middleware for external agent loops.

Use these components inside your own orchestration loop to get forge's
reliability stack (retry nudges, rescue parsing, step enforcement, error
tracking) without adopting WorkflowRunner.

Most integrators should use ``Guardrails`` (two-method API). For granular
control, use ResponseValidator, StepEnforcer, and ErrorTracker directly.

See ADR-011 (docs/decisions/011-guardrail-middleware.md) for design rationale.
"""

from forge.guardrails.nudge import Nudge
from forge.guardrails.response_validator import ResponseValidator, ValidationResult
from forge.guardrails.step_enforcer import StepEnforcer, StepCheck
from forge.guardrails.error_tracker import ErrorTracker
from forge.guardrails.guardrails import CheckResult, Guardrails

__all__ = [
    # Bundled API
    "CheckResult",
    "Guardrails",
    # Granular components
    "ErrorTracker",
    "Nudge",
    "ResponseValidator",
    "StepCheck",
    "StepEnforcer",
    "ValidationResult",
]
