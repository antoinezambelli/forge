"""Tests for the grounded_synthesis scenario.

Verifies:
- Each tool returns the expected distilled assessment for each candidate
- The validator accepts the canonical pick (Aisha + blocker reference)
  and rejects each failure tier (picks Sarah/James/Marcus, missing
  rationale, missing pick)
- Stateful backend tracks which candidates were drilled into and
  validate_state requires both blocker candidates to have been checked
"""

from __future__ import annotations

from tests.eval.scenarios import (
    grounded_synthesis,
    grounded_synthesis_stateful,
)
from tests.eval.scenarios._model_reasoning import (
    _GS_CANDIDATES,
    _grounded_synthesis_tools,
    _validate_grounded_synthesis,
)
from tests.eval.scenarios._stateful_model_reasoning import (
    HiringDecisionSystem,
    _build_grounded_synthesis_stateful,
    _validate_grounded_synthesis_stateful,
)


# ── Lambda tool callables ───────────────────────────────────────


class TestLambdaTools:
    def _call(self, tool_name: str, **kwargs: object) -> str:
        return _grounded_synthesis_tools[tool_name].callable(**kwargs)

    # get_open_role -----------------------------------------------

    def test_open_role_states_requirements_only(self) -> None:
        out = self._call("get_open_role")
        assert "Senior Backend Engineer" in out
        assert "Payments" in out
        assert "5+ years" in out
        # Timeline anchor is qualitative ("Q3 product launch", "well
        # before the launch sprint kicks off") — no explicit number of
        # days or months, so the model must reason about typical
        # hiring/onboarding cadence to apply it.
        assert "Q3" in out
        assert "60 days" not in out
        # Constraints intentionally NOT in the role spec — they live
        # only in the per-candidate compatibility check tool.
        assert "non-compete" not in out.lower()
        assert "visa" not in out.lower()
        assert "sponsor" not in out.lower()

    # get_candidate_pool ------------------------------------------

    def test_pool_lists_all_five(self) -> None:
        out = self._call("get_candidate_pool")
        for name in (
            "Sarah Chen", "James Patel", "Aisha Nakamura",
            "Marcus Reyes", "Diana Kim",
        ):
            assert name in out
        assert "5 candidates" in out

    # get_skill_summary -------------------------------------------

    def test_skill_summary_sarah_is_attractor(self) -> None:
        out = self._call("get_skill_summary", candidate_name="Sarah Chen")
        # Reinforces the trap: glowing skills + payments domain
        assert "Stripe Connect" in out
        assert "Top decile" in out

    def test_skill_summary_james_is_attractor(self) -> None:
        out = self._call("get_skill_summary", candidate_name="James Patel")
        assert "DeepMind" in out
        assert "Top decile" in out

    def test_skill_summary_aisha_solid(self) -> None:
        out = self._call("get_skill_summary", candidate_name="Aisha Nakamura")
        assert "Square" in out
        assert "78th percentile" in out

    def test_skill_summary_marcus_weaker(self) -> None:
        out = self._call("get_skill_summary", candidate_name="Marcus Reyes")
        assert "no payments background" in out
        assert "5" in out  # 5 years

    def test_skill_summary_unknown_returns_none(self) -> None:
        out = self._call("get_skill_summary", candidate_name="Unknown Person")
        assert "No candidate found" in out

    def test_skill_summary_partial_match(self) -> None:
        # Surname-only lookup should resolve.
        out = self._call("get_skill_summary", candidate_name="Nakamura")
        assert "Square" in out

    # get_compatibility_check -------------------------------------

    def test_compat_sarah_implicit_blocker(self) -> None:
        out = self._call("get_compatibility_check", candidate_name="Sarah Chen")
        # No NO/YES headers — model must read narrative and infer
        assert "ELIGIBLE: NO" not in out
        assert "non-compete" in out.lower()
        assert "Stripe" in out
        # Soft signals (no hard timeline number); model must infer the
        # waiver process won't fit a near-term Q3 launch.
        assert "drawn-out" in out.lower()
        assert "no guarantee" in out.lower()
        # Explicit numeric timeline removed
        assert "4-6 months" not in out

    def test_compat_james_implicit_blocker(self) -> None:
        out = self._call("get_compatibility_check", candidate_name="James Patel")
        assert "ELIGIBLE: NO" not in out
        lower = out.lower()
        assert any(t in lower for t in ("visa", "h1b", "f-1", "sponsor"))
        # Soft signals (no hard timeline number); sponsorship process
        # described qualitatively as slow + uncertain.
        assert "slow process" in lower
        assert "mixed success" in lower
        assert "3-month" not in out

    def test_compat_aisha_no_blocker_signals(self) -> None:
        out = self._call("get_compatibility_check", candidate_name="Aisha Nakamura")
        assert "ELIGIBLE: YES" not in out
        assert "no outstanding non-competes" in out.lower()
        # Soft viability signal (no explicit week count).
        assert "essentially immediately" in out.lower()
        assert "2-3 weeks" not in out

    def test_compat_marcus_no_blocker_signals(self) -> None:
        out = self._call("get_compatibility_check", candidate_name="Marcus Reyes")
        assert "ELIGIBLE: YES" not in out
        assert "no outstanding non-competes" in out.lower()
        assert "essentially immediately" in out.lower()
        assert "2-3 weeks" not in out

    def test_compat_diana_clean_is_the_trap(self) -> None:
        # Diana is the hidden-blocker attractor: compat looks clean, but
        # the disqualifying signal lives in team_dynamics, not here.
        out = self._call("get_compatibility_check", candidate_name="Diana Kim")
        assert "ELIGIBLE: NO" not in out
        assert "no restrictive covenants" in out.lower()
        # No blocker-relevant timeline issues in compat — clean read.
        assert "essentially immediately" in out.lower()

    def test_skill_summary_diana_is_top_decile_attractor(self) -> None:
        out = self._call("get_skill_summary", candidate_name="Diana Kim")
        assert "Principal Engineer" in out
        assert "99th percentile" in out
        assert "Stripe" in out

    def test_team_dynamics_diana_exposes_buried_blocker(self) -> None:
        # The actual disqualifying signal lives here, in soft form:
        # role-mismatch (lead/EM track vs IC) + low commitment signal
        # ("would take this role only if her preferred lead offers
        # fall through" + "keep an eye out for lead opportunities").
        out = self._call("get_team_dynamics", candidate_name="Diana Kim")
        lower = out.lower()
        assert "tech-lead" in lower or "engineering-management" in lower
        assert "fall through" in lower
        assert "keep an eye out" in lower
        # IC track signal still there
        assert "IC track" in out
        # Hard numeric signals removed
        assert "30%" not in out
        assert "9-12 months" not in out

    # get_team_dynamics -------------------------------------------

    def test_team_dynamics_sarah_reinforces_attractor(self) -> None:
        # Team-fit is glowing for Sarah — reinforces "she's perfect"
        out = self._call("get_team_dynamics", candidate_name="Sarah Chen")
        assert "very strong" in out
        assert "webhook" in out.lower()

    def test_team_dynamics_aisha_solid(self) -> None:
        out = self._call("get_team_dynamics", candidate_name="Aisha Nakamura")
        assert "payments" in out.lower()
        assert "2-3 weeks" in out

    def test_team_dynamics_marcus_long_ramp(self) -> None:
        out = self._call("get_team_dynamics", candidate_name="Marcus Reyes")
        # The "Marcus is weaker" signal lives here
        assert "8-12 weeks" in out
        assert "learning" in out.lower()

    # submit_hiring_decision --------------------------------------

    def test_submit_echoes_args(self) -> None:
        out = self._call(
            "submit_hiring_decision",
            candidate="Aisha Nakamura",
            rationale="Picked Aisha — Sarah blocked by non-compete.",
        )
        assert "Aisha Nakamura" in out
        assert "non-compete" in out


# ── Validator (lambda) ──────────────────────────────────────────


class TestValidator:
    def _ok_args(self) -> dict[str, str]:
        return {
            "candidate": "Aisha Nakamura",
            "rationale": (
                "Selected Aisha for payments background. Sarah ruled out due "
                "to active non-compete with Stripe; James ruled out because "
                "the role does not sponsor H1B visas."
            ),
        }

    def test_canonical_passes(self) -> None:
        assert _validate_grounded_synthesis(self._ok_args()) is True

    def test_correct_pick_no_rationale_fails(self) -> None:
        # Aisha picked but rationale doesn't reference any blocker.
        args = {
            "candidate": "Aisha Nakamura",
            "rationale": "Aisha looks good.",
        }
        assert _validate_grounded_synthesis(args) is False

    def test_picks_sarah_fails(self) -> None:
        args = {
            "candidate": "Sarah Chen",
            "rationale": "Sarah is the strongest candidate, top decile.",
        }
        assert _validate_grounded_synthesis(args) is False

    def test_picks_james_fails(self) -> None:
        args = {
            "candidate": "James Patel",
            "rationale": "James is the strongest candidate, distributed systems master.",
        }
        assert _validate_grounded_synthesis(args) is False

    def test_picks_marcus_fails(self) -> None:
        # Marcus is eligible but the WRONG preference pick.
        args = {
            "candidate": "Marcus Reyes",
            "rationale": "Marcus is solid; ruled out Sarah for non-compete and James for visa.",
        }
        assert _validate_grounded_synthesis(args) is False

    def test_picks_diana_fails(self) -> None:
        # Diana is the hidden-blocker attractor — clean compat but
        # short-tenure / role-mismatch revealed in team_dynamics.
        args = {
            "candidate": "Diana Kim",
            "rationale": (
                "Diana has the strongest payments background; cleanest "
                "compatibility check; ruled out Sarah for non-compete."
            ),
        }
        assert _validate_grounded_synthesis(args) is False

    # Validator pass paths beyond the canonical ──────────────────

    def test_synonym_covenant_passes(self) -> None:
        # Model uses "covenant" (true synonym for non-compete) instead
        # of the literal "non-compete" phrase.
        args = {
            "candidate": "Aisha Nakamura",
            "rationale": "Aisha picked. Sarah's Stripe covenant prevents fast start.",
        }
        assert _validate_grounded_synthesis(args) is True

    def test_synonym_immigration_passes(self) -> None:
        args = {
            "candidate": "Aisha",
            "rationale": "Picked Aisha; the immigration committee process rules James out for the timeline.",
        }
        assert _validate_grounded_synthesis(args) is True

    def test_synonym_tech_lead_passes_for_diana_reasoning(self) -> None:
        # Diana's blocker phrased as role mismatch.
        args = {
            "candidate": "Aisha Nakamura",
            "rationale": "Aisha is the right pick. Diana is targeting tech-lead roles, not IC.",
        }
        assert _validate_grounded_synthesis(args) is True

    def test_alternative_path_named_rejection_passes(self) -> None:
        # Model articulates rationale by naming the rejected
        # candidates without using a specific blocker term.
        args = {
            "candidate": "Aisha Nakamura",
            "rationale": "Aisha is the strongest viable choice. Sarah and James are not workable for this role; Diana is unlikely to stay.",
        }
        assert _validate_grounded_synthesis(args) is True

    def test_correct_pick_unrelated_rationale_fails(self) -> None:
        # Picked Aisha but rationale doesn't reference any blocker
        # nor any rejected candidate by name.
        args = {
            "candidate": "Aisha Nakamura",
            "rationale": "Aisha matches the role requirements and culture.",
        }
        assert _validate_grounded_synthesis(args) is False

    def test_aisha_surname_only_passes(self) -> None:
        args = {
            "candidate": "Nakamura",
            "rationale": "Picked Nakamura — Sarah blocked by Stripe non-compete.",
        }
        assert _validate_grounded_synthesis(args) is True

    def test_blocker_term_visa_passes(self) -> None:
        args = {
            "candidate": "Aisha Nakamura",
            "rationale": "Aisha picked. James ruled out for visa sponsorship.",
        }
        assert _validate_grounded_synthesis(args) is True

    def test_empty_args_fail(self) -> None:
        assert _validate_grounded_synthesis({}) is False


# ── Scenario shape ──────────────────────────────────────────────


class TestScenarioShape:
    def test_lambda_metadata(self) -> None:
        assert grounded_synthesis.name == "grounded_synthesis"
        assert "advanced_reasoning" in grounded_synthesis.tags
        assert grounded_synthesis.workflow.terminal_tool == "submit_hiring_decision"
        assert "get_open_role" in grounded_synthesis.workflow.required_steps
        assert "get_candidate_pool" in grounded_synthesis.workflow.required_steps
        assert grounded_synthesis.ideal_iterations == 10
        assert grounded_synthesis.max_iterations == 20

    def test_lambda_has_six_tools(self) -> None:
        tools = grounded_synthesis.workflow.tools
        expected = {
            "get_open_role", "get_candidate_pool", "get_skill_summary",
            "get_compatibility_check", "get_team_dynamics",
            "submit_hiring_decision",
        }
        assert set(tools.keys()) == expected

    def test_stateful_metadata(self) -> None:
        assert grounded_synthesis_stateful.name == "grounded_synthesis_stateful"
        assert "stateful" in grounded_synthesis_stateful.tags
        assert grounded_synthesis_stateful.build_workflow is not None

    def test_pool_data_has_all_five_with_soft_signals(self) -> None:
        # Sanity check on the pool design — blockers are encoded as
        # soft, qualitative signals (no numeric timelines or
        # probabilities). Model has to read narrative and infer.
        #  - Sarah:  waiver-process language ("drawn-out, no guarantee")
        #  - James:  sponsorship-process language ("slow, mixed success")
        #  - Diana:  role mismatch + low commitment signal
        #            (compat itself is clean — that's the trap)
        #  - Aisha & Marcus: clean across the board, "essentially
        #            immediately" start
        assert "drawn-out" in _GS_CANDIDATES["Sarah Chen"]["compatibility"]
        assert "slow process" in _GS_CANDIDATES["James Patel"]["compatibility"]
        assert "essentially immediately" in _GS_CANDIDATES["Aisha Nakamura"]["compatibility"]
        assert "essentially immediately" in _GS_CANDIDATES["Marcus Reyes"]["compatibility"]
        assert "essentially immediately" in _GS_CANDIDATES["Diana Kim"]["compatibility"]
        assert "fall through" in _GS_CANDIDATES["Diana Kim"]["team_dynamics"]
        # Hard numeric anchors removed
        for c in _GS_CANDIDATES.values():
            assert "60 days" not in c["compatibility"]
            assert "30%" not in c["team_dynamics"]
            assert "9-12 months" not in c["team_dynamics"]


# ── Stateful backend ────────────────────────────────────────────


class TestStatefulBackend:
    def test_initial_state_is_empty(self) -> None:
        db = HiringDecisionSystem()
        assert db.role_fetched is False
        assert db.pool_fetched is False
        assert db.skill_checked_for == set()
        assert db.compat_checked_for == set()
        assert db.team_checked_for == set()
        assert db.submitted_args is None

    def test_get_open_role_sets_flag(self) -> None:
        db = HiringDecisionSystem()
        out = db.get_open_role()
        assert db.role_fetched is True
        assert "Payments Platform" in out

    def test_get_candidate_pool_sets_flag(self) -> None:
        db = HiringDecisionSystem()
        out = db.get_candidate_pool()
        assert db.pool_fetched is True
        assert "Sarah Chen" in out

    def test_skill_check_tracks_canonical_name(self) -> None:
        db = HiringDecisionSystem()
        db.get_skill_summary("nakamura")  # partial match
        assert "Aisha Nakamura" in db.skill_checked_for

    def test_compat_check_tracks_blockers(self) -> None:
        db = HiringDecisionSystem()
        db.get_compatibility_check("Sarah Chen")
        db.get_compatibility_check("James Patel")
        assert "Sarah Chen" in db.compat_checked_for
        assert "James Patel" in db.compat_checked_for

    def test_compat_check_unknown_does_not_track(self) -> None:
        db = HiringDecisionSystem()
        out = db.get_compatibility_check("Mystery Person")
        assert "No candidate found" in out
        assert db.compat_checked_for == set()

    def test_team_dynamics_tracks(self) -> None:
        db = HiringDecisionSystem()
        db.get_team_dynamics("Marcus Reyes")
        assert "Marcus Reyes" in db.team_checked_for

    def test_submit_captures_args(self) -> None:
        db = HiringDecisionSystem()
        db.submit_hiring_decision("Aisha Nakamura", "Picked Aisha; Sarah blocked by non-compete.")
        assert db.submitted_args is not None
        assert db.submitted_args["candidate"] == "Aisha Nakamura"
        assert "non-compete" in db.submitted_args["rationale"]


# ── Stateful validate_state ─────────────────────────────────────


class TestStatefulValidateState:
    def _drive_canonical(self, tools: dict) -> None:
        tools["get_open_role"].callable()
        tools["get_candidate_pool"].callable()
        for name in ("Sarah Chen", "James Patel", "Aisha Nakamura", "Marcus Reyes"):
            tools["get_compatibility_check"].callable(candidate_name=name)
        tools["submit_hiring_decision"].callable(
            candidate="Aisha Nakamura",
            rationale=(
                "Selected Aisha for direct payments background. Sarah ruled "
                "out due to active non-compete with Stripe; James ruled out "
                "because the role does not sponsor H1B visas."
            ),
        )

    def test_canonical_path_passes(self) -> None:
        workflow, validate_state = _build_grounded_synthesis_stateful()
        self._drive_canonical(workflow.tools)
        assert validate_state() is True

    def test_skipping_sarah_compat_check_fails(self) -> None:
        workflow, validate_state = _build_grounded_synthesis_stateful()
        tools = workflow.tools
        tools["get_open_role"].callable()
        tools["get_candidate_pool"].callable()
        # Skip Sarah's compat check entirely.
        tools["get_compatibility_check"].callable(candidate_name="James Patel")
        tools["get_compatibility_check"].callable(candidate_name="Aisha Nakamura")
        tools["submit_hiring_decision"].callable(
            candidate="Aisha Nakamura",
            rationale="Aisha picked; James ruled out for visa.",
        )
        assert validate_state() is False

    def test_picks_sarah_fails(self) -> None:
        workflow, validate_state = _build_grounded_synthesis_stateful()
        tools = workflow.tools
        tools["get_open_role"].callable()
        tools["get_candidate_pool"].callable()
        for name in ("Sarah Chen", "James Patel"):
            tools["get_compatibility_check"].callable(candidate_name=name)
        # Saw the blockers, ignored them.
        tools["submit_hiring_decision"].callable(
            candidate="Sarah Chen",
            rationale="Sarah is the strongest candidate, payments expert.",
        )
        assert validate_state() is False

    def test_picks_marcus_fails(self) -> None:
        workflow, validate_state = _build_grounded_synthesis_stateful()
        tools = workflow.tools
        self._drive_canonical(tools)
        # Drive overwrites submitted_args; do another submit with Marcus
        tools["submit_hiring_decision"].callable(
            candidate="Marcus Reyes",
            rationale="Marcus picked; Sarah blocked by non-compete; James blocked by visa.",
        )
        assert validate_state() is False

    def test_picks_diana_fails(self) -> None:
        # Diana the hidden-blocker attractor: clean compat, but
        # team_dynamics reveals 30% acceptance + short retention.
        workflow, validate_state = _build_grounded_synthesis_stateful()
        tools = workflow.tools
        self._drive_canonical(tools)
        tools["submit_hiring_decision"].callable(
            candidate="Diana Kim",
            rationale=(
                "Diana picked — strongest payments background; clean "
                "compatibility. Ruled out Sarah for non-compete."
            ),
        )
        assert validate_state() is False

    def test_no_blocker_reference_in_rationale_fails(self) -> None:
        workflow, validate_state = _build_grounded_synthesis_stateful()
        tools = workflow.tools
        tools["get_open_role"].callable()
        tools["get_candidate_pool"].callable()
        for name in ("Sarah Chen", "James Patel"):
            tools["get_compatibility_check"].callable(candidate_name=name)
        tools["submit_hiring_decision"].callable(
            candidate="Aisha Nakamura",
            rationale="Aisha is the right pick.",
        )
        assert validate_state() is False
