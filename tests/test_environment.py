"""
Pytest suite for PRobe.

Covers:
  - One smoke test per difficulty tier (ultra-easy, easy, medium, hard)
  - Keyword-spam exploit: confirms it scores ≤ 0
  - Grader unit tests: correct match, line miss, false-positive penalty
  - SUBMIT_REVIEW terminal reward shape
"""

from __future__ import annotations

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "probe"))

from probe.server.probe_environment import ProbeEnvironment
from probe.server.grader import CodeReviewGrader
from probe.models import (
    ActionType,
    ProbeAction,
    IssueCategory,
    RewardType,
    Severity,
)
from probe.server.tasks import TASKS
from probe.server.grader import FALSE_POSITIVE_PENALTY


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_comment(line: int, comment: str, severity: str = "error", category: str = "bug") -> ProbeAction:
    return ProbeAction(
        action_type=ActionType.ADD_COMMENT,
        line_number=line,
        comment=comment,
        severity=Severity(severity),
        category=IssueCategory(category),
    )


def _submit() -> ProbeAction:
    return ProbeAction(action_type=ActionType.SUBMIT_REVIEW)


def _request_changes() -> ProbeAction:
    return ProbeAction(
        action_type=ActionType.REQUEST_CHANGES,
        comment="Issues found — requesting changes.",
    )


async def _reset_to_task(env: ProbeEnvironment, task_id: int):
    """Cycle resets until the env lands on task_id."""
    for _ in range(task_id + 1):
        obs = await env.async_reset()
    return obs


# ── Grader unit tests ─────────────────────────────────────────────────────────

class TestGraderUnitTests:
    """Direct tests of CodeReviewGrader without the environment."""

    @pytest.fixture()
    def ultra_easy_grader(self):
        return CodeReviewGrader(TASKS[0])

    @pytest.fixture()
    def easy_grader(self):
        return CodeReviewGrader(TASKS[1])

    def test_keyword_and_line_both_required_match(self, ultra_easy_grader):
        """Both keyword hit AND line proximity → positive credit."""
        reward, found, breakdown = ultra_easy_grader.score_comment(
            line_number=4,
            comment="Off-by-one error: range(len+1) causes IndexError",
            already_found=[],
        )
        assert reward > 0
        assert "bootstrap_off_by_one" in found
        assert breakdown["issue_credit"] > 0

    def test_keyword_only_no_line_hit_scores_zero(self, ultra_easy_grader):
        """Keyword present but line far away → no credit (closes reward-hacking shortcut)."""
        reward, found, breakdown = ultra_easy_grader.score_comment(
            line_number=999,
            comment="Off-by-one error: range(len+1) causes IndexError out of bounds",
            already_found=[],
        )
        assert found == []
        assert breakdown["issue_credit"] == 0.0

    def test_line_only_no_keyword_scores_zero(self, ultra_easy_grader):
        """Correct line but no matching keyword → no credit."""
        reward, found, breakdown = ultra_easy_grader.score_comment(
            line_number=4,
            comment="This function looks fine to me.",
            already_found=[],
        )
        assert found == []
        assert breakdown["issue_credit"] == 0.0

    def test_false_positive_penalty_applied(self, ultra_easy_grader):
        """Substantive comment that matches nothing → -0.02 penalty."""
        reward, found, breakdown = ultra_easy_grader.score_comment(
            line_number=50,
            comment="This is a well-written and clean function overall.",
            already_found=[],
        )
        assert found == []
        assert breakdown["false_positive_penalty"] == pytest.approx(FALSE_POSITIVE_PENALTY)
        assert reward == pytest.approx(FALSE_POSITIVE_PENALTY)

    def test_already_found_issue_skipped(self, ultra_easy_grader):
        """Duplicate detection of an already found issue scores zero."""
        reward, found, breakdown = ultra_easy_grader.score_comment(
            line_number=4,
            comment="Off-by-one error: range(len+1) causes IndexError",
            already_found=["bootstrap_off_by_one"],
        )
        assert found == []
        assert breakdown["issue_credit"] == 0.0

    def test_security_issue_keyword_and_line(self, ultra_easy_grader):
        """Hardcoded credential issue — keyword + correct line → credit."""
        reward, found, breakdown = ultra_easy_grader.score_comment(
            line_number=11,
            comment="Hardcoded password should be moved to an environment variable.",
            already_found=[],
        )
        assert "bootstrap_hardcoded_cred" in found
        assert reward > 0


# ── Keyword-spam exploit test ─────────────────────────────────────────────────

class TestKeywordSpamExploit:
    """
    Verifies that dumping all known keywords on a single wrong-line comment
    cannot earn positive reward (Q1 anti-exploit rule).
    """

    @pytest.mark.asyncio
    async def test_keyword_spam_wrong_line_scores_le_zero(self):
        """
        A comment crammed with all keywords but placed on line 999
        (far from any real issue) must score ≤ 0.
        """
        env = ProbeEnvironment()
        await env.async_reset()  # Task 0 ultra-easy

        spam_comment = (
            "off-by-one indexerror range len+1 bug hardcoded credential "
            "password secret env os.environ security sql injection eval "
            "pickle md5 verify=False shell=True path traversal"
        )
        action = _add_comment(line=999, comment=spam_comment)
        obs, reward_obj, done, info = await env.async_step(action)

        assert reward_obj.total <= 0.0, (
            f"Keyword-spam exploit yielded reward={reward_obj.total} > 0; "
            "the or cat_hit shortcut must be removed."
        )
        assert info["issues_found"] == [], (
            "Keyword spam at wrong line should not register any found issues."
        )

    @pytest.mark.asyncio
    async def test_keyword_spam_multiple_lines_no_gain(self):
        """
        Firing keyword-spam at several wrong lines should give cumulative ≤ 0.
        """
        env = ProbeEnvironment()
        await env.async_reset()

        spam = "off-by-one range len+1 indexerror hardcoded password credential security"
        cumulative = 0.0
        for line in [100, 200, 300, 400, 500]:
            _, reward_obj, _, _ = await env.async_step(_add_comment(line, spam))
            cumulative += reward_obj.total

        assert cumulative <= 0.0


# ── Per-difficulty tier tests ─────────────────────────────────────────────────

class TestUltraEasyTier:
    """Task 0 — bootstrap tier (difficulty: ultra-easy)."""

    @pytest.mark.asyncio
    async def test_reset_returns_ultra_easy_task(self):
        env = ProbeEnvironment()
        obs = await env.async_reset()
        assert obs.task_difficulty == "ultra-easy"
        assert obs.total_issues == 2

    @pytest.mark.asyncio
    async def test_correct_comment_earns_positive_reward(self):
        env = ProbeEnvironment()
        await env.async_reset()

        obs, reward_obj, done, info = await env.async_step(
            _add_comment(4, "Off-by-one: range(len+1) causes IndexError on last iteration")
        )
        assert reward_obj.total > 0
        assert reward_obj.passed is True
        assert "bootstrap_off_by_one" in info["issues_found"]

    @pytest.mark.asyncio
    async def test_full_episode_positive_terminal_reward(self):
        """A perfect episode on the ultra-easy task should yield positive terminal reward."""
        env = ProbeEnvironment()
        await env.async_reset()

        await env.async_step(_add_comment(4, "Off-by-one: range(len+1) causes IndexError"))
        await env.async_step(_add_comment(11, "Hardcoded credential: move password to environment variable", "critical", "security"))
        await env.async_step(_request_changes())
        _, reward_obj, done, info = await env.async_step(_submit())

        assert done is True
        assert reward_obj.terminal is True
        assert reward_obj.total > 0
        assert reward_obj.passed is True


class TestEasyTier:
    """Task 1 — easy tier (3 logic bugs)."""

    @pytest.mark.asyncio
    async def test_easy_task_loaded(self):
        env = ProbeEnvironment()
        obs = await _reset_to_task(env, 1)
        assert obs.task_difficulty == "easy"
        assert obs.total_issues == 3

    @pytest.mark.asyncio
    async def test_off_by_one_found_in_easy_task(self):
        env = ProbeEnvironment()
        await _reset_to_task(env, 1)

        obs, reward_obj, done, info = await env.async_step(
            _add_comment(4, "Off-by-one: range(len+1) will cause IndexError on the last iteration")
        )
        assert reward_obj.total > 0
        assert "off_by_one" in info["issues_found"]

    @pytest.mark.asyncio
    async def test_assignment_bug_found_in_easy_task(self):
        env = ProbeEnvironment()
        await _reset_to_task(env, 1)

        obs, reward_obj, done, info = await env.async_step(
            _add_comment(17, "Assignment bug: max_val == item uses == (comparison) instead of = (assignment); max is never updated")
        )
        assert reward_obj.total > 0
        assert "assignment_not_update" in info["issues_found"]


class TestMediumTier:
    """Task 2 — medium tier (security vulnerabilities in auth module)."""

    @pytest.mark.asyncio
    async def test_medium_task_loaded(self):
        env = ProbeEnvironment()
        obs = await _reset_to_task(env, 2)
        assert obs.task_difficulty == "medium"
        assert obs.total_issues >= 4

    @pytest.mark.asyncio
    async def test_sql_injection_found(self):
        env = ProbeEnvironment()
        await _reset_to_task(env, 2)

        obs, reward_obj, done, info = await env.async_step(
            _add_comment(
                13,
                "SQL injection: f-string interpolation in query allows injection — use parameterized queries",
                "critical",
                "security",
            )
        )
        assert reward_obj.total > 0

    @pytest.mark.asyncio
    async def test_submit_without_comments_penalised(self):
        """Submitting with zero coverage should yield low / negative terminal reward."""
        env = ProbeEnvironment()
        await _reset_to_task(env, 2)

        _, reward_obj, done, _ = await env.async_step(_submit())
        assert done is True
        assert reward_obj.terminal is True
        # No issues found, wrong decision → should score ≤ 0
        assert reward_obj.total <= 0.0


class TestHardTier:
    """Task 3 — hard tier (data pipeline)."""

    @pytest.mark.asyncio
    async def test_hard_task_loaded(self):
        env = ProbeEnvironment()
        obs = await _reset_to_task(env, 3)
        assert obs.task_difficulty == "hard"
        assert obs.total_issues >= 5

    @pytest.mark.asyncio
    async def test_step_budget_exhaustion_terminates_episode(self):
        """
        Taking max_steps no-op comments should exhaust the budget and terminate.
        """
        env = ProbeEnvironment()
        obs = await _reset_to_task(env, 3)
        max_steps = obs.max_steps

        done = False
        for _ in range(max_steps):
            if done:
                break
            _, reward_obj, done, _ = await env.async_step(
                _add_comment(9999, "x" * 20)  # meaningless, far line
            )

        assert done is True


# ── RewardType shape tests ────────────────────────────────────────────────────

class TestRewardTypeShape:
    """Verify RewardType always stays in [-1, 1] and has required fields."""

    @pytest.mark.asyncio
    async def test_reward_range_never_exceeds_bounds(self):
        env = ProbeEnvironment()
        await env.async_reset()

        actions = [
            _add_comment(4, "Off-by-one: range(len+1) causes IndexError"),
            _add_comment(11, "Hardcoded credential: use environment variable", "critical", "security"),
            _request_changes(),
            _submit(),
        ]
        for action in actions:
            _, reward_obj, done, _ = await env.async_step(action)
            assert -1.0 <= reward_obj.total <= 1.0, (
                f"reward {reward_obj.total} outside [-1, 1]"
            )
            assert isinstance(reward_obj.components, dict)
            assert isinstance(reward_obj.explanation, str)
            assert isinstance(reward_obj.passed, bool)
            if done:
                break

    @pytest.mark.asyncio
    async def test_terminal_reward_has_terminal_flag(self):
        env = ProbeEnvironment()
        await env.async_reset()
        _, reward_obj, done, _ = await env.async_step(_submit())
        assert done is True
        assert reward_obj.terminal is True

    @pytest.mark.asyncio
    async def test_non_terminal_steps_have_terminal_false(self):
        env = ProbeEnvironment()
        await env.async_reset()
        _, reward_obj, done, _ = await env.async_step(
            _add_comment(4, "Off-by-one: range(len+1) causes IndexError")
        )
        assert done is False
        assert reward_obj.terminal is False


# ── State / async_state tests ─────────────────────────────────────────────────

class TestAsyncState:
    @pytest.mark.asyncio
    async def test_state_reflects_step_count(self):
        env = ProbeEnvironment()
        await env.async_reset()
        await env.async_step(_add_comment(4, "Off-by-one issue range len+1"))
        state = await env.async_state()
        assert state["step_count"] == 1

    @pytest.mark.asyncio
    async def test_state_reflects_issues_found(self):
        env = ProbeEnvironment()
        await env.async_reset()
        await env.async_step(_add_comment(4, "Off-by-one: range(len+1) causes IndexError"))
        state = await env.async_state()
        assert "bootstrap_off_by_one" in state["issues_found"]
