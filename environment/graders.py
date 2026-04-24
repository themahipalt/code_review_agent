"""
Deterministic reward grader for PRobe tasks.

Scoring design
--------------
During the episode (ADD_COMMENT actions):
  + weight/total_weight * ISSUE_REWARD_POOL   per newly found issue
  - FALSE_POSITIVE_PENALTY                    per substantive unmatched comment

Terminal (SUBMIT_REVIEW):
  + coverage * COVERAGE_POOL      weighted coverage bonus  (max COVERAGE_POOL)
  +/- DECISION_REWARD             correct / incorrect final decision
  + efficiency * EFFICIENCY_POOL  step-efficiency bonus when coverage >= COVERAGE_THRESHOLD

Maximum achievable total: ~1.0   Minimum: -1.0

Anti-exploit rules:
  A comment MUST satisfy ALL of:
    1. keyword_hit  -- at least one issue keyword appears in the comment text
    2. line_hit     -- comment line_number is within +/-LINE_TOLERANCE of the issue
    3. substantive  -- comment is longer than MIN_COMMENT_LENGTH characters
  This prevents keyword-spam, wide-net line fishing, and trivial one-word matches.
"""

from __future__ import annotations

from typing import Any

try:
    from ..agent.models import RewardType
except ImportError:
    from agent.models import RewardType  # type: ignore[no-redef]

# ── Grading hyper-parameters ─────────────────────────────────────────────────

LINE_TOLERANCE: int = 2           # ± lines around a declared issue range
MIN_COMMENT_LENGTH: int = 15      # minimum chars for a comment to earn credit

ISSUE_REWARD_POOL: float = 0.40       # max cumulative credit from ADD_COMMENT
CLASSIFICATION_POOL: float = 0.20    # max cumulative credit for correct classification
COVERAGE_POOL: float = 0.15          # terminal coverage bonus ceiling
DECISION_REWARD: float = 0.15        # ± for correct/incorrect terminal action
DECISION_COVERAGE_GATE: float = 0.30 # min weighted coverage to earn the decision bonus
EFFICIENCY_POOL: float = 0.10        # max terminal efficiency bonus
COVERAGE_THRESHOLD: float = 0.60     # min coverage to unlock efficiency bonus
FALSE_POSITIVE_PENALTY: float = -0.05  # per substantive comment that matched no issue
MISSCLASSIFY_PENALTY: float = -0.05    # correct issue found but wrong classification label
FORMAT_BONUS: float = 0.02             # awarded once for a valid non-empty JSON array

# Type alias for the per-component score breakdown returned by score_comment.
ScoreBreakdown = dict[str, float]


class CodeReviewGrader:
    """
    Scores agent actions against a task's ground-truth issue list.

    All scoring is deterministic and requires no external calls — the grader
    is safe to instantiate from multiple threads or GRPO rollout workers.
    """

    def __init__(self, task: dict[str, Any]) -> None:
        self._task = task
        # Precompute once at construction — this value is read on every
        # ADD_COMMENT call, and a GRPO training run may invoke score_comment
        # thousands of times across rollout workers.
        self._total_weight: float = sum(iss["weight"] for iss in task["issues"])

    @property
    def task(self) -> dict[str, Any]:
        """Read-only access to the underlying task definition."""
        return self._task

    @property
    def total_weight(self) -> float:
        """Sum of all issue weights. Kept for backward compatibility."""
        return self._total_weight

    # ── Per-comment scoring ───────────────────────────────────────────────────

    def score_comment(
        self,
        line_number: int | None,
        comment: str | None,
        already_found: list[str],
        classification: str | None = None,
    ) -> tuple[float, list[str], ScoreBreakdown]:
        """
        Score an ADD_COMMENT action.

        A comment earns issue credit only when ALL three conditions hold:
          - keyword_hit: at least one issue keyword appears in the comment text
          - line_hit:    line_number is within ±LINE_TOLERANCE of the issue range
          - substantive: comment body is longer than MIN_COMMENT_LENGTH characters

        Returns:
            A 3-tuple of (reward_delta, newly_found_issue_ids, component_breakdown).
        """
        if not comment:
            return 0.0, [], {}

        comment_lower = comment.lower()
        is_substantive: bool = len(comment.strip()) > MIN_COMMENT_LENGTH

        newly_found_ids: list[str] = []
        issue_credit: float = 0.0
        classification_credit: float = 0.0

        for issue in self._task["issues"]:
            if issue["id"] in already_found:
                continue

            keyword_hit = any(kw.lower() in comment_lower for kw in issue["keywords"])
            line_hit = self._line_in_range(line_number, issue["line_range"])

            if keyword_hit and line_hit and is_substantive:
                issue_credit += (issue["weight"] / self._total_weight) * ISSUE_REWARD_POOL
                newly_found_ids.append(issue["id"])
                classification_credit += self._score_classification(
                    issue=issue,
                    given_classification=classification,
                )

        # Only substantive comments that match nothing earn the FP penalty.
        # Very short comments (e.g. one-word probes) are exempt so the agent
        # is not punished for exploratory low-effort queries.
        false_positive_penalty: float = (
            FALSE_POSITIVE_PENALTY if (not newly_found_ids and is_substantive) else 0.0
        )

        total = round(issue_credit + classification_credit + false_positive_penalty, 4)
        breakdown: ScoreBreakdown = {
            "issue_credit": round(issue_credit, 4),
            "classification_credit": round(classification_credit, 4),
            "false_positive_penalty": round(false_positive_penalty, 4),
        }
        return total, newly_found_ids, breakdown

    def _score_classification(
        self,
        issue: dict[str, Any],
        given_classification: str | None,
    ) -> float:
        """Return the classification bonus or penalty for a single matched issue."""
        expected = issue.get("classification")
        if not expected:
            return 0.0
        # Normalize both separator styles — training data may produce either
        # "accidental-bug" (hyphenated) or "accidental_bug" (underscored).
        normalised = (given_classification or "").lower().replace("-", "_")
        if normalised == expected.lower().replace("-", "_"):
            return (issue["weight"] / self._total_weight) * CLASSIFICATION_POOL
        return MISCLASSIFY_PENALTY

    # ── Terminal scoring ──────────────────────────────────────────────────────

    def compute_final_score(
        self,
        issues_found: list[str],
        review_decision: str | None,
        steps_used: int,
        max_steps: int,
    ) -> RewardType:
        """
        Compute the terminal reward on SUBMIT_REVIEW or ESCALATE_TO_SECURITY_REVIEW.

        Deduplicates issues_found with stable ordering so results are
        deterministic regardless of insertion order.
        """
        unique_found_ids: list[str] = sorted(set(issues_found))
        weighted_coverage = self._compute_weighted_coverage(unique_found_ids)
        correct_decision: str = self._task.get("correct_decision", "request_changes")

        coverage_bonus = round(weighted_coverage * COVERAGE_POOL, 4)
        decision_score = self._compute_decision_score(review_decision, correct_decision, weighted_coverage)
        efficiency_bonus = self._compute_efficiency_bonus(weighted_coverage, steps_used, max_steps)

        raw_total = coverage_bonus + decision_score + efficiency_bonus
        clamped_total = round(max(-1.0, min(1.0, raw_total)), 4)

        total_issue_count = len(self._task["issues"])
        is_correct = review_decision == correct_decision
        explanation = (
            f"Found {len(unique_found_ids)}/{total_issue_count} issues "
            f"(weighted coverage {weighted_coverage:.0%}). "
            f"Decision {review_decision!r} was "
            f"{'correct' if is_correct else 'incorrect'} "
            f"(expected {correct_decision!r}). "
            f"Used {steps_used}/{max_steps} steps."
        )
        return RewardType(
            total=clamped_total,
            components={
                "coverage_bonus": coverage_bonus,
                "decision_score": round(decision_score, 4),
                "efficiency_bonus": efficiency_bonus,
            },
            # passed=True only when both conditions hold: the agent chose the
            # correct terminal action AND covered enough issues to prove it
            # actually read the code (prevents lucky-guess escalations).
            passed=is_correct and weighted_coverage >= COVERAGE_THRESHOLD,
            explanation=explanation,
            step=steps_used,
            terminal=True,
        )

    # Keep the old name as a thin alias so existing call-sites aren't broken.
    def final_score(
        self,
        issues_found: list[str],
        review_decision: str | None,
        steps_used: int,
        max_steps: int,
    ) -> RewardType:
        """Alias for compute_final_score — retained for backward compatibility."""
        return self.compute_final_score(issues_found, review_decision, steps_used, max_steps)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_weighted_coverage(self, unique_found_ids: list[str]) -> float:
        """Return the fraction of total issue weight covered by the found issues."""
        if self._total_weight <= 0:
            return 0.0
        found_weight = sum(
            iss["weight"]
            for iss in self._task["issues"]
            if iss["id"] in unique_found_ids
        )
        return found_weight / self._total_weight

    @staticmethod
    def _compute_decision_score(
        review_decision: str | None,
        correct_decision: str,
        weighted_coverage: float,
    ) -> float:
        """
        Return the decision bonus or penalty.

        The bonus is gated on coverage >= DECISION_COVERAGE_GATE to prevent
        an agent from earning +DECISION_REWARD by always guessing the correct
        terminal action without reading any code.
        """
        if review_decision == correct_decision and weighted_coverage >= DECISION_COVERAGE_GATE:
            return DECISION_REWARD
        if review_decision is not None and review_decision != correct_decision:
            return -DECISION_REWARD
        # Correct decision but insufficient coverage — neutral, no bonus or penalty.
        return 0.0

    @staticmethod
    def _compute_efficiency_bonus(weighted_coverage: float, steps_used: int, max_steps: int) -> float:
        """Return the step-efficiency bonus, unlocked only when coverage >= COVERAGE_THRESHOLD."""
        if weighted_coverage < COVERAGE_THRESHOLD:
            return 0.0
        step_efficiency = max(0.0, 1.0 - steps_used / max_steps)
        return round(EFFICIENCY_POOL * step_efficiency, 4)

    @staticmethod
    def _line_in_range(
        line_number: int | None,
        line_range: tuple[int, int],
    ) -> bool:
        """Return True when line_number falls within the issue range ± LINE_TOLERANCE."""
        if line_number is None:
            return False
        start, end = line_range
        return (start - LINE_TOLERANCE) <= line_number <= (end + LINE_TOLERANCE)
