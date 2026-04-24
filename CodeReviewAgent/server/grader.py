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

Anti-exploit rules (v3):
  A comment MUST satisfy ALL of:
    1. keyword_hit  -- at least one issue keyword appears in the comment text
    2. line_hit     -- comment line_number is within +/-LINE_TOLERANCE of the issue
    3. substantive  -- comment is longer than MIN_COMMENT_LENGTH characters
  This prevents keyword-spam, wide-net line fishing, and trivial one-word matches.
"""

from __future__ import annotations

from typing import Any

try:
    from ..models import RewardType
except ImportError:
    from models import RewardType  # type: ignore[no-redef]

# -- Grading hyper-parameters ------------------------------------------------
LINE_TOLERANCE: int = 2         # lines either side of an issue's declared range
MIN_COMMENT_LENGTH: int = 15    # chars -- comments shorter than this earn no credit

ISSUE_REWARD_POOL: float = 0.40     # max cumulative credit from ADD_COMMENT (reduced to make room)
CLASSIFICATION_POOL: float = 0.20  # max cumulative credit for correct bug/backdoor classification
COVERAGE_POOL: float = 0.15        # terminal coverage bonus ceiling
DECISION_REWARD: float = 0.15       # +/- for correct/incorrect terminal action (incl. escalate)
DECISION_COVERAGE_GATE: float = 0.30  # min weighted coverage to EARN the decision bonus
EFFICIENCY_POOL: float = 0.10       # max terminal efficiency bonus
COVERAGE_THRESHOLD: float = 0.60    # min coverage to unlock efficiency bonus
FALSE_POSITIVE_PENALTY: float = -0.05  # per substantive unmatched comment
MISCLASSIFY_PENALTY: float = -0.05  # correct issue found but wrong classification label
FORMAT_BONUS: float = 0.02           # awarded once when output contains a valid non-empty JSON array


class CodeReviewGrader:
    """Scores agent actions against a task's ground-truth issue list."""

    def __init__(self, task: dict[str, Any]) -> None:
        self.task = task
        self.total_weight: float = sum(iss["weight"] for iss in task["issues"])

    # -- Per-comment scoring -------------------------------------------------

    def score_comment(
        self,
        line_number: int | None,
        comment: str | None,
        already_found: list[str],
        classification: str | None = None,
    ) -> tuple[float, list[str], dict[str, float]]:
        """
        Score an ADD_COMMENT action.

        Returns:
            (reward_delta, newly_found_issue_ids, component_breakdown)

        Match condition (ALL required -- no shortcut)::

            keyword_hit AND line_hit AND substantive

        If the issue has a ``classification`` field, the caller's classification
        is compared and either earns a bonus or a penalty.
        """
        if not comment:
            return 0.0, [], {}

        comment_lower = comment.lower()
        # Compute once -- used for both the credit path and the penalty path.
        substantive: bool = len(comment.strip()) > MIN_COMMENT_LENGTH

        newly_found: list[str] = []
        issue_credit: float = 0.0
        classification_credit: float = 0.0

        for issue in self.task["issues"]:
            if issue["id"] in already_found:
                continue

            keyword_hit = any(kw.lower() in comment_lower for kw in issue["keywords"])
            line_hit = self._line_in_range(line_number, issue["line_range"])

            if keyword_hit and line_hit and substantive:
                credit = (issue["weight"] / self.total_weight) * ISSUE_REWARD_POOL
                newly_found.append(issue["id"])
                issue_credit += credit

                # Classification bonus/penalty (only for issues that declare one)
                expected_cls = issue.get("classification")
                if expected_cls:
                    norm_cls = (classification or "").lower().replace("-", "_")
                    if norm_cls == expected_cls.lower().replace("-", "_"):
                        classification_credit += (
                            issue["weight"] / self.total_weight
                        ) * CLASSIFICATION_POOL
                    else:
                        classification_credit += MISCLASSIFY_PENALTY

        # Penalise substantive comments that matched nothing.
        false_positive_penalty: float = (
            FALSE_POSITIVE_PENALTY if (not newly_found and substantive) else 0.0
        )

        total = round(issue_credit + classification_credit + false_positive_penalty, 4)
        breakdown = {
            "issue_credit": round(issue_credit, 4),
            "classification_credit": round(classification_credit, 4),
            "false_positive_penalty": round(false_positive_penalty, 4),
        }
        return total, newly_found, breakdown

    # -- Terminal scoring ----------------------------------------------------

    def final_score(
        self,
        issues_found: list[str],
        review_decision: str | None,
        steps_used: int,
        max_steps: int,
    ) -> RewardType:
        """
        Compute the terminal reward on SUBMIT_REVIEW or ESCALATE_TO_SECURITY_REVIEW.

        Returns a fully-typed RewardType with a per-component breakdown.
        De-duplicates issues_found with stable ordering so results are
        deterministic regardless of insertion order.
        """
        unique_found: list[str] = sorted(set(issues_found))
        found_weight = sum(
            iss["weight"]
            for iss in self.task["issues"]
            if iss["id"] in unique_found
        )
        coverage = found_weight / self.total_weight if self.total_weight > 0 else 0.0

        correct_decision: str = self.task.get("correct_decision", "request_changes")
        # Decision bonus is only earned when the agent has found enough issues to
        # demonstrate it actually read the code.  This closes the exploit where an
        # agent that never reads the code always says "request_changes" and earns
        # +DECISION_REWARD for free.  An agent with zero coverage that picks the
        # correct decision still gets no penalty (0.0), but also no bonus.
        if review_decision == correct_decision and coverage >= DECISION_COVERAGE_GATE:
            decision_score = DECISION_REWARD
        elif review_decision is not None and review_decision != correct_decision:
            decision_score = -DECISION_REWARD
        else:
            decision_score = 0.0  # correct decision but insufficient coverage — no reward, no penalty

        step_efficiency = max(0.0, 1.0 - steps_used / max_steps)
        efficiency_bonus = (
            round(EFFICIENCY_POOL * step_efficiency, 4)
            if coverage >= COVERAGE_THRESHOLD
            else 0.0
        )
        coverage_bonus = round(coverage * COVERAGE_POOL, 4)

        raw_total = coverage_bonus + decision_score + efficiency_bonus
        clamped = round(max(-1.0, min(1.0, raw_total)), 4)

        components = {
            "coverage_bonus": coverage_bonus,
            "decision_score": round(decision_score, 4),
            "efficiency_bonus": efficiency_bonus,
        }
        total_issues = len(self.task["issues"])
        explanation = (
            f"Found {len(unique_found)}/{total_issues} issues "
            f"(weighted coverage {coverage:.0%}). "
            f"Decision {review_decision!r} was "
            f"{'correct' if review_decision == correct_decision else 'incorrect'} "
            f"(expected {correct_decision!r}). "
            f"Used {steps_used}/{max_steps} steps."
        )
        return RewardType(
            total=clamped,
            components=components,
            passed=review_decision == correct_decision and coverage >= COVERAGE_THRESHOLD,
            explanation=explanation,
            step=steps_used,
            terminal=True,
        )

    # -- Helper --------------------------------------------------------------

    @staticmethod
    def _line_in_range(
        line_number: int | None,
        line_range: tuple[int, int],
    ) -> bool:
        if line_number is None:
            return False
        start, end = line_range
        return (start - LINE_TOLERANCE) <= line_number <= (end + LINE_TOLERANCE)
