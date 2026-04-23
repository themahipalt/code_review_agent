"""
Data models for the PRobe Environment.

An agent reviews Python source files, identifies bugs, security issues,
and design problems, then submits a structured review.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    """All actions the agent may take during a review episode."""

    ADD_COMMENT = "add_comment"
    GET_CONTEXT = "get_context"       # probe a line for deeper causal context
    REQUEST_CHANGES = "request_changes"
    APPROVE = "approve"
    SUBMIT_REVIEW = "submit_review"


class Severity(str, Enum):
    """Severity levels for review comments."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IssueCategory(str, Enum):
    """Issue category taxonomy used in review comments."""

    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    DESIGN = "design"


class RewardType(BaseModel):
    """
    Structured reward returned by ``step()``.

    Attributes:
        total:       Final clamped score in ``[-1.0, 1.0]``.
        components:  Named sub-scores before clamping (may sum outside ``[-1, 1]``).
        passed:      ``True`` when the action produced a clear positive signal.
        explanation: Human-readable breakdown for logging / debugging.
        step:        Environment step at which this reward was issued.
        terminal:    ``True`` only on the ``SUBMIT_REVIEW`` step.
    """

    model_config = ConfigDict(frozen=True)

    total: float = Field(..., ge=-1.0, le=1.0)
    components: dict[str, float] = Field(default_factory=dict)
    passed: bool = Field(default=False)
    explanation: str = Field(default="")
    step: int = Field(default=0, ge=0)
    terminal: bool = Field(default=False)


class ProbeAction(Action):
    """
    An action the agent submits during a review episode.

    Action types:
        ADD_COMMENT     — annotate a specific line with a review comment.
        GET_CONTEXT     — reveal ±5 lines of context around a line number.
        REQUEST_CHANGES — mark the PR as requiring changes before merge.
        APPROVE         — approve the PR (penalised if issues remain).
        SUBMIT_REVIEW   — finalise and submit the review (ends the episode).
    """

    action_type: ActionType = Field(..., description="Type of review action")
    line_number: int | None = Field(
        default=None,
        ge=1,
        description="1-based source line being commented on or probed",
    )
    comment: str | None = Field(default=None, description="Review comment text")
    severity: Severity | None = Field(default=None, description="Issue severity level")
    category: IssueCategory | None = Field(default=None, description="Issue category")


class ProbeObservation(Observation):
    """
    The observation returned to the agent after every ``reset()`` / ``step()``.

    The ``reward`` field mirrors ``RewardType.total`` for the most recent step
    as a convenience; the authoritative reward object is returned by ``step()``.
    """

    code_snippet: str = Field(default="", description="Python source code to review (mutated each episode)")
    task_description: str = Field(default="", description="Review instructions and goals")
    file_name: str = Field(default="", description="Name of the file being reviewed")
    task_id: int = Field(default=0, ge=0, description="Current task index (0–6)")
    task_difficulty: str = Field(default="ultra-easy", description="Task difficulty label")
    review_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of all actions taken so far this episode",
    )
    step_count: int = Field(default=0, ge=0, description="Steps taken in current episode")
    max_steps: int = Field(default=6, ge=1, description="Step budget for this task")
    issues_found_count: int = Field(default=0, ge=0, description="Distinct issues identified so far")
    total_issues: int = Field(default=0, ge=0, description="Total ground-truth issues in this task")
    context_hints: list[str] = Field(
        default_factory=list,
        description="Causal context unlocked by finding key issues — read these before continuing",
    )
    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Most recent step reward (mirrors RewardType.total)",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra episode metadata")


__all__ = [
    "ActionType",
    "IssueCategory",
    "ProbeAction",
    "ProbeObservation",
    "RewardType",
    "Severity",
]
