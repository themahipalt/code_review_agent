"""
Data models for the CodeReviewAgent Environment.

An agent reviews Python source files, identifies bugs, security issues,
and design problems, then submits a structured review.
"""

from enum import Enum
from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    ADD_COMMENT = "add_comment"
    REQUEST_CHANGES = "request_changes"
    APPROVE = "approve"
    SUBMIT_REVIEW = "submit_review"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IssueCategory(str, Enum):
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    DESIGN = "design"


class RewardType(BaseModel):
    """
    Structured reward returned by step().

    total       : final clamped score in [-1.0, 1.0]
    components  : named sub-scores before clamping (may sum outside [-1, 1])
    passed      : True when the action was a clear positive signal
    explanation : human-readable breakdown for logging / debugging
    step        : environment step this reward was issued at
    terminal    : True only on the SUBMIT_REVIEW step
    """

    model_config = ConfigDict(frozen=True)

    total: float = Field(..., ge=-1.0, le=1.0)
    components: dict[str, float] = Field(default_factory=dict)
    passed: bool = Field(False)
    explanation: str = Field("")
    step: int = Field(0)
    terminal: bool = Field(False)


class CodereviewagentAction(Action):
    """
    - ADD_COMMENT    : annotate a specific line with a review comment
    - REQUEST_CHANGES: mark the PR as needing changes
    - APPROVE        : approve the PR (only when no significant issues remain)
    - SUBMIT_REVIEW  : finalize and submit the review (ends the episode)
    """

    action_type: ActionType = Field(..., description="Type of review action")
    line_number: int | None = Field(None, description="Source line being commented on")
    comment: str | None = Field(None, description="Review comment text")
    severity: Severity | None = Field(None, description="Issue severity level")
    category: IssueCategory | None = Field(None, description="Issue category")


class CodereviewagentObservation(Observation):
    """
    Contains the code to review, task instructions, and the running
    review history so the agent can track what it has already flagged.
    The `reward` field mirrors the most recent step reward for convenience;
    the authoritative reward is the RewardType returned by step().
    """

    code_snippet: str = Field(default="", description="Python source code to review")
    task_description: str = Field(default="", description="Review instructions and goals")
    file_name: str = Field(default="", description="Name of the file being reviewed")
    task_id: int = Field(default=0, description="Current task index")
    task_difficulty: str = Field(default="ultra-easy", description="Task difficulty label")
    review_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of actions taken so far this episode",
    )
    step_count: int = Field(default=0, description="Steps taken in current episode")
    max_steps: int = Field(default=6, description="Step budget for this task")
    issues_found_count: int = Field(default=0, description="Number of issues identified so far")
    total_issues: int = Field(default=0, description="Total issues in this task")
    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: float = Field(default=0.0, description="Most recent step reward (mirror of RewardType.total)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra episode metadata")
