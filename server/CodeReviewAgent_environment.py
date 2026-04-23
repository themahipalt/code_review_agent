"""
CodeReviewAgent Environment — async-native implementation.

Episode lifecycle:
  1. reset()  → ObservationType              (starts a new episode)
  2. step(a)  → (Obs, RewardType, done, info) (execute one action)
  3. state()  → dict                          (full internal snapshot)

Tasks cycle automatically: 0 (ultra-easy) → 1 (easy) → … → 5 (hard flask) → 0 …

Thread / task safety: each Environment instance owns its own state.
For concurrent GRPO rollouts spin up one instance per worker.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        ActionType,
        CodereviewagentAction,
        CodereviewagentObservation,
        RewardType,
    )
    from .grader import CodeReviewGrader
    from .tasks import TASKS
except ImportError:
    from models import (  # type: ignore[no-redef]
        ActionType,
        CodereviewagentAction,
        CodereviewagentObservation,
        RewardType,
    )
    from server.grader import CodeReviewGrader  # type: ignore[no-redef]
    from server.tasks import TASKS  # type: ignore[no-redef]

# Sentinel reward returned on non-terminal steps that produce no signal
_ZERO_REWARD = RewardType(total=0.0, components={}, passed=False,
                           explanation="No signal this step.", step=0, terminal=False)


class CodereviewagentEnvironment(Environment):
    """
    OpenEnv-compliant code-review environment.

    Public interface is fully async.  The sync wrappers (reset / step / state)
    required by openenv's create_app are also provided; they delegate to the
    async versions via asyncio.run() so they are safe to call from sync
    contexts (e.g. tests without an event loop, openenv HTTP wrappers).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._episode_id: str = str(uuid4())
        self._step_count: int = 0
        self._reset_count: int = 0
        task = TASKS[0]
        self._grader: CodeReviewGrader = CodeReviewGrader(task)
        self._ep: dict[str, Any] = self._fresh_episode(task)

    @staticmethod
    def _fresh_episode(task: dict[str, Any]) -> dict[str, Any]:
        return {
            "task": task,
            "review_comments": [],
            "issues_found": [],
            "review_decision": None,
            "review_submitted": False,
            "cumulative_reward": 0.0,
        }

    # ── Async-native interface (primary) ──────────────────────────────────

    async def async_reset(self) -> CodereviewagentObservation:
        task_id = self._reset_count % len(TASKS)
        self._reset_count += 1
        self._episode_id = str(uuid4())
        self._step_count = 0
        task = TASKS[task_id]
        self._grader = CodeReviewGrader(task)
        self._ep = self._fresh_episode(task)
        return self._make_obs(reward=0.0, done=False)

    async def async_step(
        self, action: CodereviewagentAction
    ) -> tuple[CodereviewagentObservation, RewardType, bool, dict[str, Any]]:
        self._step_count += 1
        task = self._ep["task"]
        done = False
        reward_obj: RewardType

        if action.action_type == ActionType.ADD_COMMENT:
            reward_obj = self._handle_add_comment(action)

        elif action.action_type == ActionType.REQUEST_CHANGES:
            reward_obj = self._handle_request_changes(action)

        elif action.action_type == ActionType.APPROVE:
            reward_obj = self._handle_approve()

        elif action.action_type == ActionType.SUBMIT_REVIEW:
            reward_obj, done = self._handle_submit_review()

        else:
            reward_obj = RewardType(
                total=-0.05,
                components={"illegal_action": -0.05},
                passed=False,
                explanation=f"Unknown action type: {action.action_type}",
                step=self._step_count,
                terminal=False,
            )

        # Step-budget exhaustion
        if not done and self._step_count >= task["max_steps"]:
            # merge budget penalty into existing reward
            penalised = max(-1.0, reward_obj.total - 0.05)
            components = {**reward_obj.components, "step_budget_penalty": -0.05}
            reward_obj = RewardType(
                total=round(penalised, 4),
                components=components,
                passed=False,
                explanation=reward_obj.explanation + " [Step limit reached.]",
                step=self._step_count,
                terminal=True,
            )
            done = True

        self._ep["cumulative_reward"] = round(
            self._ep["cumulative_reward"] + reward_obj.total, 4
        )
        obs = self._make_obs(reward=reward_obj.total, done=done)
        info = {
            "episode_id": self._episode_id,
            "cumulative_reward": self._ep["cumulative_reward"],
            "issues_found": list(self._ep["issues_found"]),
            "review_decision": self._ep.get("review_decision"),
        }
        return obs, reward_obj, done, info

    async def async_state(self) -> dict[str, Any]:
        task = self._ep["task"]
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "task_id": task["id"],
            "task_difficulty": task["difficulty"],
            "task_name": task["name"],
            "issues_found": list(self._ep["issues_found"]),
            "total_issues": len(task["issues"]),
            "review_decision": self._ep.get("review_decision"),
            "review_submitted": self._ep.get("review_submitted", False),
            "cumulative_reward": self._ep.get("cumulative_reward", 0.0),
            "max_steps": task["max_steps"],
        }

    # ── Sync wrappers (openenv / create_app compatibility) ────────────────

    def reset(self) -> CodereviewagentObservation:  # type: ignore[override]
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.async_reset())
        # Called from inside a running loop (e.g. pytest-asyncio) — run directly
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(asyncio.run, self.async_reset())
            return fut.result()

    def step(self, action: CodereviewagentAction) -> CodereviewagentObservation:  # type: ignore[override]
        """
        Sync step for openenv compatibility.
        Returns only the Observation (reward is embedded in obs.reward).
        Use async_step() for the full (obs, reward, done, info) tuple.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            obs, _, _, _ = asyncio.run(self.async_step(action))
            return obs
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(asyncio.run, self.async_step(action))
            obs, _, _, _ = fut.result()
            return obs

    @property
    def state(self) -> State:  # type: ignore[override]
        return State(episode_id=self._episode_id, step_count=self._step_count)

    # ── Action handlers ───────────────────────────────────────────────────

    def _handle_add_comment(self, action: CodereviewagentAction) -> RewardType:
        entry = {
            "type": "comment",
            "line": action.line_number,
            "text": action.comment,
            "severity": action.severity.value if action.severity else None,
            "category": action.category.value if action.category else None,
        }
        self._ep["review_comments"].append(entry)

        score, new_finds, breakdown = self._grader.score_comment(
            line_number=action.line_number,
            comment=action.comment,
            already_found=self._ep["issues_found"],
        )
        self._ep["issues_found"].extend(new_finds)

        clamped = round(max(-1.0, min(1.0, score)), 4)
        if new_finds:
            explanation = f"Identified issue(s): {new_finds}"
        elif score < 0:
            explanation = "False-positive comment — matched no known issue."
        else:
            explanation = "Comment recorded; no new issue matched."

        return RewardType(
            total=clamped,
            components=breakdown,
            passed=bool(new_finds),
            explanation=explanation,
            step=self._step_count,
            terminal=False,
        )

    def _handle_request_changes(self, action: CodereviewagentAction) -> RewardType:
        self._ep["review_decision"] = "request_changes"
        self._ep["review_comments"].append(
            {"type": "request_changes", "text": action.comment}
        )
        if self._ep["issues_found"]:
            return RewardType(
                total=0.05,
                components={"decision_bonus": 0.05},
                passed=True,
                explanation="REQUEST_CHANGES after finding issues — correct.",
                step=self._step_count,
                terminal=False,
            )
        return RewardType(
            total=-0.05,
            components={"premature_decision_penalty": -0.05},
            passed=False,
            explanation="REQUEST_CHANGES with no issues found yet.",
            step=self._step_count,
            terminal=False,
        )

    def _handle_approve(self) -> RewardType:
        self._ep["review_decision"] = "approve"
        total_issues = len(self._ep["task"]["issues"])
        found = len(set(self._ep["issues_found"]))
        if total_issues > 0 and found < total_issues * 0.5:
            return RewardType(
                total=-0.15,
                components={"bad_approval_penalty": -0.15},
                passed=False,
                explanation=f"APPROVE with only {found}/{total_issues} issues found.",
                step=self._step_count,
                terminal=False,
            )
        return RewardType(
            total=0.02,
            components={"approval_credit": 0.02},
            passed=True,
            explanation="APPROVE recorded.",
            step=self._step_count,
            terminal=False,
        )

    def _handle_submit_review(self) -> tuple[RewardType, bool]:
        if self._ep.get("review_submitted"):
            return (
                RewardType(
                    total=-0.05,
                    components={"duplicate_submit_penalty": -0.05},
                    passed=False,
                    explanation="Review already submitted.",
                    step=self._step_count,
                    terminal=False,
                ),
                False,
            )
        self._ep["review_submitted"] = True
        task = self._ep["task"]
        reward_obj = self._grader.final_score(
            issues_found=list(set(self._ep["issues_found"])),
            review_decision=self._ep.get("review_decision"),
            step_count=self._step_count,
            max_steps=task["max_steps"],
            current_step=self._step_count,
        )
        return reward_obj, True

    # ── Observation builder ───────────────────────────────────────────────

    def _make_obs(self, reward: float, done: bool) -> CodereviewagentObservation:
        task = self._ep["task"]
        return CodereviewagentObservation(
            code_snippet=task["code"],
            task_description=task["description"],
            file_name=task["file_name"],
            task_id=task["id"],
            task_difficulty=task["difficulty"],
            review_history=list(self._ep.get("review_comments", [])),
            step_count=self._step_count,
            max_steps=task["max_steps"],
            issues_found_count=len(set(self._ep.get("issues_found", []))),
            total_issues=len(task["issues"]),
            done=done,
            reward=round(max(-1.0, min(1.0, reward)), 4),
            metadata={
                "cumulative_reward": self._ep.get("cumulative_reward", 0.0),
                "review_decision": self._ep.get("review_decision"),
                "episode_id": self._episode_id,
            },
        )
