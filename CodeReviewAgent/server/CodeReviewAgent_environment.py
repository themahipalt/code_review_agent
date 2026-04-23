"""
CodeReviewAgent Environment — async-native implementation.

Episode lifecycle:
  1. reset()  → ObservationType              (starts a new episode)
  2. step(a)  → (Obs, RewardType, done, info) (execute one action)
  3. state()  → dict                          (full internal snapshot)

Tasks cycle automatically: 0 (ultra-easy) → 1 (easy) → … → 6 (causal chain) → 0 …

Dynamic world features (v3)
───────────────────────────
• Code mutation   — each episode applies surface-level variable renames,
                    a line shift, and a constant nudge so the agent must
                    read the code rather than memorise tokens.
• GET_CONTEXT     — the agent can spend a step probing a specific line to
                    receive the surrounding ±5 lines of context.
• Causal unlocks  — finding certain issues appends a new context hint to
                    the observation, modelling real-world situations where
                    one discovery leads to deeper investigation.

Thread / task safety: each Environment instance owns its own state.
For concurrent GRPO rollouts spin up one instance per worker.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ActionType, ProbeAction, ProbeObservation, RewardType
    from ._import_compat import (
        CodeReviewGrader,
        EpisodeMemory,
        LINE_TOLERANCE,
        TASKS,
        mutate_task,
        run_scanner,
    )
except ImportError:
    from models import ActionType, ProbeAction, ProbeObservation, RewardType  # type: ignore[no-redef]
    from server._import_compat import (  # type: ignore[no-redef]
        CodeReviewGrader,
        EpisodeMemory,
        LINE_TOLERANCE,
        TASKS,
        mutate_task,
        run_scanner,
    )

log = logging.getLogger(__name__)


@dataclasses.dataclass
class EpisodeState:
    """All mutable state for a single review episode.

    Using a dataclass eliminates stringly-typed dict key access and makes
    the shape of an episode explicit and statically checkable.
    """

    task: dict[str, Any]
    review_comments: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    issues_found: list[str] = dataclasses.field(default_factory=list)
    review_decision: str | None = None
    review_submitted: bool = False
    cumulative_reward: float = 0.0
    # Causal world-modelling state
    context_hints: list[str] = dataclasses.field(default_factory=list)
    hints_unlocked: set[str] = dataclasses.field(default_factory=set)
    # Scanner state
    scanner_used: bool = False


class ProbeEnvironment(Environment):
    """
    PRobe — Pull Request Investigation Environment.

    Public interface is fully async.  The sync wrappers (reset / step / state)
    required by openenv's create_app are also provided; they delegate to the
    async versions via asyncio.run() so they are safe to call from sync
    contexts (e.g. tests without an event loop, openenv HTTP wrappers).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(self, memory_dir: str | None = None) -> None:
        self._episode_id: str = str(uuid4())
        self._step_count: int = 0
        self._reset_count: int = 0
        self._memory: EpisodeMemory = EpisodeMemory(
            memory_dir=memory_dir,
            instance_id=self._episode_id[:8],
        )
        initial_task = TASKS[0]
        self._grader: CodeReviewGrader = CodeReviewGrader(initial_task)
        self._episode: EpisodeState = EpisodeState(task=initial_task)

    # ── Async-native interface (primary) ──────────────────────────────────

    async def async_reset(self) -> ProbeObservation:
        task_id = self._reset_count % len(TASKS)
        episode_seed = self._reset_count
        self._reset_count += 1
        self._episode_id = str(uuid4())
        self._step_count = 0

        # Apply surface mutation so the agent cannot memorise exact tokens
        mutated_task = mutate_task(TASKS[task_id], seed=episode_seed)
        self._grader = CodeReviewGrader(mutated_task)
        self._episode = EpisodeState(task=mutated_task)

        # Inject cross-episode prior-knowledge hint when this task was seen before
        prior_hint = self._memory.prior_hint(task_id, TASKS[task_id])
        if prior_hint:
            self._episode.context_hints.append(prior_hint)
            log.debug("EpisodeMemory: injected prior hint for task %d", task_id)

        return self._build_observation(reward=0.0, done=False)

    async def async_step(
        self, action: ProbeAction
    ) -> tuple[ProbeObservation, RewardType, bool, dict[str, Any]]:
        self._step_count += 1
        current_task = self._episode.task
        done = False
        reward_obj: RewardType

        if action.action_type == ActionType.ADD_COMMENT:
            reward_obj = self._handle_add_comment(action)
        elif action.action_type == ActionType.GET_CONTEXT:
            reward_obj = self._handle_get_context(action)
        elif action.action_type == ActionType.RUN_SCANNER:
            reward_obj = self._handle_run_scanner()
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
        if not done and self._step_count >= current_task["max_steps"]:
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

        self._episode.cumulative_reward = round(
            self._episode.cumulative_reward + reward_obj.total, 4
        )
        obs = self._build_observation(reward=reward_obj.total, done=done)
        info = {
            "episode_id": self._episode_id,
            "cumulative_reward": self._episode.cumulative_reward,
            "issues_found": list(self._episode.issues_found),
            "review_decision": self._episode.review_decision,
        }
        return obs, reward_obj, done, info

    async def async_state(self) -> dict[str, Any]:
        task = self._episode.task
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "task_id": task["id"],
            "task_difficulty": task["difficulty"],
            "task_name": task["name"],
            "issues_found": list(self._episode.issues_found),
            "total_issues": len(task["issues"]),
            "review_decision": self._episode.review_decision,
            "review_submitted": self._episode.review_submitted,
            "cumulative_reward": self._episode.cumulative_reward,
            "max_steps": task["max_steps"],
            "scanner_used": self._episode.scanner_used,
        }

    # ── Sync wrappers (openenv / create_app compatibility) ────────────────

    def reset(self) -> ProbeObservation:  # type: ignore[override]
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.async_reset())
        # Called from inside a running loop (e.g. pytest-asyncio) -- run in a
        # fresh thread that has its own event loop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.async_reset()).result()

    def step(self, action: ProbeAction) -> ProbeObservation:  # type: ignore[override]
        """
        Sync step for openenv compatibility.
        Returns only the Observation (reward is embedded in obs.reward).
        Use async_step() for the full (obs, reward, done, info) tuple.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            obs, _, _, _ = asyncio.run(self.async_step(action))
            return obs
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            obs, _, _, _ = pool.submit(asyncio.run, self.async_step(action)).result()
            return obs

    @property
    def state(self) -> State:  # type: ignore[override]
        return State(episode_id=self._episode_id, step_count=self._step_count)

    # ── Action handlers ───────────────────────────────────────────────────

    def _handle_add_comment(self, action: ProbeAction) -> RewardType:
        entry = {
            "type": "comment",
            "line": action.line_number,
            "text": action.comment,
            "severity": action.severity.value if action.severity else None,
            "category": action.category.value if action.category else None,
        }
        self._episode.review_comments.append(entry)

        score, new_finds, breakdown = self._grader.score_comment(
            line_number=action.line_number,
            comment=action.comment,
            already_found=self._episode.issues_found,
        )
        self._episode.issues_found.extend(new_finds)

        clamped = round(max(-1.0, min(1.0, score)), 4)
        if new_finds:
            explanation = f"Identified issue(s): {new_finds}"
        elif score < 0:
            explanation = "False-positive comment — matched no known issue."
        else:
            explanation = "Comment recorded; no new issue matched."

        # ── Causal unlock: check whether any newly found issue reveals context
        self._unlock_causal_hints(new_finds)

        return RewardType(
            total=clamped,
            components=breakdown,
            passed=bool(new_finds),
            explanation=explanation,
            step=self._step_count,
            terminal=False,
        )

    def _unlock_causal_hints(self, newly_found: list[str]) -> None:
        """Append context hint text for any issue that has an 'unlocks' key."""
        task = self._episode.task
        hint_map: dict[str, str] = task.get("context_hints", {})
        for issue in task["issues"]:
            unlock_key = issue.get("unlocks")
            if (
                unlock_key
                and issue["id"] in newly_found
                and unlock_key not in self._episode.hints_unlocked
                and unlock_key in hint_map
            ):
                self._episode.hints_unlocked.add(unlock_key)
                self._episode.context_hints.append(hint_map[unlock_key])

    def _handle_get_context(
        self, action: ProbeAction
    ) -> RewardType:
        """
        GET_CONTEXT — reveal ±5 lines around the requested line number.

        Costs a small step penalty (-0.01) to discourage random probing,
        but rewards focused investigation (line near an actual issue: 0.0
        net cost — penalty waived).
        """
        line_number = action.line_number
        task = self._episode.task
        code_lines = task["code"].split("\n")

        if line_number is None:
            return RewardType(
                total=-0.02,
                components={"invalid_context_probe": -0.02},
                passed=False,
                explanation="GET_CONTEXT requires a line_number.",
                step=self._step_count,
                terminal=False,
            )

        # Build context snippet centred on the requested line
        window_start = max(0, line_number - 6)
        window_end = min(len(code_lines), line_number + 5)
        snippet = "\n".join(
            f"{idx + 1:3}: {code_lines[idx]}"
            for idx in range(window_start, window_end)
        )

        is_near_known_issue = any(
            (iss["line_range"][0] - LINE_TOLERANCE) <= line_number <= (iss["line_range"][1] + LINE_TOLERANCE)
            for iss in task["issues"]
        )
        penalty = 0.0 if is_near_known_issue else -0.01

        self._episode.review_comments.append({
            "type": "context_probe",
            "line": line_number,
            "context": snippet,
        })

        return RewardType(
            total=penalty,
            components={"context_probe_penalty": penalty},
            passed=is_near_known_issue,
            explanation=f"Context around line {line_number}:\n{snippet}",
            step=self._step_count,
            terminal=False,
        )

    def _handle_run_scanner(self) -> RewardType:
        """
        RUN_SCANNER — invoke the simulated static-analysis tool.

        Reward design
        ─────────────
        • First use in an episode: free (+0.0) — the agent should always
          try the scanner at least once.
        • Repeated use costs -0.02 per call — the tool output doesn't change
          within an episode (same seed), so redundant calls waste the step
          budget without new information.

        The scan result is stored in ``review_comments`` so it appears in
        ``review_history`` on the next observation.  The agent must still
        call ``ADD_COMMENT`` to earn reward from any finding.
        """
        task = self._episode.task
        episode_seed = task.get("_mutation_seed", self._reset_count)
        is_first_scan = not self._episode.scanner_used
        self._episode.scanner_used = True

        scan_result = run_scanner(task, seed=episode_seed)

        self._episode.review_comments.append({
            "type": "scanner_result",
            "tool": scan_result["tool"],
            "findings": scan_result["findings"],
            "missed_count": scan_result["missed_count"],
            "note": scan_result["note"],
        })

        penalty = 0.0 if is_first_scan else -0.02
        finding_count = len(scan_result["findings"])
        explanation = (
            f"[{scan_result['tool']}] {finding_count} finding(s) reported "
            f"({scan_result['missed_count']} issue(s) may have been missed). "
            f"{scan_result['note']}"
        )
        if not is_first_scan:
            explanation = "Scanner already run this episode — results unchanged. " + explanation

        log.debug(
            "RUN_SCANNER: %d findings, missed=%d, seed=%d",
            finding_count, scan_result["missed_count"], episode_seed,
        )

        return RewardType(
            total=penalty,
            components={"scanner_penalty": penalty},
            passed=is_first_scan,
            explanation=explanation,
            step=self._step_count,
            terminal=False,
        )

    def _handle_request_changes(self, action: ProbeAction) -> RewardType:
        self._episode.review_decision = "request_changes"
        self._episode.review_comments.append(
            {"type": "request_changes", "text": action.comment}
        )
        if self._episode.issues_found:
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
        self._episode.review_decision = "approve"
        total_issue_count = len(self._episode.task["issues"])
        found_count = len(set(self._episode.issues_found))
        if total_issue_count > 0 and found_count < total_issue_count * 0.5:
            return RewardType(
                total=-0.15,
                components={"bad_approval_penalty": -0.15},
                passed=False,
                explanation=f"APPROVE with only {found_count}/{total_issue_count} issues found.",
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
        if self._episode.review_submitted:
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
        self._episode.review_submitted = True
        task = self._episode.task
        unique_issues_found = list(set(self._episode.issues_found))
        reward_obj = self._grader.final_score(
            issues_found=unique_issues_found,
            review_decision=self._episode.review_decision,
            steps_used=self._step_count,
            max_steps=task["max_steps"],
        )
        if unique_issues_found:
            self._memory.record(task["id"], unique_issues_found)
            log.debug(
                "EpisodeMemory: recorded %d finding(s) for task %d",
                len(unique_issues_found),
                task["id"],
            )
        return reward_obj, True

    # ── Observation builder ───────────────────────────────────────────────

    def _build_observation(self, reward: float, done: bool) -> ProbeObservation:
        task = self._episode.task
        return ProbeObservation(
            code_snippet=task["code"],
            task_description=task["description"],
            file_name=task["file_name"],
            task_id=task["id"],
            task_difficulty=task["difficulty"],
            review_history=list(self._episode.review_comments),
            step_count=self._step_count,
            max_steps=task["max_steps"],
            issues_found_count=len(set(self._episode.issues_found)),
            total_issues=len(task["issues"]),
            done=done,
            reward=round(max(-1.0, min(1.0, reward)), 4),
            context_hints=list(self._episode.context_hints),
            metadata={
                "cumulative_reward": self._episode.cumulative_reward,
                "review_decision": self._episode.review_decision,
                "episode_id": self._episode_id,
                "mutation_seed": task.get("_mutation_seed"),
            },
        )
