"""
Tests for the dynamic world features:
  - server/mutator.py      (code mutation engine)
  - Task 6                 (causal chain / progressive observation)
  - GET_CONTEXT action     (line-context probing)
  - Causal unlock chain    (context_hints injected into observation)
  - Tasks 3 & 5 unlocks    (causal chains across tasks)
  - EpisodeMemory          (cross-episode persistence)
  - RUN_SCANNER action     (live tool interaction, noisy results)
"""

import sys
import os
import copy

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.mutator import mutate_task
from server.CodeReviewAgent_environment import EpisodeState
from server.tasks import TASKS
from server.grader import CodeReviewGrader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASK6 = TASKS[6]   # causal chain task


def _grader(task):
    return CodeReviewGrader(task)


# ===========================================================================
# MUTATOR TESTS
# ===========================================================================

class TestMutator:

    def test_returns_deep_copy(self):
        """mutate_task must not modify the original TASKS entry."""
        original_code = TASKS[1]["code"]
        _ = mutate_task(TASKS[1], seed=0)
        assert TASKS[1]["code"] == original_code

    def test_mutation_seed_tag(self):
        """Mutated task carries _mutation_seed matching the supplied seed."""
        t = mutate_task(TASKS[1], seed=42)
        assert t["_mutation_seed"] == 42

    def test_different_seeds_differ(self):
        """Two different seeds should (almost always) produce different code."""
        t1 = mutate_task(TASKS[1], seed=0)
        t2 = mutate_task(TASKS[1], seed=1)
        # At minimum the blank-line insert shifts are different; codes differ
        assert t1["code"] != TASKS[1]["code"] or t2["code"] != TASKS[1]["code"]

    def test_same_seed_is_deterministic(self):
        """Same seed must always produce identical output."""
        t1 = mutate_task(TASKS[2], seed=99)
        t2 = mutate_task(TASKS[2], seed=99)
        assert t1["code"] == t2["code"]
        assert t1["issues"] == t2["issues"]

    def test_line_shift_applied(self):
        """Line shift must move every issue line_range down by exactly 1."""
        original = copy.deepcopy(TASKS[1])
        mutated = mutate_task(TASKS[1], seed=7)
        orig_ranges = [iss["line_range"] for iss in original["issues"]]
        mut_ranges = [iss["line_range"] for iss in mutated["issues"]]
        for orig_r, mut_r in zip(orig_ranges, mut_ranges):
            assert mut_r[0] == orig_r[0] + 1
            assert mut_r[1] == orig_r[1] + 1

    def test_issue_count_preserved(self):
        """Mutation must not add or remove issues."""
        for task in TASKS[:6]:   # skip task 6 here, tested separately
            mutated = mutate_task(task, seed=5)
            assert len(mutated["issues"]) == len(task["issues"])

    def test_issue_ids_preserved(self):
        """Issue ids must be unchanged after mutation."""
        original_ids = [i["id"] for i in TASKS[2]["issues"]]
        mutated_ids = [i["id"] for i in mutate_task(TASKS[2], seed=3)["issues"]]
        assert original_ids == mutated_ids

    def test_grader_still_matches_after_mutation(self):
        """
        The grader must still award credit after mutation.
        Use the off-by-one issue in task 1 — keyword 'range' is always present
        and line_range shifts by exactly 1.
        """
        mutated = mutate_task(TASKS[1], seed=10)
        g = _grader(mutated)
        off_by_one = next(i for i in mutated["issues"] if i["id"] == "off_by_one")
        target_line = off_by_one["line_range"][0]

        score, found, _ = g.score_comment(
            line_number=target_line,
            comment="off-by-one error: range(len + 1) causes IndexError on the last iteration",
            already_found=[],
        )
        assert "off_by_one" in found
        assert score > 0.0

    def test_correct_decision_preserved(self):
        """correct_decision must be unchanged by mutation."""
        for task in TASKS:
            mutated = mutate_task(task, seed=1)
            assert mutated["correct_decision"] == task["correct_decision"]


# ===========================================================================
# TASK 6 STRUCTURE TESTS
# ===========================================================================

class TestTask6Structure:

    def test_task6_exists(self):
        assert len(TASKS) >= 7, "Task 6 (causal chain) must exist in TASKS"

    def test_task6_has_context_hints(self):
        assert "context_hints" in TASK6
        assert len(TASK6["context_hints"]) >= 2

    def test_task6_unlock_keys_present(self):
        """Every 'unlocks' key in an issue must exist in context_hints dict."""
        hints = TASK6["context_hints"]
        for issue in TASK6["issues"]:
            key = issue.get("unlocks")
            if key:
                assert key in hints, f"Issue {issue['id']} unlocks '{key}' but key not in context_hints"

    def test_task6_total_weight_positive(self):
        g = _grader(TASK6)
        assert g.total_weight > 0.0

    def test_task6_has_chained_issues(self):
        """At least two issues must have an 'unlocks' field."""
        unlocking = [i for i in TASK6["issues"] if i.get("unlocks")]
        assert len(unlocking) >= 2

    def test_task6_correct_decision(self):
        assert TASK6["correct_decision"] == "request_changes"


# ===========================================================================
# CAUSAL UNLOCK CHAIN TESTS (environment layer)
# ===========================================================================

class TestCausalUnlock:
    """
    Test the unlock mechanic via the environment's _unlock_causal_hints helper
    and _handle_add_comment pipeline.
    """

    def _make_env(self):
        """Return a fresh environment instance fast-forwarded to task 6."""
        import asyncio
        try:
            from server.CodeReviewAgent_environment import ProbeEnvironment
        except ImportError:
            from CodeReviewAgent_environment import ProbeEnvironment  # type: ignore

        env = ProbeEnvironment()
        # force-set episode to task 6 (bypass cycling for test speed)
        from server.mutator import mutate_task as _mt
        from server.CodeReviewAgent_environment import EpisodeState
        task = _mt(TASK6, seed=0)
        from server.grader import CodeReviewGrader as _G
        env._grader = _G(task)
        env._episode = EpisodeState(task=task)
        return env

    def test_no_hints_at_start(self):
        env = self._make_env()
        assert env._episode.context_hints == []

    def test_unlock_fires_after_finding_trigger_issue(self):
        """Finding hardcoded_jwt_secret must append db_schema_hint."""
        env = self._make_env()
        jwt_issue = next(i for i in env._episode.task["issues"] if i["id"] == "hardcoded_jwt_secret")
        target_line = jwt_issue["line_range"][0]

        env._step_count = 1
        reward = env._handle_add_comment(
            type("A", (), {
                "line_number": target_line,
                "comment": "JWT_SECRET is hardcoded — must be loaded from environment variable to prevent token forgery",
                "severity": type("S", (), {"value": "critical"})(),
                "category": type("C", (), {"value": "security"})(),
            })()
        )
        assert "hardcoded_jwt_secret" in env._episode.issues_found
        assert len(env._episode.context_hints) == 1
        assert "db_schema_hint" in env._episode.hints_unlocked
        assert "Database Schema" in env._episode.context_hints[0]

    def test_unlock_fires_only_once(self):
        """The same hint must not be appended twice even if issue found again."""
        env = self._make_env()
        jwt_issue = next(i for i in env._episode.task["issues"] if i["id"] == "hardcoded_jwt_secret")
        target_line = jwt_issue["line_range"][0]

        for _ in range(3):
            env._step_count += 1
            env._handle_add_comment(
                type("A", (), {
                    "line_number": target_line,
                    "comment": "JWT_SECRET is hardcoded — must be loaded from environment variable",
                    "severity": type("S", (), {"value": "critical"})(),
                    "category": type("C", (), {"value": "security"})(),
                })()
            )
        assert len(env._episode.context_hints) == 1

    def test_second_unlock_fires_independently(self):
        """Finding no_rate_limit must append nginx_config_hint independently."""
        env = self._make_env()
        rate_issue = next(i for i in env._episode.task["issues"] if i["id"] == "no_rate_limit")
        target_line = rate_issue["line_range"][0]

        env._step_count = 1
        env._handle_add_comment(
            type("A", (), {
                "line_number": target_line,
                "comment": "No rate limiting on /auth endpoint — susceptible to brute-force attacks",
                "severity": type("S", (), {"value": "error"})(),
                "category": type("C", (), {"value": "security"})(),
            })()
        )
        assert "nginx_config_hint" in env._episode.hints_unlocked
        assert any("nginx" in h.lower() for h in env._episode.context_hints)

    def test_both_unlocks_can_fire_in_same_episode(self):
        """Both hints can be unlocked within one episode."""
        env = self._make_env()
        task = env._episode.task

        jwt_issue = next(i for i in task["issues"] if i["id"] == "hardcoded_jwt_secret")
        rate_issue = next(i for i in task["issues"] if i["id"] == "no_rate_limit")

        for step, (issue, kw) in enumerate([
            (jwt_issue, "JWT_SECRET is hardcoded — must be loaded from environment variable to prevent forgery"),
            (rate_issue, "No rate limiting on /auth endpoint — susceptible to brute-force attacks"),
        ], start=1):
            env._step_count = step
            env._handle_add_comment(
                type("A", (), {
                    "line_number": issue["line_range"][0],
                    "comment": kw,
                    "severity": type("S", (), {"value": "critical"})(),
                    "category": type("C", (), {"value": "security"})(),
                })()
            )

        assert len(env._episode.context_hints) == 2
        assert env._episode.hints_unlocked == {"db_schema_hint", "nginx_config_hint"}

    def test_context_hints_appear_in_observation(self):
        """context_hints list must be non-empty in the observation after an unlock."""
        env = self._make_env()
        jwt_issue = next(i for i in env._episode.task["issues"] if i["id"] == "hardcoded_jwt_secret")
        env._step_count = 1
        env._handle_add_comment(
            type("A", (), {
                "line_number": jwt_issue["line_range"][0],
                "comment": "JWT_SECRET is hardcoded — must be loaded from environment variable",
                "severity": type("S", (), {"value": "critical"})(),
                "category": type("C", (), {"value": "security"})(),
            })()
        )
        obs = env._build_observation(reward=0.0, done=False)
        assert len(obs.context_hints) == 1
        assert "Database Schema" in obs.context_hints[0]


# ===========================================================================
# GET_CONTEXT ACTION TESTS
# ===========================================================================

class TestGetContext:

    def _make_env(self):
        try:
            from server.CodeReviewAgent_environment import ProbeEnvironment
        except ImportError:
            from CodeReviewAgent_environment import ProbeEnvironment  # type: ignore
        from server.mutator import mutate_task as _mt
        from server.grader import CodeReviewGrader as _G
        env = ProbeEnvironment()
        task = _mt(TASKS[1], seed=0)
        env._grader = _G(task)
        env._episode = EpisodeState(task=task)
        return env

    def test_get_context_near_issue_no_penalty(self):
        """Probing a line near a real issue must cost 0.0."""
        env = self._make_env()
        issue_line = env._episode.task["issues"][0]["line_range"][0]
        env._step_count = 1
        reward = env._handle_get_context(
            type("A", (), {"line_number": issue_line})()
        )
        assert reward.total == 0.0
        assert reward.passed is True

    def test_get_context_far_from_issue_costs_penalty(self):
        """Probing a line far from any issue must cost -0.01."""
        env = self._make_env()
        env._step_count = 1
        reward = env._handle_get_context(
            type("A", (), {"line_number": 999})()
        )
        assert reward.total == pytest.approx(-0.01, abs=0.001)
        assert reward.passed is False

    def test_get_context_no_line_number_penalised(self):
        """GET_CONTEXT with no line_number must return -0.02."""
        env = self._make_env()
        env._step_count = 1
        reward = env._handle_get_context(
            type("A", (), {"line_number": None})()
        )
        assert reward.total == pytest.approx(-0.02, abs=0.001)

    def test_get_context_snippet_stored_in_history(self):
        """The context probe must be recorded in review_comments."""
        env = self._make_env()
        env._step_count = 1
        env._handle_get_context(
            type("A", (), {"line_number": 4})()
        )
        probes = [c for c in env._episode.review_comments if c.get("type") == "context_probe"]
        assert len(probes) == 1
        assert probes[0]["line"] == 4
        assert "context" in probes[0]

    def test_get_context_snippet_contains_requested_line(self):
        """The returned snippet must reference the requested line number."""
        env = self._make_env()
        env._step_count = 1
        reward = env._handle_get_context(
            type("A", (), {"line_number": 4})()
        )
        # explanation contains the formatted snippet with line numbers
        assert "4:" in reward.explanation or "4 :" in reward.explanation


# ===========================================================================
# TASK 3 & 5 CAUSAL UNLOCK TESTS
# ===========================================================================

class TestTask3CausalUnlocks:
    """Task 3 (data_pipeline) should unlock context hints via issue findings."""

    TASK3 = TASKS[3]

    def _make_env(self):
        try:
            from server.CodeReviewAgent_environment import ProbeEnvironment
        except ImportError:
            from CodeReviewAgent_environment import ProbeEnvironment  # type: ignore
        env = ProbeEnvironment()
        task = copy.deepcopy(self.TASK3)
        from server.grader import CodeReviewGrader as _G
        env._grader = _G(task)
        env._episode = EpisodeState(task=task)
        return env

    def test_task3_has_context_hints(self):
        """Task 3 must declare a context_hints dict with both expected keys."""
        hints = self.TASK3.get("context_hints", {})
        assert "api_docs_hint" in hints
        assert "network_topology_hint" in hints

    def test_task3_hardcoded_api_key_has_unlocks(self):
        """hardcoded_api_key issue must carry unlocks='api_docs_hint'."""
        issue = next(i for i in self.TASK3["issues"] if i["id"] == "hardcoded_api_key")
        assert issue.get("unlocks") == "api_docs_hint"

    def test_task3_ssl_disabled_has_unlocks(self):
        """ssl_disabled issue must carry unlocks='network_topology_hint'."""
        issue = next(i for i in self.TASK3["issues"] if i["id"] == "ssl_disabled")
        assert issue.get("unlocks") == "network_topology_hint"

    def test_task3_api_key_unlock_fires(self):
        """Finding hardcoded_api_key must append api_docs_hint to context_hints."""
        env = self._make_env()
        api_issue = next(i for i in env._episode.task["issues"] if i["id"] == "hardcoded_api_key")
        env._step_count = 1
        env._handle_add_comment(
            type("A", (), {
                "line_number": api_issue["line_range"][0],
                "comment": "API key is hardcoded in source — move to os.environ",
                "severity": type("S", (), {"value": "critical"})(),
                "category": type("C", (), {"value": "security"})(),
            })()
        )
        assert "api_docs_hint" in env._episode.hints_unlocked
        assert any("batch" in h for h in env._episode.context_hints)

    def test_task3_ssl_unlock_fires(self):
        """Finding ssl_disabled must append network_topology_hint to context_hints."""
        env = self._make_env()
        ssl_issue = next(i for i in env._episode.task["issues"] if i["id"] == "ssl_disabled")
        env._step_count = 1
        env._handle_add_comment(
            type("A", (), {
                "line_number": ssl_issue["line_range"][0],
                "comment": "SSL certificate verification disabled (verify=False) — MITM risk",
                "severity": type("S", (), {"value": "error"})(),
                "category": type("C", (), {"value": "security"})(),
            })()
        )
        assert "network_topology_hint" in env._episode.hints_unlocked
        assert any("internet" in h.lower() for h in env._episode.context_hints)

    def test_task3_hints_not_duplicated(self):
        """The same unlock key must not fire twice even if the issue is found twice."""
        env = self._make_env()
        api_issue = next(i for i in env._episode.task["issues"] if i["id"] == "hardcoded_api_key")
        for step in range(1, 4):
            env._step_count = step
            env._handle_add_comment(
                type("A", (), {
                    "line_number": api_issue["line_range"][0],
                    "comment": "Hardcoded API key — use environment variable",
                    "severity": type("S", (), {"value": "critical"})(),
                    "category": type("C", (), {"value": "security"})(),
                })()
            )
        api_hints = [h for h in env._episode.context_hints if "batch" in h]
        assert len(api_hints) == 1


class TestTask5CausalUnlocks:
    """Task 5 (Flask API) should unlock context hints via issue findings."""

    TASK5 = TASKS[5]

    def _make_env(self):
        try:
            from server.CodeReviewAgent_environment import ProbeEnvironment
        except ImportError:
            from CodeReviewAgent_environment import ProbeEnvironment  # type: ignore
        env = ProbeEnvironment()
        task = copy.deepcopy(self.TASK5)
        from server.grader import CodeReviewGrader as _G
        env._grader = _G(task)
        env._episode = EpisodeState(task=task)
        return env

    def test_task5_has_context_hints(self):
        """Task 5 must declare a context_hints dict with both expected keys."""
        hints = self.TASK5.get("context_hints", {})
        assert "server_config_hint" in hints
        assert "client_usage_hint" in hints

    def test_task5_command_injection_has_unlocks(self):
        """command_injection issue must carry unlocks='server_config_hint'."""
        issue = next(i for i in self.TASK5["issues"] if i["id"] == "command_injection")
        assert issue.get("unlocks") == "server_config_hint"

    def test_task5_insecure_deserialization_has_unlocks(self):
        """insecure_deserialization issue must carry unlocks='client_usage_hint'."""
        issue = next(i for i in self.TASK5["issues"] if i["id"] == "insecure_deserialization")
        assert issue.get("unlocks") == "client_usage_hint"

    def test_task5_command_injection_unlock_fires(self):
        """Finding command_injection must append server_config_hint."""
        env = self._make_env()
        ci_issue = next(i for i in env._episode.task["issues"] if i["id"] == "command_injection")
        env._step_count = 1
        env._handle_add_comment(
            type("A", (), {
                "line_number": ci_issue["line_range"][0],
                "comment": "Command injection via shell=True with unsanitised user input",
                "severity": type("S", (), {"value": "critical"})(),
                "category": type("C", (), {"value": "security"})(),
            })()
        )
        assert "server_config_hint" in env._episode.hints_unlocked
        assert any("root" in h or "privileged" in h for h in env._episode.context_hints)

    def test_task5_deserialization_unlock_fires(self):
        """Finding insecure_deserialization must append client_usage_hint."""
        env = self._make_env()
        deser_issue = next(i for i in env._episode.task["issues"] if i["id"] == "insecure_deserialization")
        env._step_count = 1
        env._handle_add_comment(
            type("A", (), {
                "line_number": deser_issue["line_range"][0],
                "comment": "pickle.loads on untrusted data — insecure deserialization RCE",
                "severity": type("S", (), {"value": "critical"})(),
                "category": type("C", (), {"value": "security"})(),
            })()
        )
        assert "client_usage_hint" in env._episode.hints_unlocked
        assert any("pickle" in h for h in env._episode.context_hints)


# ===========================================================================
# EPISODE MEMORY TESTS
# ===========================================================================

class TestEpisodeMemory:
    """Cross-episode memory — records findings and injects prior hints."""

    def _fresh_memory(self, tmp_path):
        from server.episode_memory import EpisodeMemory
        return EpisodeMemory(memory_dir=str(tmp_path), instance_id="test")

    def test_empty_memory_returns_no_hint(self, tmp_path):
        """New memory store must return None for any task."""
        mem = self._fresh_memory(tmp_path)
        assert mem.prior_hint(1, TASKS[1]) is None

    def test_record_and_retrieve(self, tmp_path):
        """After recording, prior_hint must return a non-None string."""
        mem = self._fresh_memory(tmp_path)
        mem.record(1, ["off_by_one", "assignment_not_update"])
        hint = mem.prior_hint(1, TASKS[1])
        assert hint is not None
        assert isinstance(hint, str)
        assert len(hint) > 20

    def test_hint_mentions_category(self, tmp_path):
        """Prior hint must mention the category of the recorded issue."""
        mem = self._fresh_memory(tmp_path)
        mem.record(1, ["off_by_one"])   # category='bug' in Task 1
        hint = mem.prior_hint(1, TASKS[1])
        assert "bug" in hint

    def test_hint_mentions_task_name(self, tmp_path):
        """Prior hint must mention the task name."""
        mem = self._fresh_memory(tmp_path)
        mem.record(1, ["off_by_one"])
        hint = mem.prior_hint(1, TASKS[1])
        assert TASKS[1]["name"] in hint

    def test_record_persists_across_instances(self, tmp_path):
        """Memory written by one instance must be readable by a fresh instance."""
        mem1 = self._fresh_memory(tmp_path)
        mem1.record(2, ["sql_injection", "eval_use"])
        mem2 = self._fresh_memory(tmp_path)
        hint = mem2.prior_hint(2, TASKS[2])
        assert hint is not None

    def test_record_deduplicates(self, tmp_path):
        """Recording the same issue_id twice must not inflate the stored list."""
        mem = self._fresh_memory(tmp_path)
        mem.record(1, ["off_by_one"])
        mem.record(1, ["off_by_one"])
        assert mem._data["1"].count("off_by_one") == 1

    def test_record_merges_across_calls(self, tmp_path):
        """Findings across two episodes must be merged, not overwritten."""
        mem = self._fresh_memory(tmp_path)
        mem.record(1, ["off_by_one"])
        mem.record(1, ["assignment_not_update"])
        assert set(mem._data["1"]) == {"off_by_one", "assignment_not_update"}

    def test_clear_single_task(self, tmp_path):
        """clear(task_id) must remove only that task's memory."""
        mem = self._fresh_memory(tmp_path)
        mem.record(1, ["off_by_one"])
        mem.record(2, ["sql_injection"])
        mem.clear(1)
        assert mem.prior_hint(1, TASKS[1]) is None
        assert mem.prior_hint(2, TASKS[2]) is not None

    def test_clear_all(self, tmp_path):
        """clear() with no args must wipe all memory."""
        mem = self._fresh_memory(tmp_path)
        mem.record(1, ["off_by_one"])
        mem.record(2, ["sql_injection"])
        mem.clear()
        assert mem.prior_hint(1, TASKS[1]) is None
        assert mem.prior_hint(2, TASKS[2]) is None

    def test_env_injects_prior_hint_on_second_reset(self, tmp_path):
        """After a full episode, the next reset for the same task_id must inject a hint."""
        try:
            from server.CodeReviewAgent_environment import ProbeEnvironment
        except ImportError:
            from CodeReviewAgent_environment import ProbeEnvironment  # type: ignore
        import asyncio

        env = ProbeEnvironment(memory_dir=str(tmp_path))
        # reset_count starts at 0; task_id = reset_count % len(TASKS)
        # Do one reset to consume task 0, then seed task-1 memory.
        asyncio.run(env.async_reset())   # reset_count → 1; ran task 0

        # Manually seed task-1 memory so the next task-1 reset gets a hint.
        task1_id = TASKS[1]["id"]   # == 1
        env._memory.record(task1_id, ["off_by_one"])

        # Cycle through tasks 1..6 (6 resets) so reset_count reaches 7 (≡ task 0).
        # Then one more reset puts us at reset_count=8 (≡ task 1) with prior memory.
        for _ in range(len(TASKS)):
            asyncio.run(env.async_reset())

        # reset_count is now (1 + len(TASKS) + 1) % len(TASKS) == 1 → task 1
        obs = asyncio.run(env.async_reset())
        assert any("PRIOR KNOWLEDGE" in h for h in obs.context_hints)

    def test_env_records_memory_after_submit(self, tmp_path):
        """Submitting a review with findings must persist them in EpisodeMemory."""
        try:
            from server.CodeReviewAgent_environment import ProbeEnvironment
        except ImportError:
            from CodeReviewAgent_environment import ProbeEnvironment  # type: ignore
        import asyncio
        from models import ProbeAction, ActionType

        env = ProbeEnvironment(memory_dir=str(tmp_path))
        asyncio.run(env.async_reset())   # task 0

        # Add a correct comment on task 0 bootstrap issue
        bootstrap_issue = next(
            i for i in env._episode.task["issues"] if i["id"] == "bootstrap_off_by_one"
        )
        add_action = ProbeAction(
            action_type=ActionType.ADD_COMMENT,
            line_number=bootstrap_issue["line_range"][0],
            comment="Off-by-one error: range(len+1) causes IndexError on the last iteration",
            severity=None,
            category=None,
        )
        asyncio.run(env.async_step(add_action))

        # Submit review
        from models import ActionType as AT
        submit_action = ProbeAction(
            action_type=AT.SUBMIT_REVIEW,
            line_number=None,
            comment=None,
            severity=None,
            category=None,
        )
        asyncio.run(env.async_step(submit_action))

        # Memory for task 0 must now be non-empty
        assert env._memory._data.get("0") is not None
        assert len(env._memory._data["0"]) > 0


# ===========================================================================
# RUN_SCANNER TESTS
# ===========================================================================

class TestRunScanner:
    """Tests for the scanner module and RUN_SCANNER action handler."""

    def _make_env(self, task_index: int = 1):
        try:
            from server.CodeReviewAgent_environment import ProbeEnvironment
        except ImportError:
            from CodeReviewAgent_environment import ProbeEnvironment  # type: ignore
        env = ProbeEnvironment()
        import copy
        task = copy.deepcopy(TASKS[task_index])
        from server.grader import CodeReviewGrader as _G
        from server.mutator import mutate_task as _mt
        task = _mt(task, seed=7)
        env._grader = _G(task)
        env._episode = EpisodeState(task=task)
        return env

    # ── scanner module unit tests ────────────────────────────────────────

    def test_scanner_returns_required_keys(self):
        """run_scanner must return dict with tool, findings, missed_count, note."""
        from server.scanner import run_scanner
        result = run_scanner(TASKS[1], seed=0)
        assert "tool" in result
        assert "findings" in result
        assert "missed_count" in result
        assert "note" in result

    def test_scanner_findings_are_list(self):
        """findings must be a list."""
        from server.scanner import run_scanner
        result = run_scanner(TASKS[1], seed=0)
        assert isinstance(result["findings"], list)

    def test_scanner_finding_has_required_fields(self):
        """Every finding dict must have line, rule, message, category, severity, verified."""
        from server.scanner import run_scanner
        result = run_scanner(TASKS[2], seed=42)
        for f in result["findings"]:
            for key in ("line", "rule", "message", "category", "severity", "verified"):
                assert key in f, f"Missing key '{key}' in finding: {f}"

    def test_scanner_verified_always_false(self):
        """All scanner findings start unverified — agent must confirm them."""
        from server.scanner import run_scanner
        result = run_scanner(TASKS[2], seed=99)
        for f in result["findings"]:
            assert f["verified"] is False

    def test_scanner_recall_below_100_percent(self):
        """With enough seeds, at least some issues must be missed (recall < 1.0)."""
        from server.scanner import run_scanner
        total_issues = len(TASKS[2]["issues"])   # 5 issues in Task 2
        missed_any = any(
            run_scanner(TASKS[2], seed=s)["missed_count"] > 0
            for s in range(20)
        )
        assert missed_any, "Scanner should miss at least one issue across 20 seeds"

    def test_scanner_deterministic_per_seed(self):
        """Same seed must produce identical results."""
        from server.scanner import run_scanner
        r1 = run_scanner(TASKS[3], seed=123)
        r2 = run_scanner(TASKS[3], seed=123)
        assert r1["findings"] == r2["findings"]
        assert r1["missed_count"] == r2["missed_count"]

    def test_scanner_different_seeds_differ(self):
        """Different seeds should (almost always) produce different findings."""
        from server.scanner import run_scanner
        results = {
            tuple(f["line"] for f in run_scanner(TASKS[3], seed=s)["findings"])
            for s in range(10)
        }
        assert len(results) > 1, "Scanner findings should vary across seeds"

    def test_scanner_line_numbers_within_code(self):
        """All reported line numbers must be within the code's line count."""
        from server.scanner import run_scanner
        task = TASKS[2]
        total_lines = len(task["code"].split("\n"))
        result = run_scanner(task, seed=5)
        for f in result["findings"]:
            assert 1 <= f["line"] <= total_lines, (
                f"Finding line {f['line']} out of range [1, {total_lines}]"
            )

    def test_scanner_tool_is_known_string(self):
        """tool field must be a non-empty string."""
        from server.scanner import run_scanner
        result = run_scanner(TASKS[1], seed=0)
        assert isinstance(result["tool"], str)
        assert len(result["tool"]) > 0

    # ── RUN_SCANNER action handler tests ────────────────────────────────

    def test_run_scanner_first_call_free(self):
        """First RUN_SCANNER in an episode must cost 0.0."""
        env = self._make_env()
        env._step_count = 1
        reward = env._handle_run_scanner()
        assert reward.total == 0.0
        assert reward.passed is True

    def test_run_scanner_repeated_penalised(self):
        """Second RUN_SCANNER call must cost -0.02."""
        env = self._make_env()
        env._step_count = 1
        env._handle_run_scanner()       # first — free
        env._step_count = 2
        reward = env._handle_run_scanner()   # second — penalised
        assert reward.total == pytest.approx(-0.02, abs=0.001)
        assert reward.passed is False

    def test_run_scanner_stored_in_history(self):
        """Scanner result must be stored as 'scanner_result' in review_comments."""
        env = self._make_env()
        env._step_count = 1
        env._handle_run_scanner()
        scanner_entries = [
            c for c in env._episode.review_comments if c.get("type") == "scanner_result"
        ]
        assert len(scanner_entries) == 1
        entry = scanner_entries[0]
        assert "tool" in entry
        assert "findings" in entry
        assert "note" in entry

    def test_run_scanner_sets_scanner_used_flag(self):
        """scanner_used flag must be False before, True after first call."""
        env = self._make_env()
        assert env._episode.scanner_used is False
        env._step_count = 1
        env._handle_run_scanner()
        assert env._episode.scanner_used is True

    def test_run_scanner_result_appears_in_obs_history(self):
        """After RUN_SCANNER, the next observation's review_history must contain the result."""
        env = self._make_env()
        env._step_count = 1
        env._handle_run_scanner()
        obs = env._build_observation(reward=0.0, done=False)
        scanner_entries = [
            e for e in obs.review_history if e.get("type") == "scanner_result"
        ]
        assert len(scanner_entries) == 1

    def test_run_scanner_via_async_step(self):
        """RUN_SCANNER dispatched through async_step must return a valid reward."""
        import asyncio
        try:
            from server.CodeReviewAgent_environment import ProbeEnvironment
        except ImportError:
            from CodeReviewAgent_environment import ProbeEnvironment  # type: ignore
        from models import ProbeAction, ActionType

        env = ProbeEnvironment()
        asyncio.run(env.async_reset())
        action = ProbeAction(
            action_type=ActionType.RUN_SCANNER,
            line_number=None,
            comment=None,
            severity=None,
            category=None,
        )
        obs, reward, done, info = asyncio.run(env.async_step(action))
        assert reward.total == 0.0          # first use is free
        assert done is False
        assert any(
            e.get("type") == "scanner_result" for e in obs.review_history
        )

    def test_scanner_used_tracked_in_async_state(self):
        """async_state must reflect scanner_used after the action fires."""
        import asyncio
        try:
            from server.CodeReviewAgent_environment import ProbeEnvironment
        except ImportError:
            from CodeReviewAgent_environment import ProbeEnvironment  # type: ignore
        from models import ProbeAction, ActionType

        env = ProbeEnvironment()
        asyncio.run(env.async_reset())
        state_before = asyncio.run(env.async_state())
        assert state_before["scanner_used"] is False

        action = ProbeAction(
            action_type=ActionType.RUN_SCANNER,
            line_number=None, comment=None, severity=None, category=None,
        )
        asyncio.run(env.async_step(action))
        state_after = asyncio.run(env.async_state())
        assert state_after["scanner_used"] is True
