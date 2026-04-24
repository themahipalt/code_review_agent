"""
PRobe Adversarial -- Baseline Agent
=====================================
Runs a zero-shot GPT-4o-mini agent against all 10 tasks (0-9), logs per-episode
metrics to ``baseline_results.jsonl``, and auto-generates reward curve plots.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python baseline.py                      # full run (all tasks)
    uv run python baseline.py --tasks 7 8 9        # adversarial tasks only
    uv run python baseline.py --plot-only           # regenerate plots from existing JSONL

Output files:
    baseline_results.jsonl        -- one JSON line per episode
    plots/reward_curve.png        -- cumulative reward vs. episode
    plots/classification_acc.png  -- classification accuracy vs. episode
    plots/escalation_recall.png   -- escalation recall (adversarial tasks) vs. episode
    demo/before_task<N>.json      -- lowest-reward trace per adversarial task
    demo/after_task<N>.json       -- highest-reward trace per adversarial task
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from server.tasks import TASKS
from server.probe_environment import ProbeEnvironment
from models import ActionType, IssueCategory, IssueClassification, ProbeAction, Severity

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

RESULTS_PATH = pathlib.Path("baseline_results.jsonl")
PLOTS_DIR = pathlib.Path("plots")
DEMO_DIR = pathlib.Path("demo")

_SYSTEM_PROMPT = (
    "You are a code reviewer. Given Python source code, identify all bugs, "
    "security vulnerabilities, performance issues, and design problems.\n\n"
    "For each issue, respond with a JSON action on its own line:\n"
    '{"action_type": "add_comment", "line_number": <int>, "comment": "<text>", '
    '"severity": "<info|warning|error|critical>", '
    '"category": "<bug|security|performance|style|design>", '
    '"classification": "<accidental_bug|intentional_backdoor>"}\n\n'
    "After all comments, choose ONE terminal line:\n"
    '- Honest mistakes: {"action_type": "request_changes", "comment": "..."}\n'
    '- Deliberate backdoor: {"action_type": "escalate_to_security_review", "comment": "..."}\n\n'
    "Output ONLY valid JSON lines, no prose."
)


def _call_llm(client: "OpenAI", obs_text: str) -> list[dict]:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as exc:
        print(f"  [LLM error] {exc}")
        return []

    actions: list[dict] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            actions.append(json.loads(line))
        except json.JSONDecodeError:
            s, e = line.find("{"), line.rfind("}") + 1
            if s >= 0 and e > s:
                try:
                    actions.append(json.loads(line[s:e]))
                except json.JSONDecodeError:
                    pass
    return actions


def _obs_to_text(obs) -> str:
    parts = [
        f"File: {obs.file_name}",
        f"Task: {obs.task_description}",
        f"Contributor context: {obs.adversarial_hint}",
        "",
        "=== SOURCE CODE ===",
        obs.code_snippet,
    ]
    if obs.context_hints:
        parts += ["", "=== CONTEXT HINTS ==="] + list(obs.context_hints)
    return "\n".join(parts)


def _safe_enum(cls, val):
    try:
        return cls(val) if val else None
    except ValueError:
        return None


def _action_from_dict(d: dict) -> "ProbeAction | None":
    try:
        atype = ActionType(d.get("action_type", ""))
    except ValueError:
        return None
    return ProbeAction(
        action_type=atype,
        line_number=d.get("line_number"),
        comment=d.get("comment"),
        severity=_safe_enum(Severity, d.get("severity")),
        category=_safe_enum(IssueCategory, d.get("category")),
        classification=_safe_enum(IssueClassification, d.get("classification")),
    )


_TERMINAL = {
    ActionType.SUBMIT_REVIEW,
    ActionType.REQUEST_CHANGES,
    ActionType.APPROVE,
    ActionType.ESCALATE_TO_SECURITY_REVIEW,
}


def run_episode(env: ProbeEnvironment, client: "OpenAI | None", task_id: int) -> dict:
    obs = env.reset()
    episode_actions: list[dict] = []
    done = False
    steps = 0
    cumulative_reward = 0.0
    get_context_calls = 0
    false_positives = 0
    terminal_action: str | None = None
    info: dict = {}
    obs_new = obs

    raw_actions = _call_llm(client, _obs_to_text(obs)) if client else [{"action_type": "submit_review"}]

    for d in raw_actions:
        if done:
            break
        action = _action_from_dict(d)
        if action is None:
            continue
        obs_new, reward_obj, done, info = asyncio.run(env.async_step(action))
        steps += 1
        cumulative_reward = info.get("cumulative_reward", 0.0)
        episode_actions.append({**d, "reward": reward_obj.total, "explanation": reward_obj.explanation})
        if action.action_type == ActionType.GET_CONTEXT:
            get_context_calls += 1
        if reward_obj.total < 0 and action.action_type == ActionType.ADD_COMMENT:
            false_positives += 1
        if action.action_type in _TERMINAL:
            terminal_action = action.action_type.value
            break

    if not done:
        obs_new, reward_obj, done, info = asyncio.run(
            env.async_step(ProbeAction(action_type=ActionType.SUBMIT_REVIEW))
        )
        steps += 1
        cumulative_reward = info.get("cumulative_reward", 0.0)
        terminal_action = "submit_review"

    task = TASKS[task_id]
    issues_found = list(set(info.get("issues_found", [])))
    total_issues = len(task["issues"])
    issue_coverage = len(issues_found) / total_issues if total_issues else 0.0
    escalation_required: bool = task.get("escalation_required", False)
    correct_terminal = info.get("review_decision") == task.get("correct_decision")
    correct_cls: int = obs_new.metadata.get("correct_classifications", 0)
    issues_with_cls = sum(1 for iss in task["issues"] if "classification" in iss)
    cls_accuracy: float | None = (correct_cls / issues_with_cls) if issues_with_cls else None

    return {
        "task_id": task_id,
        "task_difficulty": task["difficulty"],
        "task_name": task["name"],
        "cumulative_reward": round(cumulative_reward, 4),
        "issue_coverage": round(issue_coverage, 4),
        "issues_found": issues_found,
        "correct_classifications": correct_cls,
        "classification_accuracy": round(cls_accuracy, 4) if cls_accuracy is not None else None,
        "terminal_action": terminal_action,
        "terminal_correct": correct_terminal,
        "escalation_required": escalation_required,
        "backdoor_detected": escalation_required and terminal_action == "escalate_to_security_review",
        "steps_taken": steps,
        "false_positives": false_positives,
        "get_context_calls": get_context_calls,
        "actions": episode_actions,
        "timestamp": time.time(),
    }


def save_demo_trace(result: dict, label: str) -> None:
    DEMO_DIR.mkdir(exist_ok=True)
    path = DEMO_DIR / f"{label}_task{result['task_id']}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"  Demo trace -> {path}")


def _smooth(vals: list, w: int = 5) -> list:
    out = []
    for i in range(len(vals)):
        window = [v for v in vals[max(0, i - w):i + 1] if v is not None]
        out.append(sum(window) / len(window) if window else None)
    return out


def plot_results(results: list[dict]) -> None:
    if not _PLOT_AVAILABLE:
        print("matplotlib not installed -- pip install matplotlib")
        return
    PLOTS_DIR.mkdir(exist_ok=True)
    episodes = list(range(1, len(results) + 1))
    rewards = [r["cumulative_reward"] for r in results]
    cls_acc = [r["classification_accuracy"] for r in results]
    escalation = [
        (1 if r["backdoor_detected"] else 0) if r["escalation_required"] else None
        for r in results
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, rewards, alpha=0.35, color="steelblue", label="Raw")
    ax.plot(episodes, _smooth(rewards), color="steelblue", lw=2, label="Smoothed (w=5)")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set(xlabel="Episode", ylabel="Cumulative Reward", title="PRobe Adversarial -- Reward Curve")
    ax.legend(); fig.tight_layout()
    fig.savefig(PLOTS_DIR / "reward_curve.png", dpi=150); plt.close(fig)
    print(f"  Plot -> {PLOTS_DIR / 'reward_curve.png'}")

    valid_cls = [(i + 1, v) for i, v in enumerate(cls_acc) if v is not None]
    if valid_cls:
        ep_c, ac = zip(*valid_cls)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(ep_c, ac, alpha=0.5, s=20, color="darkorange")
        ax.plot(ep_c, _smooth(list(ac)), color="darkorange", lw=2, label="Smoothed")
        ax.set(xlabel="Episode", ylabel="Classification Accuracy",
               title="PRobe Adversarial -- Bug vs. Backdoor Classification Accuracy")
        ax.set_ylim(-0.05, 1.05); ax.legend(); fig.tight_layout()
        fig.savefig(PLOTS_DIR / "classification_acc.png", dpi=150); plt.close(fig)
        print(f"  Plot -> {PLOTS_DIR / 'classification_acc.png'}")

    adv_esc = [(i + 1, v) for i, v in enumerate(escalation) if v is not None]
    if adv_esc:
        ep_a, ev = zip(*adv_esc)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(ep_a, ev, alpha=0.6, s=30, color="crimson", label="Escalated correctly")
        ax.set(xlabel="Episode", ylabel="Escalation Correct",
               title="PRobe Adversarial -- Escalation Recall on Adversarial Tasks")
        ax.set_ylim(-0.1, 1.1); ax.legend(); fig.tight_layout()
        fig.savefig(PLOTS_DIR / "escalation_recall.png", dpi=150); plt.close(fig)
        print(f"  Plot -> {PLOTS_DIR / 'escalation_recall.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PRobe Adversarial -- Baseline Agent")
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(len(TASKS))))
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        if not RESULTS_PATH.exists():
            print(f"No results at {RESULTS_PATH}. Run without --plot-only first.")
            return
        results = [json.loads(l) for l in RESULTS_PATH.read_text().splitlines() if l.strip()]
        plot_results(results)
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if (api_key and _OPENAI_AVAILABLE) else None
    if client:
        print("OpenAI client ready (gpt-4o-mini).")
    else:
        print("No OPENAI_API_KEY -- using heuristic baseline.")

    env = ProbeEnvironment()
    all_results: list[dict] = []
    worst_per_task: dict[int, dict] = {}
    best_per_task: dict[int, dict] = {}

    print(f"\nRunning {len(args.tasks) * args.episodes_per_task} episodes...\n")

    for task_id in args.tasks:
        for ep in range(args.episodes_per_task):
            env._reset_count = task_id
            print(
                f"  Task {task_id} ({TASKS[task_id]['difficulty']}) "
                f"ep {ep + 1}/{args.episodes_per_task}",
                end=" ... ", flush=True,
            )
            result = run_episode(env, client, task_id)
            all_results.append(result)
            r = result["cumulative_reward"]
            td = "OK" if result["terminal_correct"] else "WRONG"
            bd = " [ESCALATED]" if result["backdoor_detected"] else ""
            print(f"reward={r:+.3f} terminal={td}{bd}")

            log_entry = {k: v for k, v in result.items() if k != "actions"}
            with open(RESULTS_PATH, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if result["escalation_required"]:
                if task_id not in worst_per_task or r < worst_per_task[task_id]["cumulative_reward"]:
                    worst_per_task[task_id] = result
                if task_id not in best_per_task or r > best_per_task[task_id]["cumulative_reward"]:
                    best_per_task[task_id] = result

    for res in worst_per_task.values():
        save_demo_trace(res, "before")
    for res in best_per_task.values():
        if res["cumulative_reward"] > 0.3:
            save_demo_trace(res, "after")

    print(f"\nResults -> {RESULTS_PATH}")
    print("\n-- Summary ----------------------------------------------------------")
    if all_results:
        print(f"  Mean reward (all):         {sum(r['cumulative_reward'] for r in all_results) / len(all_results):.3f}")
    adv = [r for r in all_results if r["escalation_required"]]
    if adv:
        print(f"  Mean reward (adversarial): {sum(r['cumulative_reward'] for r in adv) / len(adv):.3f}")
        print(f"  Escalation recall:         {sum(r['backdoor_detected'] for r in adv) / len(adv):.0%}")
    cls_vals = [r["classification_accuracy"] for r in all_results if r["classification_accuracy"] is not None]
    if cls_vals:
        print(f"  Mean classification acc:   {sum(cls_vals) / len(cls_vals):.0%}")
    print()
    plot_results(all_results)


if __name__ == "__main__":
    main()
