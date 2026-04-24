"""
PRobe — Scripted Rule-Based Baseline Agents
============================================
Four deterministic agents that stress-test the reward verifier without any ML:

  1. keyword_spammer   — dumps all known keywords at wrong lines → should score LOW
  2. line_flooder      — comments on every 5th line with generic text → should score LOW
  3. perfect_oracle    — uses ground-truth task data to construct the ideal output → scores HIGH
  4. random_agent      — randomly picks lines and generic category labels → scores ~0

Run:
  uv run python scripted_baseline.py

Outputs:
  outputs/baseline_comparison.png   — bar chart comparing all four agents
  outputs/scripted_baseline.jsonl   — per-task per-agent scores
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from server.tasks import TASKS
from server.mutator import mutate_task
from train_grpo import compute_reward

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLOT = True
except ImportError:
    _PLOT = False

OUTPUTS_DIR = pathlib.Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

ALL_KEYWORDS = [
    "sql injection", "hardcoded", "off-by-one", "eval", "pickle", "md5",
    "race condition", "resource leak", "command injection", "path traversal",
    "backdoor", "authentication bypass", "jwt", "ssl bypass", "buffer overflow",
]


def keyword_spammer(task: dict, seed: int = 0) -> str:
    """
    Dumps every known keyword into a single comment at a wrong line.
    Tests that keyword-only matches WITHOUT line proximity earn no credit.
    """
    comment_text = "Issues found: " + ", ".join(ALL_KEYWORDS)
    payload = [
        {
            "line": 9999,  # deliberately wrong line
            "comment": comment_text,
            "severity": "critical",
            "category": "security",
            "classification": "accidental_bug",
        }
    ]
    return json.dumps(payload) + "\nDECISION: request_changes"


def line_flooder(task: dict, seed: int = 0) -> str:
    """
    Posts a generic vague comment on every 5th line.
    Tests that vague/short comments earn no credit even at correct line numbers.
    """
    code_lines = task["code"].splitlines()
    comments = []
    for i in range(0, len(code_lines), 5):
        comments.append({
            "line": i + 1,
            "comment": "potential issue",  # intentionally too short / vague
            "severity": "warning",
            "category": "bug",
            "classification": "accidental_bug",
        })
    return json.dumps(comments) + "\nDECISION: request_changes"


def perfect_oracle(task: dict, seed: int = 0) -> str:
    """
    Constructs the ideal review by reading ground-truth issue data.
    This sets the upper-bound ceiling and validates the grader awards max credit.
    """
    mutated = mutate_task(task, seed=seed)
    comments = []
    for issue in mutated["issues"]:
        line = (issue["line_range"][0] + issue["line_range"][1]) // 2
        kw = issue["keywords"][0]
        cls = issue.get("classification", "accidental_bug")
        comments.append({
            "line": line,
            "comment": f"{kw}: {issue.get('description', kw)} — must be fixed immediately",
            "severity": "critical",
            "category": "security",
            "classification": cls,
        })
    decision = mutated.get("correct_decision", "request_changes")
    return json.dumps(comments) + f"\nDECISION: {decision}"


def random_agent(task: dict, seed: int = 0) -> str:
    """
    Picks random lines with random categories.
    Establishes the random baseline floor.
    """
    rng = random.Random(seed)
    code_lines = task["code"].splitlines()
    n = min(len(task["issues"]), len(code_lines))
    lines = rng.sample(range(1, len(code_lines) + 1), k=max(1, n))
    comments = []
    categories = ["bug", "security", "performance", "style"]
    for ln in lines:
        comments.append({
            "line": ln,
            "comment": f"Possible {rng.choice(categories)} issue at this location worth investigating",
            "severity": rng.choice(["info", "warning", "error"]),
            "category": rng.choice(categories),
            "classification": rng.choice(["accidental_bug", "intentional_backdoor"]),
        })
    decision = rng.choice(["request_changes", "approve", "escalate_to_security_review"])
    return json.dumps(comments) + f"\nDECISION: {decision}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

AGENTS = {
    "keyword_spammer": keyword_spammer,
    "line_flooder": line_flooder,
    "perfect_oracle": perfect_oracle,
    "random_agent": random_agent,
}

EXPECTED_RANKING = ["perfect_oracle", "random_agent", "line_flooder", "keyword_spammer"]


def run_evaluation() -> dict[str, list[float]]:
    results: dict[str, list[float]] = {name: [] for name in AGENTS}
    records: list[dict] = []

    print("\nScripted Baseline Evaluation")
    print("=" * 60)
    print(f"{'Agent':<20} {'Task':<6} {'Diff':<12} {'Reward':>8}")
    print("-" * 60)

    for task in TASKS:
        for agent_name, agent_fn in AGENTS.items():
            raw = agent_fn(task, seed=42)
            score = compute_reward(task, raw, seed=42)
            r = score["total"]
            results[agent_name].append(r)
            records.append({
                "agent": agent_name,
                "task_id": task["id"],
                "task_difficulty": task["difficulty"],
                "reward_total": r,
                "issue_reward": score["issue_reward"],
                "classification_reward": score["classification_reward"],
                "false_positive_penalty": score["false_positive_penalty"],
                "format_bonus": score.get("format_bonus", 0.0),
                "coverage_bonus": score["coverage_bonus"],
                "decision_score": score["decision_score"],
            })
            print(f"  {agent_name:<18} T{task['id']:<5} {task['difficulty']:<12} {r:+.4f}")

    # Save JSONL
    jsonl_path = OUTPUTS_DIR / "scripted_baseline.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nSaved {jsonl_path}")

    return results


def print_summary(results: dict[str, list[float]]) -> None:
    print("\n" + "=" * 60)
    print("Summary (mean reward across all 10 tasks)")
    print("=" * 60)
    means = {name: sum(vals) / len(vals) for name, vals in results.items()}
    for name in sorted(means, key=lambda n: -means[n]):
        bar = "#" * int(max(0, means[name]) * 30)
        print(f"  {name:<20} {means[name]:+.4f}  {bar}")

    # Verify anti-gaming property
    print("\nAnti-gaming check:")
    oracle_mean = means["perfect_oracle"]
    for bad_agent in ["keyword_spammer", "line_flooder"]:
        ratio = means[bad_agent] / oracle_mean if oracle_mean > 0 else 0
        ok = "PASS" if ratio < 0.4 else "FAIL"
        print(f"  {bad_agent:<20} scores {ratio:.0%} of oracle  [{ok}]")


def plot_comparison(results: dict[str, list[float]]) -> None:
    if not _PLOT:
        print("matplotlib not available — skipping plot")
        return

    task_ids = list(range(len(TASKS)))
    agent_names = list(AGENTS.keys())
    colors = ["tomato", "gold", "steelblue", "mediumpurple"]
    n = len(agent_names)
    width = 0.8 / n

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # -- Top panel: per-task bars ------------------------------------------
    ax = axes[0]
    for i, (name, color) in enumerate(zip(agent_names, colors)):
        x = [t + (i - n / 2 + 0.5) * width for t in task_ids]
        ax.bar(x, results[name], width=width * 0.9, label=name, color=color, alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Task ID")
    ax.set_ylabel("Reward")
    ax.set_title("PRobe — Scripted Baseline Agents: Per-Task Reward")
    ax.set_xticks(task_ids)
    task_labels = [f"T{t['id']}\n{t['difficulty'][:4]}" for t in TASKS]
    ax.set_xticklabels(task_labels)
    ax.legend(loc="upper right", fontsize=9)

    # -- Bottom panel: mean reward bar chart --------------------------------
    ax = axes[1]
    means = {name: sum(vals) / len(vals) for name, vals in results.items()}
    sorted_agents = sorted(means.items(), key=lambda x: -x[1])
    names, vals = zip(*sorted_agents)
    bar_colors = [colors[agent_names.index(n)] for n in names]
    bars = ax.bar(names, vals, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:+.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Mean Reward (all 10 tasks)")
    ax.set_title("PRobe — Mean Reward by Agent Type\n(oracle ≫ random ≫ spammer validates reward is hard to game)")

    fig.tight_layout()
    out = OUTPUTS_DIR / "baseline_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    results = run_evaluation()
    print_summary(results)
    plot_comparison(results)
    print("\nDone.")
