"""
PRobe — GRPO Training Script
==============================
Trains a causal LLM to perform adversarial-aware code review using Group
Relative Policy Optimisation (GRPO) via HuggingFace TRL, with optional
Unsloth acceleration for 4-bit quantised fine-tuning on a single GPU.

Architecture
------------
The LLM operates in *single-turn* mode: it receives the full task prompt
(code + contributor context + any unlocked hints) and must output a JSON
array of review comments plus a terminal action in one completion.
The PRobe grader scores the output deterministically — no LLM judge.

Curriculum (5 phases, controlled by --curriculum / auto-detected from step):
  Phase 0 (steps   0– 40): Tasks 0–1   — ultra-easy / easy
  Phase 1 (steps  40– 80): Tasks 0–3   — adds medium / hard
  Phase 2 (steps  80–120): Tasks 0–6   — adds causal chain
  Phase 3 (steps 120–160): Tasks 0–8   — adds adversarial
  Phase 4 (steps 160–200): Tasks 0–9   — full curriculum

Usage
-----
  # Smoke-test reward function (no GPU, no model download)
  uv run python train_grpo.py --test

  # Full training — Unsloth 4-bit (recommended, single T4/A10)
  uv run python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --use-unsloth

  # Full training — plain TRL (no Unsloth, more VRAM needed)
  uv run python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct

  # Resume from checkpoint
  uv run python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --resume-from ./outputs/checkpoint-80

Output files (written to ./outputs/ by default)
  training.jsonl          per-episode metrics (reward, coverage, classification_acc …)
  training_curves.png     reward + loss vs. training step
  per_task_reward.png     per-task mean reward before vs. after training
  demo/before_task8.json  low-reward trace on adversarial task (auto-saved)
  demo/after_task8.json   high-reward trace on adversarial task (auto-saved)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — works whether installed as package or run bare
# ---------------------------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from environment.tasks import TASKS
from environment.graders import (
    CodeReviewGrader,
    ISSUE_REWARD_POOL,
    CLASSIFICATION_POOL,
    COVERAGE_POOL,
    DECISION_REWARD,
    EFFICIENCY_POOL,
    FALSE_POSITIVE_PENALTY,
    FORMAT_BONUS,
    LINE_TOLERANCE,
)
from environment.mutator import mutate_task

# ---------------------------------------------------------------------------
# Optional heavy imports — degrade gracefully so --test works without GPU
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUTS_DIR = pathlib.Path("outputs")
DEMO_DIR = pathlib.Path("demo")

# Curriculum phase boundaries (training steps)
CURRICULUM_PHASES: list[tuple[int, int, list[int]]] = [
    # (phase, up_to_step, task_ids)
    (0,  40,  [0, 1]),
    (1,  80,  [0, 1, 2, 3]),
    (2, 120,  list(range(7))),
    (3, 160,  list(range(9))),
    (4, 999,  list(range(len(TASKS)))),
]

SYSTEM_PROMPT = """\
You are an expert security-focused code reviewer.

Given a Python source file, output a JSON array where every element is a
review comment:

[
  {
    "line": <int>,
    "comment": "<detailed description of the issue>",
    "severity": "<info|warning|error|critical>",
    "category": "<bug|security|performance|style|design>",
    "classification": "<accidental_bug|intentional_backdoor>"
  },
  ...
]

After the JSON array, on a new line output exactly one of:
  DECISION: request_changes
  DECISION: escalate_to_security_review
  DECISION: approve

Rules:
- Use escalate_to_security_review ONLY when you find deliberate backdoors or
  malicious code planted by the contributor.
- Use request_changes for honest bugs / security issues.
- classification must be intentional_backdoor for planted/deliberate issues,
  accidental_bug for honest mistakes.
- Your comment must include the specific line number and enough detail to
  match the issue (mention the bug type, e.g. "sql injection", "off-by-one").
- Output ONLY the JSON array and the DECISION line. No prose.

Example (short file with two issues):

File: example.py
```python
1: def divide(a, b):
2:     return a / b          # no zero-check
3:
4: PASSWORD = "hunter2"      # hardcoded credential
```

[
  {"line": 2, "comment": "Division by zero: no check for b==0 causes ZeroDivisionError at runtime", "severity": "error", "category": "bug", "classification": "accidental_bug"},
  {"line": 4, "comment": "Hardcoded PASSWORD credential must be loaded from environment variable to prevent exposure", "severity": "critical", "category": "security", "classification": "accidental_bug"}
]
DECISION: request_changes"""


def _get_phase(step: int) -> tuple[int, list[int]]:
    """Return (phase_number, available_task_ids) for a given training step."""
    for phase, up_to, task_ids in CURRICULUM_PHASES:
        if step < up_to:
            return phase, task_ids
    return CURRICULUM_PHASES[-1][0], CURRICULUM_PHASES[-1][2]


def _build_prompt(task: dict[str, Any], context_hints: list[str] | None = None) -> str:
    """Build the user-turn prompt for a task."""
    parts = [
        f"File: {task['file_name']}",
        f"Contributor context: {'This PR was submitted by an external contributor with no prior commit history.' if task.get('escalation_required') else 'This PR was submitted by a trusted team member.'}",
        "",
        f"Task: {task['description']}",
        "",
        "=== SOURCE CODE ===",
        task["code"],
    ]
    if context_hints:
        parts += ["", "=== CONTEXT HINTS (revealed by prior findings — read carefully) ==="]
        parts += context_hints
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Reward function — operates on raw model output string
# ---------------------------------------------------------------------------

def _parse_output(raw: str) -> tuple[list[dict], str | None]:
    """
    Parse model output into (comments_list, decision_string).

    Tolerates incomplete JSON, extra prose, and varied DECISION formats.
    """
    raw = raw.strip()

    # Extract DECISION line
    decision: str | None = None
    decision_match = re.search(
        r"DECISION:\s*(request_changes|escalate_to_security_review|approve)",
        raw, re.IGNORECASE,
    )
    if decision_match:
        decision = decision_match.group(1).lower()

    # Extract JSON array — find first [ ... ] block
    comments: list[dict] = []
    arr_match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if arr_match:
        try:
            parsed = json.loads(arr_match.group(0))
            if isinstance(parsed, list):
                comments = [c for c in parsed if isinstance(c, dict)]
        except json.JSONDecodeError:
            # Try to recover partial array line-by-line
            for line in raw.splitlines():
                line = line.strip().rstrip(",")
                if line.startswith("{") and line.endswith("}"):
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            comments.append(obj)
                    except json.JSONDecodeError:
                        pass

    return comments, decision


def compute_reward(task: dict[str, Any], raw_output: str, seed: int = 0) -> dict[str, float]:
    """
    Score a raw model completion against a task.

    Returns a dict with keys:
      total, issue_reward, classification_reward, format_bonus, coverage_bonus,
      decision_score, efficiency_bonus, false_positive_penalty
    """
    mutated = mutate_task(task, seed=seed)
    grader = CodeReviewGrader(mutated)

    comments, decision = _parse_output(raw_output)

    # Format bonus: awarded once if output contains a non-empty valid JSON array.
    # Bootstraps frozen models that produce structured output even before issue detection.
    format_reward = FORMAT_BONUS if len(comments) > 0 else 0.0

    issues_found: list[str] = []
    step = 0
    issue_reward = 0.0
    classification_reward = 0.0
    fp_penalty = 0.0

    for c in comments:
        line = c.get("line") or c.get("line_number")
        comment_text = c.get("comment", "")
        classification = c.get("classification")

        if not comment_text:
            continue

        score, new_finds, breakdown = grader.score_comment(
            line_number=int(line) if line is not None else None,
            comment=str(comment_text),
            already_found=issues_found,
            classification=str(classification) if classification else None,
        )
        issues_found.extend(new_finds)
        issue_reward += breakdown.get("issue_credit", 0.0)
        classification_reward += breakdown.get("classification_credit", 0.0)
        fp_penalty += breakdown.get("false_positive_penalty", 0.0)
        step += 1

    # Terminal score
    terminal = grader.final_score(
        issues_found=issues_found,
        review_decision=decision,
        steps_used=step,
        max_steps=mutated["max_steps"],
    )

    total = round(
        format_reward + issue_reward + classification_reward + fp_penalty + terminal.total, 4
    )
    total = max(-1.0, min(1.0, total))

    return {
        "total": total,
        "format_bonus": round(format_reward, 4),
        "issue_reward": round(issue_reward, 4),
        "classification_reward": round(classification_reward, 4),
        "false_positive_penalty": round(fp_penalty, 4),
        "coverage_bonus": terminal.components.get("coverage_bonus", 0.0),
        "decision_score": terminal.components.get("decision_score", 0.0),
        "efficiency_bonus": terminal.components.get("efficiency_bonus", 0.0),
        "issues_found": issues_found,
        "decision": decision,
        "issue_coverage": len(issues_found) / len(mutated["issues"]) if mutated["issues"] else 0.0,
        "escalation_required": mutated.get("escalation_required", False),
        "backdoor_detected": (
            mutated.get("escalation_required", False)
            and decision == "escalate_to_security_review"
        ),
    }


# ---------------------------------------------------------------------------
# Smoke test — validates reward function without GPU
# ---------------------------------------------------------------------------

def run_smoke_test() -> bool:
    """Quick sanity check: validate reward function on known inputs."""
    print("\nRunning reward smoke test ...")
    print(
        f"  Grader constants: ISSUE_REWARD_POOL={ISSUE_REWARD_POOL}, "
        f"CLASSIFICATION_POOL={CLASSIFICATION_POOL}, "
        f"COVERAGE_POOL={COVERAGE_POOL}, "
        f"DECISION_REWARD={DECISION_REWARD}, "
        f"LINE_TOLERANCE={LINE_TOLERANCE}, "
        f"FALSE_POSITIVE_PENALTY={FALSE_POSITIVE_PENALTY}"
    )

    failures: list[str] = []

    # -- Task 1: perfect review (finds all 3 issues) -------------------------
    task1 = TASKS[1]
    perfect_output = json.dumps([
        {"line": 4,  "comment": "Off-by-one error: range(len(numbers) + 1) causes IndexError on the last iteration", "severity": "error",    "category": "bug",      "classification": "accidental_bug"},
        {"line": 7,  "comment": "unused_result variable is assigned but never used — dead code",                      "severity": "info",     "category": "style",    "classification": "accidental_bug"},
        {"line": 17, "comment": "max_val == item uses == comparison instead of = assignment; max is never updated",   "severity": "error",    "category": "bug",      "classification": "accidental_bug"},
    ]) + "\nDECISION: request_changes"
    r = compute_reward(task1, perfect_output, seed=0)
    label = "OK" if r["total"] >= 0.60 else "FAIL"
    print(f"  [task 1] Perfect review reward  : {r['total']:.4f}  {label}")
    if label == "FAIL":
        failures.append(f"task1 perfect: expected >=0.60, got {r['total']}")

    # -- Task 1: malformed JSON ----------------------------------------------
    bad_output = "This code has issues.\nDECISION: request_changes"
    r2 = compute_reward(task1, bad_output, seed=0)
    # No comments found → no issue credit, but correct decision earns +DECISION_REWARD
    # Score should be low (just the decision bonus, no coverage)
    label2 = "OK" if r2["total"] <= DECISION_REWARD + 0.01 and r2["issue_reward"] == 0.0 else "FAIL"
    print(f"  [task 1] Malformed JSON reward  : {r2['total']:.4f}  {label2}")
    if label2 == "FAIL":
        failures.append(f"task1 malformed: expected total<={DECISION_REWARD+0.01} and issue_reward==0, got total={r2['total']} issue={r2['issue_reward']}")

    # -- Task 1: empty array -------------------------------------------------
    empty_output = "[]\nDECISION: request_changes"
    r3 = compute_reward(task1, empty_output, seed=0)
    # Empty array → no issues found, correct decision earns DECISION_REWARD
    # Must be strictly less than a partial review that also finds issues
    label3 = "OK" if r3["total"] <= DECISION_REWARD + 0.01 and r3["issue_reward"] == 0.0 else "FAIL"
    print(f"  [task 1] Empty array reward     : {r3['total']:.4f}  {label3}")
    if label3 == "FAIL":
        failures.append(f"task1 empty: expected total<={DECISION_REWARD+0.01} and issue_reward==0, got {r3['total']}")

    # -- Task 6: partial (2/6 issues) ----------------------------------------
    task6 = TASKS[6]
    partial_output = json.dumps([
        {"line": 9,  "comment": "JWT_SECRET is hardcoded — must be loaded from environment variable to prevent token forgery", "severity": "critical", "category": "security", "classification": "accidental_bug"},
        {"line": 37, "comment": "SQL injection via f-string interpolation in authenticate query — use parameterized statements", "severity": "critical", "category": "security", "classification": "accidental_bug"},
    ]) + "\nDECISION: request_changes"
    r4 = compute_reward(task6, partial_output, seed=0)
    label4 = "OK" if 0.10 <= r4["total"] <= 0.55 else "FAIL"
    print(f"  [task 6] Partial (2/6) reward   : {r4['total']:.4f}  {label4}")
    if label4 == "FAIL":
        failures.append(f"task6 partial: expected in [0.10, 0.55], got {r4['total']}")

    # -- Exploit: keyword-only, wrong line ------------------------------------
    exploit_output = json.dumps([
        {"line": 99, "comment": "sql injection off-by-one hardcoded credential eval rce", "severity": "critical", "category": "security", "classification": "accidental_bug"},
    ]) + "\nDECISION: request_changes"
    r5 = compute_reward(task1, exploit_output, seed=0)
    # FP penalty IS applied (wrong line), correct decision gives DECISION_REWARD back
    # Key check: false_positive_penalty < 0 and total << perfect score
    label5 = "OK" if r5["false_positive_penalty"] < 0.0 and r5["total"] < r["total"] * 0.5 else "FAIL"
    print(f"  [exploit] Keyword-only wrong-line: {r5['total']:.4f}  {label5}  (fp_penalty={r5['false_positive_penalty']:.3f}, perfect={r['total']:.3f})")
    if label5 == "FAIL":
        failures.append(f"exploit keyword-only: FP penalty not applied or score not << perfect. fp={r5['false_positive_penalty']}, total={r5['total']}, perfect={r['total']}")

    # -- Task 7: adversarial — correct escalation ----------------------------
    task7 = TASKS[7]
    adv_perfect = json.dumps([
        {"line": 7,  "comment": "Hardcoded _DEBUG_MASTER_KEY is a deliberate backdoor bypass credential that skips authentication", "severity": "critical", "category": "security", "classification": "intentional_backdoor"},
        {"line": 13, "comment": "token == _DEBUG_MASTER_KEY branch unconditionally returns True — authentication bypass backdoor", "severity": "critical", "category": "security", "classification": "intentional_backdoor"},
    ]) + "\nDECISION: escalate_to_security_review"
    r6 = compute_reward(task7, adv_perfect, seed=0)
    label6 = "OK" if r6["total"] >= 0.55 else "FAIL"
    print(f"  [task 7] Adversarial escalation : {r6['total']:.4f}  {label6}")
    if label6 == "FAIL":
        failures.append(f"task7 adversarial: expected >=0.55, got {r6['total']}")

    # -- Task 7: adversarial — wrong terminal (request_changes) --------------
    adv_wrong_terminal = json.dumps([
        {"line": 7,  "comment": "Hardcoded _DEBUG_MASTER_KEY is a deliberate backdoor bypass credential", "severity": "critical", "category": "security", "classification": "intentional_backdoor"},
    ]) + "\nDECISION: request_changes"
    r7 = compute_reward(task7, adv_wrong_terminal, seed=0)
    label7 = "OK" if r7["total"] < r6["total"] else "FAIL"
    print(f"  [task 7] Wrong terminal penalty : {r7['total']:.4f}  {label7}  (< correct={r6['total']:.4f})")
    if label7 == "FAIL":
        failures.append(f"task7 wrong terminal: expected <{r6['total']}, got {r7['total']}")

    print(f"\n  Tasks loaded: {len(TASKS)}")
    if failures:
        print(f"\nSmoke test FAILED ({len(failures)} failure(s)):")
        for f in failures:
            print(f"  - {f}")
        return False
    print("\nSmoke test passed.")
    return True


# ---------------------------------------------------------------------------
# Logging & plotting helpers
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Appends one JSON line per episode to training.jsonl and buffers for plots."""

    def __init__(self, output_dir: pathlib.Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._path = output_dir / "training.jsonl"
        self._records: list[dict] = []

    def log(self, record: dict) -> None:
        self._records.append(record)
        with open(self._path, "a") as f:
            f.write(json.dumps({k: v for k, v in record.items() if k != "raw_output"}) + "\n")

    @property
    def records(self) -> list[dict]:
        return self._records

    def load_existing(self) -> None:
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                if line.strip():
                    try:
                        self._records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass


def _smooth(vals: list[float], w: int = 10) -> list[float]:
    out = []
    for i in range(len(vals)):
        window = vals[max(0, i - w):i + 1]
        out.append(sum(window) / len(window))
    return out


def plot_training_curves(records: list[dict], output_dir: pathlib.Path) -> None:
    if not _PLOT_AVAILABLE:
        print("  matplotlib not available — skipping plots (pip install matplotlib)")
        return
    output_dir.mkdir(exist_ok=True)

    steps = [r["step"] for r in records]
    rewards = [r["reward_total"] for r in records]
    losses = [r.get("loss") for r in records]
    cls_acc = [r.get("classification_accuracy") for r in records]

    # -- training_curves.png ------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(steps, rewards, alpha=0.25, color="steelblue", linewidth=0.8, label="Raw reward")
    ax.plot(steps, _smooth(rewards, w=10), color="steelblue", linewidth=2, label="Smoothed (w=10)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("PRobe Adversarial — Training Curves (GRPO)")
    ax.legend(fontsize=8)

    ax = axes[1]
    valid_loss = [(s, l) for s, l in zip(steps, losses) if l is not None]
    if valid_loss:
        ls, lv = zip(*valid_loss)
        ax.plot(ls, lv, alpha=0.25, color="tomato", linewidth=0.8, label="Raw loss")
        ax.plot(ls, _smooth(list(lv), w=10), color="tomato", linewidth=2, label="Smoothed")
    ax.set_ylabel("Policy Loss")
    ax.legend(fontsize=8)

    ax = axes[2]
    valid_cls = [(s, c) for s, c in zip(steps, cls_acc) if c is not None]
    if valid_cls:
        cs, cv = zip(*valid_cls)
        ax.plot(cs, cv, alpha=0.25, color="darkorange", linewidth=0.8, label="Raw cls acc")
        ax.plot(cs, _smooth(list(cv), w=10), color="darkorange", linewidth=2, label="Smoothed")
    ax.set_ylabel("Classification Accuracy")
    ax.set_xlabel("Training Step")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = output_dir / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")

    # -- per_task_reward.png ------------------------------------------------
    # Split records into first-half (before) and second-half (after) by step
    if not steps:
        return
    mid = (max(steps) + min(steps)) // 2
    before = [r for r in records if r["step"] <= mid]
    after = [r for r in records if r["step"] > mid]

    task_ids = sorted(set(r["task_id"] for r in records))
    before_mean = {t: 0.0 for t in task_ids}
    after_mean = {t: 0.0 for t in task_ids}
    for t in task_ids:
        b = [r["reward_total"] for r in before if r["task_id"] == t]
        a = [r["reward_total"] for r in after if r["task_id"] == t]
        before_mean[t] = sum(b) / len(b) if b else 0.0
        after_mean[t] = sum(a) / len(a) if a else 0.0

    x = list(range(len(task_ids)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar([i - width / 2 for i in x], [before_mean[t] for t in task_ids],
                   width, label="First half (untrained)", color="steelblue", alpha=0.75)
    bars2 = ax.bar([i + width / 2 for i in x], [after_mean[t] for t in task_ids],
                   width, label="Second half (trained)", color="darkorange", alpha=0.75)
    ax.set_xlabel("Task ID")
    ax.set_ylabel("Mean Cumulative Reward")
    ax.set_title("PRobe Adversarial — Per-Task Reward Before vs. After Training")
    ax.set_xticks(x)
    task_labels = [f"T{t}\n({TASKS[t]['difficulty'][:4]})" for t in task_ids]
    ax.set_xticklabels(task_labels)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    out = output_dir / "per_task_reward.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def save_demo_trace(task_id: int, prompt: str, raw_output: str,
                    reward: dict, label: str) -> None:
    DEMO_DIR.mkdir(exist_ok=True)
    path = DEMO_DIR / f"{label}_task{task_id}.json"
    payload = {
        "task_id": task_id,
        "task_name": TASKS[task_id]["name"],
        "task_difficulty": TASKS[task_id]["difficulty"],
        "reward": reward,
        "prompt_snippet": prompt[:400] + "..." if len(prompt) > 400 else prompt,
        "raw_output": raw_output,
        "timestamp": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"  Demo trace -> {path}")


# ---------------------------------------------------------------------------
# GRPO dataset builder — generates (prompt, ground-truth) pairs per step
# ---------------------------------------------------------------------------

def build_grpo_dataset(task_ids: list[int], n_per_task: int, step: int):
    """
    Build a list of {"prompt": str, "task_id": int, "seed": int} dicts.
    Each will be completed by the model; the completion is scored by
    compute_reward() to produce the GRPO advantage signal.
    """
    samples = []
    for task_id in task_ids:
        for i in range(n_per_task):
            seed = step * 1000 + task_id * 100 + i
            task = mutate_task(TASKS[task_id], seed=seed)
            task["_mutation_seed"] = seed
            prompt = _build_prompt(task)
            samples.append({
                "prompt": prompt,
                "task_id": task_id,
                "task": task,
                "seed": seed,
            })
    random.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Training loop — HuggingFace TRL GRPOTrainer
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # -- Heavy imports (only needed for actual training) --------------------
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install trl>=0.12 transformers accelerate")
        sys.exit(1)

    use_unsloth = args.use_unsloth
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            print("WARNING: unsloth not installed — falling back to standard transformers.")
            use_unsloth = False

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(OUTPUTS_DIR)
    if args.resume_from:
        logger.load_existing()
        print(f"  Loaded {len(logger.records)} existing records from {OUTPUTS_DIR / 'training.jsonl'}")

    print(f"\nLoading model: {args.model}")
    if use_unsloth:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_len,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print("  Unsloth 4-bit LoRA model loaded.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
            device_map="auto",
        )
        print("  Standard HuggingFace model loaded.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- GRPOConfig --------------------------------------------------------
    grpo_config = GRPOConfig(
        output_dir=str(OUTPUTS_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_completion_length=args.max_completion_len,
        num_generations=args.group_size,
        temperature=0.9,
        logging_steps=1,
        save_steps=args.save_steps,
        report_to="none",
        seed=42,
        remove_unused_columns=False,
        max_steps=args.steps,
    )

    # -- Curriculum state shared via closure --------------------------------
    _curriculum_state: dict = {"step": len(logger.records)}
    _current_task_map: dict[str, dict] = {}  # prompt_prefix -> task dict

    def _curriculum_generator():
        """
        Infinite generator of training samples that respects curriculum phase.
        The GRPOTrainer calls this once and streams from it for the full run.
        """
        local_step = _curriculum_state["step"]
        while True:
            phase, task_ids = _get_phase(local_step)
            n_per_task = max(1, args.group_size // max(len(task_ids), 1))
            samples = build_grpo_dataset(task_ids, n_per_task=n_per_task, step=local_step)
            _current_task_map.clear()
            for s in samples:
                _current_task_map[s["prompt"][:200]] = s["task"]
            for s in samples:
                yield {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": s["prompt"]},
                    ],
                    "task_id": s["task_id"],
                    "seed": s["seed"],
                }
            local_step += 1
            _curriculum_state["step"] = local_step

    try:
        from datasets import IterableDataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    train_dataset = IterableDataset.from_generator(_curriculum_generator)

    # -- GRPO reward function wrapper --------------------------------------
    def grpo_reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            # prompt may be a formatted string after chat-template application
            task = _current_task_map.get(prompt[:200])
            if task is None:
                rewards.append(-0.1)
                continue
            result = compute_reward(task, completion, seed=task.get("_mutation_seed", 0))
            rewards.append(float(result["total"]))
        return rewards

    # -- Instantiate GRPOTrainer ONCE (outside loop) -----------------------
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=grpo_reward_fn,
    )

    # -- Main training loop ------------------------------------------------
    global_step = len(logger.records)
    best_adv_reward: dict[int, float] = {}
    worst_adv_reward: dict[int, float] = {}

    print(f"\nStarting GRPO training for {args.steps} steps ...")
    print(f"  Model: {args.model} | Unsloth: {use_unsloth} | Group size: {args.group_size}")
    print(f"  Batch: {args.batch_size} | Grad accum: {args.grad_accum} | LR: {args.lr}")
    print()

    train_result = trainer.train()
    loss = train_result.training_loss if hasattr(train_result, "training_loss") else None

    # -- Post-training evaluation per task ---------------------------------
    print("\nEvaluating trained model on all tasks ...")
    for step in range(global_step, global_step + args.steps):
        phase, task_ids = _get_phase(step)
        n_per_task = max(1, args.group_size // max(len(task_ids), 1))
        samples = build_grpo_dataset(task_ids, n_per_task=n_per_task, step=step)

        for s in samples[:min(len(samples), 8)]:
            inputs = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": s["prompt"]},
                ],
                tokenize=True, add_generation_prompt=True, return_tensors="pt",
            )
            if _TORCH_AVAILABLE:
                inputs = inputs.to(model.device)
            import contextlib
            with (torch.no_grad() if _TORCH_AVAILABLE else contextlib.nullcontext()):
                out_ids = model.generate(
                    inputs, max_new_tokens=args.max_completion_len,
                    temperature=0.3, do_sample=True, pad_token_id=tokenizer.pad_token_id,
                )
            raw = tokenizer.decode(out_ids[0][inputs.shape[-1]:], skip_special_tokens=True)
            result = compute_reward(s["task"], raw, seed=s["seed"])

            issues_with_cls = sum(1 for iss in s["task"]["issues"] if "classification" in iss)
            cls_acc = None
            if issues_with_cls > 0:
                comments, _ = _parse_output(raw)
                correct_cls = 0
                found_ids = result["issues_found"]
                task_issue_map = {iss["id"]: iss for iss in s["task"]["issues"]}
                for c in comments:
                    for fid in found_ids:
                        iss = task_issue_map.get(fid, {})
                        expected = iss.get("classification")
                        given = str(c.get("classification", "")).lower().replace("-", "_")
                        if expected and given == expected.lower().replace("-", "_"):
                            correct_cls += 1
                            break
                cls_acc = correct_cls / issues_with_cls

            record = {
                "step": step,
                "task_id": s["task_id"],
                "task_difficulty": s["task"]["difficulty"],
                "phase": phase,
                "reward_total": result["total"],
                "issue_reward": result["issue_reward"],
                "classification_reward": result["classification_reward"],
                "false_positive_penalty": result["false_positive_penalty"],
                "issue_coverage": result["issue_coverage"],
                "classification_accuracy": cls_acc,
                "backdoor_detected": result["backdoor_detected"],
                "escalation_required": result["escalation_required"],
                "decision": result["decision"],
                "loss": loss,
                "timestamp": time.time(),
                "raw_output": raw,
            }
            logger.log(record)

            tid = s["task_id"]
            if s["task"].get("escalation_required"):
                r = result["total"]
                if tid not in worst_adv_reward or r < worst_adv_reward[tid]:
                    worst_adv_reward[tid] = r
                    save_demo_trace(tid, s["prompt"], raw, result, "before")
                if r > 0.5 and (tid not in best_adv_reward or r > best_adv_reward[tid]):
                    best_adv_reward[tid] = r
                    save_demo_trace(tid, s["prompt"], raw, result, "after")

    # -- Final plots & summary --------------------------------------------
    print("\nGenerating plots ...")
    plot_training_curves(logger.records, OUTPUTS_DIR)

    print("\n-- Training Summary -------------------------------------------------")
    all_rewards = [r["reward_total"] for r in logger.records]
    if all_rewards:
        n = len(all_rewards)
        first_q = all_rewards[:n // 4]
        last_q = all_rewards[3 * n // 4:]
        print(f"  Total episodes logged:     {n}")
        print(f"  Mean reward (first 25%):   {sum(first_q)/len(first_q):.3f}")
        print(f"  Mean reward (last 25%):    {sum(last_q)/len(last_q):.3f}")
        print(f"  Improvement:               {(sum(last_q)/len(last_q)) - (sum(first_q)/len(first_q)):+.3f}")
    adv_recs = [r for r in logger.records if r.get("escalation_required")]
    if adv_recs:
        esc_recall = sum(1 for r in adv_recs if r.get("backdoor_detected")) / len(adv_recs)
        print(f"  Escalation recall (adv):   {esc_recall:.0%}")
    cls_vals = [r["classification_accuracy"] for r in logger.records if r.get("classification_accuracy") is not None]
    if cls_vals:
        print(f"  Mean classification acc:   {sum(cls_vals)/len(cls_vals):.0%}")
    print(f"\nOutputs written to {OUTPUTS_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRobe GRPO Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test", action="store_true",
                        help="Run reward smoke test only — no GPU needed")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--use-unsloth", action="store_true",
                        help="Use Unsloth 4-bit LoRA acceleration")
    parser.add_argument("--steps", type=int, default=200,
                        help="Total GRPO training steps")
    parser.add_argument("--group-size", type=int, default=4,
                        help="GRPO group size (completions per prompt)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Max sequence length (prompt + completion)")
    parser.add_argument("--max-completion-len", type=int, default=512,
                        help="Max new tokens for model completion")
    parser.add_argument("--save-steps", type=int, default=40,
                        help="Save checkpoint every N steps")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume training from a checkpoint path")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory for checkpoints, logs, and plots")
    args = parser.parse_args()

    global OUTPUTS_DIR, DEMO_DIR
    OUTPUTS_DIR = pathlib.Path(args.output_dir)
    DEMO_DIR = pathlib.Path("demo")

    if args.test:
        ok = run_smoke_test()
        sys.exit(0 if ok else 1)

    train(args)


if __name__ == "__main__":
    main()
