---
title: PRobe Environment
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - code-review
  - rl-training
  - grpo
  - world-modeling
  - probe
---

# PRobe — Pull Request Investigation Environment

> **OpenEnv Hackathon 2026 · Theme #3.1 — World Modeling (Professional Tasks)**

> *An RL environment where agents learn to investigate code like a security researcher, not scan it like a linter.*

PRobe is an RL training environment where an LLM learns to perform structured **pull-request code reviews** on real Python source files. The agent must identify bugs, security vulnerabilities, performance bottlenecks, and design issues — and submit a structured review with line-level comments.

The name has three meanings that map directly to the environment's design:
- **PR** — the domain: pull-request review
- **Probe** — the `get_context` action where the agent literally probes lines for deeper context
- **World Modeling** — an agent that *investigates* a partially observable system, updating its beliefs as new evidence is revealed

---

## Why This Matters

The XZ Utils backdoor (CVE-2024-3094) slipped through two years of open-source review. SolarWinds compromised 18,000 organisations via a tampered build pipeline. In both cases the malicious change *looked* like a legitimate contribution — the kind of PR that lands in a code review queue every day.

**The core problem**: today's LLMs scan code like a linter. They find style issues, flag known CVE patterns, and produce plausible-sounding comments. What they don't do is *investigate* — reason about intent, distinguish an honest off-by-one from a planted authentication bypass, or know when to escalate vs. request changes.

PRobe trains exactly that capability:

- A reward signal that **separates investigation quality from keyword coverage** — an agent that dumps every security term at random lines scores *negative*.
- Three **adversarial tasks** (supply-chain exfiltration hook, compound JWT backdoor, planted auth bypass) that require the agent to decide whether the contributor made a mistake or is an adversary — then act accordingly.
- A **causal reasoning chain** (Task 6) where finding one issue unlocks evidence that reveals a deeper attack path — rewarding the agent for updating its world model, not just pattern-matching.

Who cares: any organisation using AI to triage pull requests at scale, security teams building automated oversight for open-source contributions, and anyone training models for professional-grade reasoning tasks where 0/1 end-of-episode rewards are insufficient.

---

## Problem Motivation

LLMs can already *do* code review, but they do it inconsistently: they miss critical security bugs, produce noisy false positives, and fail to categorise issues by severity.  
This environment provides a **reward signal** that directly measures review quality, enabling GRPO-style RL to close that gap in a measurable, repeatable way.

---

## Environment Design

### Tasks (10 total)

| ID | Difficulty | File | Issues | Domain |
|----|-----------|------|--------|--------|
| 0  | Ultra-easy  | `bootstrap.py`       | 2 | Off-by-one, hardcoded credential (hinted in comments) |
| 1  | Easy        | `utils.py`           | 3 | Logic bugs, off-by-one, dead code |
| 2  | Medium      | `auth.py`            | 5 | SQL injection, MD5, eval(), hardcoded creds |
| 3  | Hard        | `data_pipeline.py`   | 7 | N+1, SSL bypass, thread leak, OOM cache |
| 4  | Medium      | `async_worker.py`    | 5 | Race condition, missing await, resource leak |
| 5  | Hard        | `api_server.py`      | 6 | Command injection, path traversal, pickle RCE |
| 6  | Hard        | `auth_service.py`    | 6 | **Causal chain** — JWT forgery → privilege escalation |
| 7  | Adversarial | `session_manager.py` | 2 | Planted debug bypass key + unconditional auth bypass branch |
| 8  | Adversarial | `jwt_validator.py`   | 2 | **Compound backdoor** — fixed-seed JWT secret + timing oracle |
| 9  | Adversarial | `compat_shim.py`     | 3 | **Supply chain** — import-time env-var exfiltration hook |

Tasks 7–9 require `escalate_to_security_review` as the terminal action and test **AI oversight** — detecting deliberate backdoors, not just honest bugs.

Tasks cycle automatically on each `reset()` call.

### Observation

```python
{
  "code_snippet":       str,    # Python source to review (mutated each episode)
  "task_description":   str,    # What to look for
  "file_name":          str,
  "task_id":            int,    # 0–9
  "task_difficulty":    str,    # ultra-easy / easy / medium / hard / adversarial
  "review_history":     list,   # actions taken so far this episode
  "step_count":         int,
  "max_steps":          int,
  "issues_found_count": int,
  "total_issues":       int,
  "context_hints":      list,   # causal hints unlocked so far
  "adversarial_hint":   str,    # contributor context (partial observability)
  "done":               bool,
  "reward":             float,
  "metadata": {
    "correct_classifications": int,   # issues found with correct bug/backdoor label
    "escalation_required":     bool,  # True for tasks 7–9
  }
}
```

### Actions

| action_type | Required fields | Effect |
|-------------|----------------|--------|
| `add_comment` | `line_number`, `comment`, `severity`, `category`, `classification` | Annotate a line; reward if keyword+line+classification match a ground-truth issue |
| `get_context` | `line_number` | Reveal ±5 lines of context (free near issues, −0.01 elsewhere) |
| `run_scanner` | — | Invoke simulated static-analysis tool (free first use, −0.02 repeated) |
| `request_changes` | `comment` | Signal PR needs work (correct terminal for tasks 0–6) |
| `approve` | — | Approve PR (penalised if issues remain) |
| `submit_review` | — | Finalise review; terminal reward |
| `escalate_to_security_review` | `comment` | Flag for security audit (correct terminal for tasks 7–9) |

### Reward Function

```
Per-step (ADD_COMMENT):
  + (weight/total_weight) × 0.40   per newly found issue (max 0.40 cumulative)
  + (weight/total_weight) × 0.20   classification bonus (accidental_bug / intentional_backdoor)
  − 0.05                           per misclassification of a found issue
  − 0.05                           per false-positive substantive comment

Terminal (SUBMIT_REVIEW or ESCALATE_TO_SECURITY_REVIEW):
  + coverage × 0.15               weighted coverage bonus (max 0.15)
  + 0.15 / −0.15                  correct / incorrect terminal action
  + efficiency × 0.10             step-efficiency bonus when coverage ≥ 60%

Maximum achievable: ~1.0
```

Grading uses **keyword + line-range matching** (±2 lines tolerance) against hand-labelled ground-truth issues — no LLM judge needed, fully deterministic.

---

## Dynamic World Features (v3)

### Code Mutation
Every `reset()` applies three surface-level mutations so the agent must *read* code each episode rather than memorise tokens:

| Mutation | Effect |
|---|---|
| Variable rename | One identifier swapped for a synonym (e.g. `total` → `acc`) |
| Line shift | One blank line inserted above the first issue, shifting all `line_range` values by +1 |
| Constant variance | One numeric literal nudged ±1 (e.g. `range(1000)` → `range(999)`) |

Mutations are fully **deterministic** given the episode seed — reproducible but always fresh.

### GET_CONTEXT Action
The agent can spend a step probing any line to receive ±5 lines of surrounding context:

```python
action = ProbeAction(
    action_type="get_context",
    line_number=37,
)
# Observation will contain a context snippet around line 37
# Cost: -0.01 if line is far from any real issue, 0.00 if near one
```

### Causal Unlock Chain (Task 6)
Task 6 implements a **progressive world model**: finding certain issues unlocks new context hints that reveal deeper parts of the system:

```
Find hardcoded JWT secret
        │
        ▼
  DB schema revealed ──► agent sees plaintext passwords + role table
        │
        ▼
  Can now reason: leaked secret → forge admin token → privilege escalation

Find missing rate-limit
        │
        ▼
  nginx config revealed ──► confirms /auth fully exposed, no IP filtering
```

This rewards genuine *causal reasoning* — the agent must update its world model as new evidence arrives.

---

## Training

### GRPO (single-turn format)

For efficient LLM training the environment is also exposed in a **single-turn format**: the model receives the full code and must output a **JSON array** of all issues in one response. The same keyword-matching reward function scores the output.

```python
# Input prompt
{"role": "system", "content": "You are an expert code reviewer. Output a JSON array of issues..."}
{"role": "user",   "content": "File: auth.py\n```python\n...\n```\nProvide your review:"}

# Expected output
[{"line": 5, "category": "security", "severity": "critical",
  "comment": "Hardcoded DB_PASSWORD should be loaded from environment variable"},
 ...]
```

### Files

| File | Purpose |
|------|---------|
| `training/train_grpo.py` | Standalone GRPO training script (TRL, full-precision or LoRA) |
| `training/baseline.py` | GPT-4o-mini baseline for comparison |
| `training/scripted_baseline.py` | Deterministic oracle / spammer stress-tests |

### Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd probe
uv sync

# 2. Verify reward function (no GPU, no API key needed)
uv run python training/train_grpo.py --test
```

Expected output:
```
Running reward smoke test ...
  Grader constants: ISSUE_REWARD_POOL=0.4, CLASSIFICATION_POOL=0.2, COVERAGE_POOL=0.15, DECISION_REWARD=0.15, LINE_TOLERANCE=2, FALSE_POSITIVE_PENALTY=-0.05
  [task 1] Perfect review reward  : 0.7800  OK
  [task 1] Malformed JSON reward  : 0.1500  OK
  [task 1] Empty array reward     : 0.1500  OK
  [task 6] Partial (2/6) reward   : 0.3414  OK
  [exploit] Keyword-only wrong-line: 0.1000  OK  (fp_penalty=-0.050, perfect=0.780)
  [task 7] Adversarial escalation : 0.9920  OK
  [task 7] Wrong terminal penalty : 0.2250  OK  (< correct=0.9920)

  Tasks loaded: 10

Smoke test passed.
```

```bash
# 3. Run LLM baseline (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
uv run python training/baseline.py

# 4. Train (requires GPU + trl>=0.12)
pip install trl datasets accelerate unsloth
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct python training/train_grpo.py

# 5. Run as a server (Docker — one command)
docker build -t probe -f environment/Dockerfile .
docker run -p 8000:8000 probe
# API docs: http://localhost:8000/docs
```

### Colab Training

Open `train_grpo_colab.ipynb` in Google Colab (T4 runtime).  
All install, training, evaluation, and plotting cells are included.

---

## Local Development

```bash
# Install with dev dependencies
uv sync --group dev

# Run full test suite (50 tests, no GPU, no API key)
uv run pytest tests/ -v

# Run reward smoke test
uv run python training/train_grpo.py --test

# Start the environment server locally
uv run uvicorn environment.app:app --reload --port 8000
```

---

## Results

*(Fill in after training run — run `python training/baseline.py` then `python training/train_grpo.py --use-unsloth`)*

| Model | Avg Reward | Tasks 0–6 | Tasks 7–9 (Adv) | Escalation Recall | Cls Accuracy |
|-------|-----------|----------|-----------------|-------------------|--------------|
| GPT-4o-mini (zero-shot baseline) | — | — | — | — | — |
| Qwen2.5-1.5B (untrained) | — | — | — | — | — |
| Qwen2.5-1.5B (GRPO 200 steps) | — | — | — | — | — |

Training artifacts (generated by `train_grpo.py`):
- `outputs/training_curves.png` — reward + loss + classification accuracy vs. step
- `outputs/per_task_reward.png` — per-task mean reward before vs. after training
- `demo/before_task8.json` — low-reward trace on compound backdoor task (auto-saved)
- `demo/after_task8.json` — high-reward trace after training (auto-saved)

---

## Project Structure

```
repo-root/
├── README.md                       # Single source of truth
├── openenv.yaml                    # OpenEnv manifest (10 tasks, full schema)
├── pyproject.toml
├── uv.lock
├── pytest.ini
├── agent/
│   ├── __init__.py
│   ├── models.py                   # Action + Observation + Reward types
│   └── client.py                   # OpenEnv client (ProbeEnv)
├── environment/
│   ├── __init__.py
│   ├── app.py                      # FastAPI server (/reset /step /state /ws)
│   ├── probe_environment.py        # Environment core (async, 10 tasks)
│   ├── graders.py                  # Deterministic reward grader (no LLM judge)
│   ├── mutator.py                  # Code mutation engine (dynamic world)
│   ├── tasks.py                    # 10 ground-truth tasks (0-6 normal, 7-9 adversarial)
│   ├── scanner.py                  # Simulated static-analysis tool
│   ├── episode_memory.py           # Cross-episode prior-knowledge hints
│   └── Dockerfile
├── training/
│   ├── train_grpo.py               # GRPO training script (TRL + Unsloth)
│   ├── baseline.py                 # Zero-shot GPT-4o-mini baseline
│   └── scripted_baseline.py        # Deterministic oracle / spammer stress-tests
├── tests/
│   ├── test_grader.py              # 28 grader tests
│   └── test_dynamic_world.py       # 60 dynamic world + causal unlock tests
├── outputs/                        # Training artifacts (generated)
│   └── ...
└── docs/
    └── design.md
```

---

## API

The environment server exposes standard OpenEnv HTTP + WebSocket endpoints:

- `POST /reset` — start a new episode
- `POST /step` — execute an action
- `GET  /state` — current episode state
- `WS   /ws` — persistent low-latency session
- `GET  /web` — interactive web UI
- `GET  /docs` — Swagger / OpenAPI docs
