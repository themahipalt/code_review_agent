---
title: "Can an AI Actually *Investigate* Code — or Just Scan It?"
thumbnail: /blog/assets/probe/thumbnail.png
authors:
  - user: <!-- FILL: your-hf-username -->
---

# Can an AI Actually *Investigate* Code — or Just Scan It?

> **PRobe** is an RL training environment that teaches language models to do
> what no linter can: read a pull request, pinpoint deliberate backdoors,
> and decide when to escalate to a security team — all without an LLM judge.

<!-- FILL: embed a GIF or screenshot of the live dashboard here -->
<!-- ![PRobe dashboard](assets/probe/dashboard.gif) -->

---

## TL;DR

We built **PRobe** — a reinforcement-learning environment where an AI agent learns to review Python code the way a senior security engineer would.
The agent reads a pull request, flags bugs line-by-line, decides whether each flaw is an honest mistake or a deliberate attack, and knows when to escalate.
Our fully deterministic reward function requires no LLM judge: keyword spam scores *negative*, careful reading scores close to `+1.0`.
A scripted "perfect oracle" agent scores **+0.778** on average; a random agent scores **−0.260** — the gap the trained model has to close.

---

## The Problem (Why This Environment?)

Imagine you are the only security engineer at a fast-moving startup.
On Monday morning you open GitHub and find 47 open pull requests.
Most are routine changes — a new endpoint, a refactored helper, a dependency bump.
But one of them hides a backdoor: a two-line change that lets any request bypass authentication if a specific hidden header is present.
It looks completely legitimate. It passed CI. It was submitted by someone with a plausible GitHub profile.

This is not a hypothetical.
The **XZ Utils backdoor** (CVE-2024-3094) hid in plain sight for two years of open-source review.
The **SolarWinds breach** compromised 18,000 organisations through a tampered build pipeline that looked like a routine update.
In both cases the malicious change *looked* like a normal contribution.

Today's AI coding assistants are great at *generating* code.
They are much weaker at *evaluating* it — reasoning about intent, distinguishing an honest off-by-one from a planted authentication bypass, or knowing when the right answer is "stop and call the security team."
Reward signals for code generation are everywhere; reward signals for critical code *evaluation* barely exist.

PRobe is our attempt to close that gap.

---

## Meet the Environment

### Plain English First

The agent is handed a Python source file and a brief: *"Review this module. Flag every bug. Decide whether to request changes or escalate to security."*
On every new episode, the code surface changes slightly — variable names are renamed, line numbers shift, constants are tweaked — so the agent cannot memorise answers.
It has to actually read.

The agent can:
- **Leave comments** on specific lines (like a human reviewer adding inline notes)
- **Probe for more context** around a suspicious region
- **Run a simulated scanner** (which, like real tools, misses ~30 % of issues and occasionally cries wolf)
- **Submit a verdict**: request changes, approve, or escalate to a security team

At the end of each episode the reward function totals up how well the agent read the code.

### Observation → Agent → Action → Reward

```
┌──────────────────────────────────────────────────────┐
│  OBSERVATION (what the agent sees each step)         │
│  • Mutated Python source code                        │
│  • Review instructions + file name                   │
│  • History of actions taken so far                   │
│  • Steps remaining, issues found so far              │
│  • Causal hints (unlocked by finding key issues)     │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
           ┌───────────────┐
           │  LLM  Agent   │
           └───────┬───────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  ACTION (one per step)                               │
│  add_comment │ get_context │ run_scanner             │
│  request_changes │ approve │ escalate_to_security    │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  REWARD  (deterministic, no LLM judge)               │
│  +credit for finding real issues at correct lines    │
│  −penalty for spam, false positives, wrong decision  │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
           Next Observation
```

### Difficulty Tiers (10 Tasks Total)

| Tier | Example Task | What Makes It Hard |
|---|---|---|
| **Ultra-Easy** (T0) | Off-by-one + hardcoded credential, *hinted in the comments above each bug* | Nothing — it's a bootstrapper so GRPO sees positive reward from step one |
| **Easy** (T1) | Three logic bugs in a utility module | No hints; agent must read carefully |
| **Medium** (T2) | SQL injection, MD5 password hash, `eval()` call, hardcoded DB creds | Wider code surface; 5 issues spread across the file |
| **Hard** (T6) | JWT forgery → privilege escalation *causal chain* | Finding the first issue unlocks a new hint that reveals the second |
| **Adversarial** (T9) | Import-hook that silently exfiltrates env-vars to an attacker's domain | Agent must classify it as *intentional* AND escalate — `request_changes` is wrong even if it finds the bug |

---

## Designing the Reward Function

A reward function is the score the agent gets for each action — it's how the environment teaches the model what "good" means.
We spent significant time on ours, because the obvious designs are easy to game.

### What "Success" Looks Like

A perfect episode: the agent reads the code, annotates every real issue at the exact line with a substantive, keyword-bearing comment, correctly labels each as an accidental bug or a deliberate backdoor, and submits the right verdict with steps to spare.
Score approaches **+1.0**.

A failing episode: the agent dumps every security buzzword ("SQL injection, buffer overflow, RCE, XSS…") onto random lines, never co-locates a keyword with an actual issue, and submits the wrong verdict.
Score approaches **−1.0**.

### The Three-Gate Verifier

A comment earns issue credit only when **all three** conditions hold at once:

1. **Keyword hit** — at least one issue-specific keyword appears in the comment text
2. **Line hit** — the line number is within ±2 of the declared issue range
3. **Substantive** — the comment is longer than 15 characters

This closes three common reward-hacking paths:

| Exploit Attempt | Gate It Fails |
|---|---|
| Dump all security keywords at line 9999 | Line hit |
| Comment on the right line with "bad" | Substantive |
| Write a long, relevant comment on line 1 for every task | Keyword hit |

### Reward Components

| Component | Weight | When It Triggers |
|---|---|---|
| Issue credit | up to **+0.40** | Comment passes all three gates |
| Classification credit | up to **+0.20** | Correct `accidental_bug` / `intentional_backdoor` label |
| Misclassify penalty | **−0.05** | Issue found but labelled wrong |
| False positive penalty | **−0.05** | Substantive comment, no issue matched |
| Coverage bonus | up to **+0.15** | Proportional to fraction of issues found |
| Decision score | **±0.15** | Correct vs. wrong terminal action |
| Efficiency bonus | up to **+0.10** | Finishing early (only unlocked at ≥ 60 % coverage) |

### A Step-by-Step Reward Example

> **Task:** `auth.py` — SQL injection at line 12, hardcoded credential at line 3.

1. Agent calls `add_comment(line=12, comment="Unparameterised query allows SQL injection via username field", category="security")` → passes all three gates → **+0.20 issue credit**
2. Agent calls `add_comment(line=3, comment="Hardcoded DB password in source", category="security", classification="accidental_bug")` → issue credit **+0.20** + classification credit **+0.10**
3. Agent calls `request_changes()` → correct terminal action for a non-adversarial task → **+0.15 decision score** + small coverage + efficiency bonuses
4. **Total: ~+0.78**

---

## The Agent & Baseline

### Scripted Baselines (Reward Validation)

Before touching any ML, we ran four scripted agents to validate the reward function:

```python
# Perfect oracle — reads ground-truth issue data
oracle.add_comment(line=issue.line, comment=issue.keywords[0] + " vulnerability detected here", ...)
oracle.escalate_to_security_review()   # only on adversarial tasks

# Keyword spammer — tries to game the reward
for kw in ALL_SECURITY_KEYWORDS:
    spammer.add_comment(line=9999, comment=kw)   # wrong line every time

# Line flooder — comments on every 5th line
for line in range(5, max_lines, 5):
    flooder.add_comment(line=line, comment="potential issue here, review carefully")
```

| Agent | Mean Reward | What It Proves |
|---|---|---|
| **Perfect Oracle** | **+0.778** | Upper bound — the reward is achievable |
| Line Flooder | −0.025 | Wide-net fishing doesn't work |
| Keyword Spammer | −0.075 | Buzzword dumps don't work |
| Random Agent | −0.260 | Lower bound for comparison |

Both exploit strategies score less than −3 % of oracle reward. ✅

### Zero-Shot LLM Baseline

We ran `gpt-4o-mini` with a system prompt asking it to review the code and call the environment's actions.

<details>
<summary>System prompt (simplified)</summary>

```
You are a senior security engineer reviewing a Python pull request.
You have access to the following tools:
- add_comment(line, comment, category, severity, classification)
- get_context(line)
- run_scanner()
- request_changes() / approve() / escalate_to_security_review()

Review the provided code carefully. Flag every real issue at the correct
line with a specific, substantive comment. On adversarial tasks, classify
each issue as accidental_bug or intentional_backdoor and escalate if the
code contains a deliberate attack.
```
</details>

**Example trajectory — Task 2 (`auth.py`, medium difficulty):**

<details>
<summary>Agent saw → decided → got reward</summary>

```
Step 1  get_context(line=12)        → reveals f-string SQL query
        reward = 0.00               (context action, no reward)

Step 2  add_comment(
          line=12,
          comment="Direct string interpolation in SQL query — classic injection vector",
          category="security",
          classification="accidental_bug"
        )                           → keyword hit ✓, line hit ✓, substantive ✓
        reward = +0.20 (issue) + 0.10 (classification)

Step 3  add_comment(
          line=3,
          comment="DB_PASS hardcoded — rotate and move to env var",
          category="security"
        )                           → reward = +0.20

Step 4  request_changes()           → correct terminal action
        reward = +0.15 (decision) + coverage + efficiency bonuses

Total episode reward: ~+0.72
```
</details>

### GRPO Training

We trained using **GRPO** (Group Relative Policy Optimisation) — an algorithm well-suited to environments where the agent produces a sequence of tool calls and gets a scalar reward at the end.
The training follows a 5-phase curriculum that starts on the ultra-easy bootstrap task and gradually adds harder and adversarial tasks.

```python
CURRICULUM = [
    (0,  40,  [0, 1]),          # Phase 0: ultra-easy + easy
    (40, 80,  [0, 1, 2, 3]),    # Phase 1: adds medium/hard
    (80, 120, list(range(7))),  # Phase 2: adds causal chain
    (120,160, list(range(9))),  # Phase 3: adds adversarial
    (160,200, list(range(10))), # Phase 4: full curriculum
]
```

---

## Results

> **Headline:** <!-- FILL after training: e.g. "After 200 GRPO steps, the trained agent reached +0.XX mean reward — XX% of oracle ceiling — vs. −0.260 for the random baseline." -->

### Reward by Difficulty Tier

<!-- FILL: embed outputs/training_curves.png after training -->
<!-- ![Training curves](outputs/training_curves.png) -->
<!-- *Mean episode reward vs. training step. Smoothed over a 10-step window.* -->

| Tier | Random Baseline | Oracle Ceiling | Trained Agent |
|---|---|---|---|
| Ultra-easy | −0.260 | +0.787 | <!-- FILL --> |
| Easy | −0.260 | +0.800 | <!-- FILL --> |
| Medium | −0.260 | +0.795 | <!-- FILL --> |
| Hard | −0.260 | +0.800 | <!-- FILL --> |
| Adversarial | −0.260 | +1.000 | <!-- FILL --> |

### Where the Agent Consistently Struggled

<!-- FILL: update after training with real failure modes observed -->

From the scripted baselines and zero-shot runs, we already know the hardest failure modes:

- **Adversarial classification** — correctly finding a backdoor but labelling it `accidental_bug` instead of `intentional_backdoor` costs 0.05 per issue and loses the classification credit
- **Causal chains** — Task 6 requires finding the JWT secret *first* to unlock the hint that reveals the privilege escalation path; agents that skip `get_context` miss this entirely
- **False positives under pressure** — agents near their step budget sometimes spam comments hoping to find issues, racking up −0.05 FP penalties instead

---

## What We Learned

- **Reward design is the hard part.** Our first reward was purely binary (found-all-issues or not). The agent learned nothing for 50 steps. Adding partial credit for individual issue hits unlocked learning immediately.

- **The three-gate verifier was essential.** Without the line-hit gate, keyword-spamming was profitable. Without the substantive gate, single-word comments gamed the keyword hit. All three gates are needed simultaneously.

- **The curriculum matters more than we expected.** Starting GRPO on medium-difficulty tasks produced flat reward curves. Routing the first 40 steps through the ultra-easy bootstrap task (with issue keywords literally hinted in the code comments) gave the model enough positive trajectories to bootstrap policy improvement.

- **Adversarial tasks are a qualitatively different capability.** `request_changes` and `escalate_to_security_review` look the same to a model that doesn't understand intent. The classification label (`intentional_backdoor`) is the signal that bridges them — and it only earns credit if the issue was actually found first.

- **The live dashboard made iteration 10× faster.** Watching a model spam comments and collect −0.05 penalties in real time — with the reward ring turning red — made reward-function bugs obvious in seconds instead of minutes of log parsing.

---

## Try It Yourself

```bash
# Clone and install (requires Python 3.10+ and uv)
git clone <!-- FILL: https://github.com/your-org/probe -->
cd probe
uv sync

# Start the server + browser UI in one command
uv run python run.py
# → open http://localhost:8000/ui/
```

| Resource | Link |
|---|---|
| 📓 Training notebook (Colab) | <!-- FILL: add Colab URL --> |
| 🤗 Live environment (HF Spaces) | <!-- FILL: add HF Space URL --> |
| 📄 OpenEnv manifest | [`openenv.yaml`](openenv.yaml) |
| 🔬 Reward grader tests | [`tests/test_grader.py`](tests/test_grader.py) |

---

## What's Next

The ten tasks in PRobe are a starting point, not a ceiling.
We want to add multi-file reviews (where the backdoor only becomes visible when two files are read together), multi-agent variants (one agent reviews, another tries to sneak in the backdoor), and a harder curriculum drawn from real CVE patches.
If you work on code security, LLM evaluation, or RL environments — open an issue or send a PR. We'd love collaborators.

---

## Acknowledgments

We built PRobe at the **OpenEnv Hackathon India 2026**, organised by the [OpenEnv](https://github.com/openenv) team.
Thanks to **Meta** and **PyTorch** for the foundational research behind GRPO and policy optimisation, **Hugging Face** for TRL, Transformers, and the infrastructure that makes running RL on LLMs accessible, **Unsloth** for 4-bit quantisation that fits training on a free T4 GPU, and **Scaler School of Technology** for hosting and mentorship throughout the hackathon.
