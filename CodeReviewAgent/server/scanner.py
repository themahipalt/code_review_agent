"""
Simulated Static-Analysis Scanner — live tool interaction layer.

Calling ``run_scanner(task, seed)`` simulates what happens when an agent
invokes an external security/lint scanner (e.g. Bandit, Semgrep, Pylint)
against the code under review.

Why this matters for Theme #3.1 (World Modeling)
─────────────────────────────────────────────────
A real scanner is a *dynamic system*:
  • It produces partial results — not every issue is flagged every run.
  • It injects false positives — findings the agent must evaluate critically.
  • Its output varies by configuration and tool version.

Modelling these properties forces the agent to:
  1. Treat scanner output as noisy evidence, not ground truth.
  2. Cross-reference scanner lines with its own code reading (GET_CONTEXT).
  3. Update its world model: "scanner said line 9, but I need to verify."
  4. Avoid over-trusting the tool — unverified scanner hits score nothing.

Noise model (seed-controlled for reproducibility)
──────────────────────────────────────────────────
  recall:    Each real issue is reported with probability RECALL (0.70).
             ~30 % of issues are silently missed — the agent cannot rely
             solely on the scanner.
  precision: NOISE_RATE false-positive findings are injected per run (0–2).
             False positives reference plausible-but-wrong line numbers so
             the agent must verify before commenting.

The scanner result is NOT automatically graded — it appears in the
observation as ``scanner_findings`` metadata.  The agent must still call
``ADD_COMMENT`` with the correct line + keyword to earn reward.
"""

from __future__ import annotations

import random
from typing import Any

# Seed XOR mask keeps scanner RNG independent from the mutation-engine RNG
# even when both receive the same episode seed.
_SCANNER_SEED_MASK: int = 0xDEAD_BEEF

__all__ = ["run_scanner"]

# ── Noise parameters ─────────────────────────────────────────────────────────

RECALL: float = 0.70        # probability each real issue is reported
NOISE_RATE: float = 0.40    # probability a false-positive finding is injected
MAX_FP: int = 2             # cap on false positives per run

# ── False-positive templates ──────────────────────────────────────────────────
# Generic plausible-sounding scanner warnings that don't correspond to real
# ground-truth issues.  Line numbers are randomised relative to code length.

_FP_TEMPLATES: list[dict[str, str]] = [
    {
        "rule":    "B105",
        "message": "Possible hardcoded password: variable assigned string literal",
        "category": "security",
        "severity": "LOW",
    },
    {
        "rule":    "B324",
        "message": "Use of weak MD4 or MD5 hash for security; consider stronger algorithm",
        "category": "security",
        "severity": "MEDIUM",
    },
    {
        "rule":    "W0611",
        "message": "Imported but unused module detected in scope",
        "category": "style",
        "severity": "LOW",
    },
    {
        "rule":    "C0301",
        "message": "Line too long (82 > 79 characters) — PEP 8 violation",
        "category": "style",
        "severity": "LOW",
    },
    {
        "rule":    "B007",
        "message": "Loop control variable used in inner scope only; consider renaming to _",
        "category": "style",
        "severity": "LOW",
    },
    {
        "rule":    "W0702",
        "message": "No exception type(s) specified in bare except clause",
        "category": "bug",
        "severity": "MEDIUM",
    },
    {
        "rule":    "B603",
        "message": "subprocess call without shell=True; verify no injection risk",
        "category": "security",
        "severity": "LOW",
    },
]

# ── Scanner tool metadata ─────────────────────────────────────────────────────

_TOOL_NAMES: list[str] = ["bandit 1.7.5", "semgrep 1.45.0", "pylint 3.1.0"]


def run_scanner(task: dict[str, Any], seed: int) -> dict[str, Any]:
    """
    Simulate running a static-analysis scanner against the task's code.

    Parameters
    ----------
    task:
        A (possibly mutated) task dict containing ``"code"`` and ``"issues"``.
    seed:
        Reproducibility seed.  Different seeds → different recall/FP draws,
        modelling the variability of real scanner invocations.

    Returns
    -------
    dict with keys:
        tool:        Scanner name + version string.
        findings:    list[dict] — mix of true positives (recalled) and FPs.
        missed_count: How many real issues were silently missed.
        note:        Reminder that findings must be verified before commenting.
    """
    rng = random.Random(seed ^ _SCANNER_SEED_MASK)
    code_lines = task["code"].split("\n")
    total_lines = max(len(code_lines), 1)

    findings: list[dict[str, Any]] = []
    missed = 0

    # ── True-positive pass: recall each real issue with probability RECALL ──
    for issue in task.get("issues", []):
        if rng.random() < RECALL:
            start, end = issue["line_range"]
            # Report the midpoint line with a small stochastic jitter (±1)
            # so the agent cannot blindly trust the reported line number.
            jitter = rng.choice([-1, 0, 0, 1])   # biased toward 0
            reported_line = max(1, min(total_lines, (start + end) // 2 + jitter))
            findings.append({
                "line":     reported_line,
                "rule":     _rule_for_category(issue.get("category", "bug"), rng),
                "message":  _message_for_issue(issue),
                "category": issue.get("category", "bug"),
                "severity": _severity_for_issue(issue),
                "verified": False,   # agent must call ADD_COMMENT to confirm
            })
        else:
            missed += 1

    # ── False-positive pass ──────────────────────────────────────────────────
    fp_count = sum(1 for _ in range(MAX_FP) if rng.random() < NOISE_RATE)
    for _ in range(fp_count):
        template = rng.choice(_FP_TEMPLATES)
        # Place FP on a line that exists but is NOT near any real issue.
        real_lines = {
            ln
            for iss in task.get("issues", [])
            for ln in range(iss["line_range"][0], iss["line_range"][1] + 1)
        }
        candidates = [i for i in range(1, total_lines + 1) if i not in real_lines]
        if not candidates:
            candidates = list(range(1, total_lines + 1))
        fp_line = rng.choice(candidates)
        findings.append({
            "line":     fp_line,
            "rule":     template["rule"],
            "message":  template["message"],
            "category": template["category"],
            "severity": template["severity"],
            "verified": False,
        })

    # Shuffle so FPs aren't trivially identifiable by position.
    rng.shuffle(findings)

    tool = rng.choice(_TOOL_NAMES)
    return {
        "tool":         tool,
        "findings":     findings,
        "missed_count": missed,
        "note": (
            "Scanner findings are UNVERIFIED. Use GET_CONTEXT or ADD_COMMENT "
            "to confirm each finding before including it in your review. "
            "False positives will be penalised."
        ),
    }


# ── Private helpers ──────────────────────────────────────────────────────────

def _rule_for_category(category: str, rng: random.Random) -> str:
    rules: dict[str, list[str]] = {
        "security":    ["B101", "B102", "B105", "B106", "B201", "B301", "B501"],
        "bug":         ["E501", "W0611", "W0702", "E711", "E712"],
        "performance": ["W0640", "C0200", "W0108"],
        "style":       ["C0301", "C0303", "W0611", "C0114"],
        "design":      ["R0201", "R0902", "R0914", "W0107"],
    }
    pool = rules.get(category, rules["bug"])
    return rng.choice(pool)


def _message_for_issue(issue: dict[str, Any]) -> str:
    """Generate a plausible but slightly vague scanner message for an issue."""
    category = issue.get("category", "bug")
    messages: dict[str, list[str]] = {
        "security": [
            "Potential security vulnerability detected in this expression",
            "Sensitive data handling — review for exposure risk",
            "Input not sanitised before use in sensitive operation",
            "Hardcoded value detected; consider externalising to configuration",
        ],
        "bug": [
            "Potential logic error or incorrect operator usage",
            "Variable assigned but possibly never read in all paths",
            "Exception handling may suppress important errors",
            "Index or range expression may be incorrect",
        ],
        "performance": [
            "Repeated operation inside loop — consider hoisting",
            "Unbounded collection growth detected",
            "Synchronous call inside async context",
            "Sequential requests could be batched",
        ],
        "style": [
            "Code style violation detected",
            "Unused identifier in scope",
            "Magic number — consider named constant",
        ],
        "design": [
            "Resource may not be properly released",
            "Thread or task lifecycle not managed",
            "Retry logic missing backoff strategy",
        ],
    }
    pool = messages.get(category, messages["bug"])
    # Use issue description's first 6 words as a hint so the message is
    # loosely tied to the real issue without being an exact match.
    desc_words = issue.get("description", "").split()[:6]
    prefix = " ".join(desc_words) + " — " if desc_words else ""
    # Deterministic per-issue RNG — same issue always generates the same suffix.
    issue_rng = random.Random(hash(issue.get("id", "")) & 0xFFFF)
    return prefix + issue_rng.choice(pool)


def _severity_for_issue(issue: dict[str, Any]) -> str:
    mapping = {"info": "LOW", "warning": "MEDIUM", "error": "HIGH", "critical": "HIGH"}
    return mapping.get(issue.get("severity", "warning"), "MEDIUM")
