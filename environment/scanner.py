"""
Simulated Static-Analysis Scanner - live tool interaction layer.

Calling run_scanner(task, seed) simulates what happens when an agent invokes
an external security/lint scanner (e.g. Bandit, Semgrep, Pylint) against
the code under review.

Noise model (seed-controlled for reproducibility)
--------------------------------------------------
  Recall:    Each real issue is reported with probability SCANNER_RECALL (0.70).
             ~30 pct of issues are silently missed - the agent cannot rely
             solely on the scanner.
  Precision: SCANNER_NOISE_RATE false-positive findings are injected per run
             (0 to MAX_FALSE_POSITIVES). False positives reference
             plausible-but-wrong line numbers so the agent must verify first.

Important: scanner results are NOT automatically graded.  The agent must
still call ADD_COMMENT with the correct line + keyword to earn reward.
"""

from __future__ import annotations

import random
from typing import Any

# XOR mask keeps the scanner RNG independent from the mutation-engine RNG
# even when both receive the same episode seed.
_SCANNER_RNG_SEED_MASK: int = 0xDEAD_BEEF

__all__ = ["run_scanner"]

# -- Noise parameters --------------------------------------------------------

SCANNER_RECALL: float = 0.70       # probability each real issue is reported
SCANNER_NOISE_RATE: float = 0.40   # probability a false-positive is injected
MAX_FALSE_POSITIVES: int = 2       # cap on false positives per run

# -- False-positive templates ------------------------------------------------
# Plausible-sounding scanner warnings that do not correspond to any real
# ground-truth issue.  Line numbers are randomised at runtime.

_FALSE_POSITIVE_TEMPLATES: list[dict[str, str]] = [
    {
        "rule":     "B105",
        "message":  "Possible hardcoded password: variable assigned string literal",
        "category": "security",
        "severity": "LOW",
    },
    {
        "rule":     "B324",
        "message":  "Use of weak MD4 or MD5 hash for security; consider stronger algorithm",
        "category": "security",
        "severity": "MEDIUM",
    },
    {
        "rule":     "W0611",
        "message":  "Imported but unused module detected in scope",
        "category": "style",
        "severity": "LOW",
    },
    {
        "rule":     "C0301",
        "message":  "Line too long (82 chars) - PEP 8 violation",
        "category": "style",
        "severity": "LOW",
    },
    {
        "rule":     "B007",
        "message":  "Loop control variable used in inner scope only; consider renaming to _",
        "category": "style",
        "severity": "LOW",
    },
    {
        "rule":     "W0702",
        "message":  "No exception type(s) specified in bare except clause",
        "category": "bug",
        "severity": "MEDIUM",
    },
    {
        "rule":     "B603",
        "message":  "subprocess call without shell=True; verify no injection risk",
        "category": "security",
        "severity": "LOW",
    },
]

_SCANNER_TOOL_NAMES: list[str] = ["bandit 1.7.5", "semgrep 1.45.0", "pylint 3.1.0"]

_UNVERIFIED_FINDINGS_NOTE: str = (
    "Scanner findings are UNVERIFIED. Use GET_CONTEXT or ADD_COMMENT "
    "to confirm each finding before including it in your review. "
    "False positives will be penalised."
)


def run_scanner(task: dict[str, Any], seed: int) -> dict[str, Any]:
    """
    Simulate running a static-analysis scanner against the task's code.

    Parameters
    ----------
    task:
        A (possibly mutated) task dict containing 'code' and 'issues'.
    seed:
        Reproducibility seed.  Different seeds produce different recall/FP draws.

    Returns
    -------
    dict with keys:
        tool:         Scanner name + version string.
        findings:     list[dict] - mix of true positives (recalled) and FPs.
        missed_count: How many real issues were silently missed.
        note:         Reminder that findings must be verified before commenting.
    """
    rng = random.Random(seed ^ _SCANNER_RNG_SEED_MASK)
    total_code_lines = max(len(task["code"].split("\n")), 1)

    true_positive_findings, missed_count = _build_true_positive_findings(
        task=task,
        total_code_lines=total_code_lines,
        rng=rng,
    )
    false_positive_findings = _build_false_positive_findings(
        task=task,
        total_code_lines=total_code_lines,
        rng=rng,
    )

    all_findings = true_positive_findings + false_positive_findings
    # Shuffle so false positives are not trivially identifiable by position.
    rng.shuffle(all_findings)

    return {
        "tool":         rng.choice(_SCANNER_TOOL_NAMES),
        "findings":     all_findings,
        "missed_count": missed_count,
        "note":         _UNVERIFIED_FINDINGS_NOTE,
    }


# -- Private helpers ---------------------------------------------------------

def _build_true_positive_findings(
    task: dict[str, Any],
    total_code_lines: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int]:
    """
    Recall each real issue with probability SCANNER_RECALL.

    Returns a 2-tuple of (findings_list, missed_count).
    """
    findings: list[dict[str, Any]] = []
    missed_count = 0

    for issue in task.get("issues", []):
        if rng.random() >= SCANNER_RECALL:
            missed_count += 1
            continue

        start, end = issue["line_range"]
        # Report midpoint with small stochastic jitter (+-1) so the agent
        # cannot blindly trust the reported line number.
        jitter = rng.choice([-1, 0, 0, 1])  # biased toward 0
        reported_line = max(1, min(total_code_lines, (start + end) // 2 + jitter))

        findings.append({
            "line":     reported_line,
            "rule":     _pick_rule_for_category(issue.get("category", "bug"), rng),
            "message":  _build_issue_message(issue),
            "category": issue.get("category", "bug"),
            "severity": _map_severity_to_scanner_level(issue.get("severity", "warning")),
            "verified": False,
        })

    return findings, missed_count


def _build_false_positive_findings(
    task: dict[str, Any],
    total_code_lines: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Inject up to MAX_FALSE_POSITIVES noise findings on lines away from real issues.
    """
    false_positive_count = sum(
        1 for _ in range(MAX_FALSE_POSITIVES) if rng.random() < SCANNER_NOISE_RATE
    )
    if false_positive_count == 0:
        return []

    # Collect line numbers occupied by real issues to avoid placing FPs there.
    real_issue_lines: set[int] = {
        line_num
        for issue in task.get("issues", [])
        for line_num in range(issue["line_range"][0], issue["line_range"][1] + 1)
    }
    candidate_lines = [
        line_num for line_num in range(1, total_code_lines + 1)
        if line_num not in real_issue_lines
    # If every line belongs to a real issue (tiny synthetic tasks), fall back to
    # the full line range so rng.choice always has candidates to pick from.
    ] or list(range(1, total_code_lines + 1))

    findings: list[dict[str, Any]] = []
    for _ in range(false_positive_count):
        template = rng.choice(_FALSE_POSITIVE_TEMPLATES)
        findings.append({
            "line":     rng.choice(candidate_lines),
            "rule":     template["rule"],
            "message":  template["message"],
            "category": template["category"],
            "severity": template["severity"],
            "verified": False,
        })
    return findings


def _pick_rule_for_category(category: str, rng: random.Random) -> str:
    """Return a plausible lint/security rule ID for the given issue category."""
    rules_by_category: dict[str, list[str]] = {
        "security":    ["B101", "B102", "B105", "B106", "B201", "B301", "B501"],
        "bug":         ["E501", "W0611", "W0702", "E711", "E712"],
        "performance": ["W0640", "C0200", "W0108"],
        "style":       ["C0301", "C0303", "W0611", "C0114"],
        "design":      ["R0201", "R0902", "R0914", "W0107"],
    }
    rule_pool = rules_by_category.get(category, rules_by_category["bug"])
    return rng.choice(rule_pool)


def _build_issue_message(issue: dict[str, Any]) -> str:
    """Build a plausible but intentionally vague scanner message for an issue."""
    messages_by_category: dict[str, list[str]] = {
        "security": [
            "Potential security vulnerability detected in this expression",
            "Sensitive data handling - review for exposure risk",
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
            "Repeated operation inside loop - consider hoisting",
            "Unbounded collection growth detected",
            "Synchronous call inside async context",
            "Sequential requests could be batched",
        ],
        "style": [
            "Code style violation detected",
            "Unused identifier in scope",
            "Magic number - consider named constant",
        ],
        "design": [
            "Resource may not be properly released",
            "Thread or task lifecycle not managed",
            "Retry logic missing backoff strategy",
        ],
    }
    category = issue.get("category", "bug")
    message_pool = messages_by_category.get(category, messages_by_category["bug"])

    # Prefix with the first few words of the issue description so the message
    # is loosely tied to the real issue without being an exact keyword match.
    description_prefix = " ".join(issue.get("description", "").split()[:6])
    suffix_prefix = f"{description_prefix} - " if description_prefix else ""

    # Use a separate RNG seeded on the issue id so this issue always maps to
    # the same scanner message regardless of evaluation order or FP count.
    # Using the main rng here would make messages shift whenever recall draws
    # change, which breaks the reproducibility guarantee.
    issue_rng = random.Random(hash(issue.get("id", "")) & 0xFFFF)
    return suffix_prefix + issue_rng.choice(message_pool)


def _map_severity_to_scanner_level(severity: str) -> str:
    """Map PRobe severity labels to scanner-style HIGH/MEDIUM/LOW levels."""
    severity_map: dict[str, str] = {
        "info":     "LOW",
        "warning":  "MEDIUM",
        "error":    "HIGH",
        "critical": "HIGH",
    }
    return severity_map.get(severity, "MEDIUM")
