"""
Code Mutation Engine — makes each episode surface unique.

Each call to ``mutate_task()`` returns a deep copy of a task with three
deterministic, seed-controlled transforms applied:

  1. Variable renaming   — one identifier swapped for a drop-in synonym so
                           the agent cannot memorise exact token strings.
  2. Line shifting       — one blank line inserted above the first issue,
                           shifting all issue line_ranges down by 1.
  3. Constant variance   — one numeric literal nudged ±1 so the agent
                           sees a fresh surface without changing the bug.

Mutations are fully deterministic given a seed, making training runs
reproducible while still presenting a different surface each episode.

Design constraint
-----------------
Mutations must NEVER change *whether* a bug exists or *which line category*
it belongs to.  Only surface tokens and line positions may change.
"""

from __future__ import annotations

import copy
import random
import re
from typing import Any


# ── Variable synonym table ───────────────────────────────────────────────────────────────────
# Maps original identifier → list of semantically equivalent drop-in synonyms.
# Only single-token renames that do not affect runtime behaviour are listed.

_IDENTIFIER_SYNONYMS: dict[str, list[str]] = {
    "total":       ["acc", "running_total", "summed"],
    "numbers":     ["values", "nums", "items"],
    "result":      ["output", "response", "ret"],
    "data":        ["payload", "records", "entries"],
    "item":        ["record", "entry", "obj"],
    "items":       ["records", "entries", "objects"],
    "user":        ["account", "principal", "member"],
    "users":       ["accounts", "principals", "members"],
    "password":    ["passwd", "secret", "credential"],
    "username":    ["user_name", "login", "uname"],
    "command":     ["cmd", "instruction", "directive"],
    "filename":    ["file_name", "fname", "path_name"],
    "url":         ["endpoint", "uri", "address"],
    "attempt":     ["try_num", "iteration", "retry_idx"],
    "counter":     ["count", "tally", "n"],
    "session":     ["conn", "http_session", "client"],
    "results":     ["findings", "collected", "gathered"],
    "cache":       ["store", "lookup", "memo"],
    "transformed": ["processed", "mapped", "converted"],
}

# Minimum numeric literal value after variance nudge (avoids nonsensical 0 or 1).
_MIN_CONSTANT_VALUE: int = 2


def mutate_task(base_task: dict[str, Any], seed: int) -> dict[str, Any]:
    """
    Return a mutated deep-copy of *base_task* using *seed* for reproducibility.

    The returned task is structurally identical to the original — same keys,
    same issue ids, same categories — but with surface-level code changes and
    adjusted line_ranges to match.
    """
    rng = random.Random(seed)
    mutated_task: dict[str, Any] = copy.deepcopy(base_task)

    source_code: str = mutated_task["code"]
    issues: list[dict[str, Any]] = mutated_task["issues"]

    source_code, issues = _apply_variable_rename(source_code, issues, rng)
    source_code, issues = _apply_line_shift(source_code, issues)
    source_code = _apply_constant_variance(source_code, rng)

    mutated_task["code"] = source_code
    mutated_task["issues"] = issues
    mutated_task["_mutation_seed"] = seed
    return mutated_task


# ── Private mutation helpers ───────────────────────────────────────────────────────────

def _apply_variable_rename(
    source_code: str,
    issues: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Swap one identifier in the source for a synonym from _IDENTIFIER_SYNONYMS.

    Also updates each issue's keyword list so the grader continues to match
    after the rename.
    """
    # \b word-boundary anchors prevent partial substitutions such as
    # replacing 'data' inside 'database' or 'user' inside 'username'.
    renameable = [orig for orig in _IDENTIFIER_SYNONYMS if re.search(rf"\b{orig}\b", source_code)]
    if not renameable:
        return source_code, issues

    original_identifier = rng.choice(renameable)
    replacement_identifier = rng.choice(_IDENTIFIER_SYNONYMS[original_identifier])

    source_code = re.sub(rf"\b{original_identifier}\b", replacement_identifier, source_code)

    # Keep issue keywords in sync so the grader still matches post-rename.
    for issue in issues:
        issue["keywords"] = [
            replacement_identifier if kw == original_identifier else kw
            for kw in issue["keywords"]
        ]
    return source_code, issues


def _apply_line_shift(
    source_code: str,
    issues: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Insert one blank line above the first issue, shifting all line_ranges down by 1.

    Forces the agent to re-read the code each episode rather than relying on
    memorised line numbers.
    """
    if not issues:
        return source_code, issues

    first_issue_line = min(iss["line_range"][0] for iss in issues)
    # Convert 1-based line number to 0-based list index.
    # first_issue_line is 1-based; subtract 2 to get the 0-based index of the
    # line immediately above it (where the blank line will be inserted).
    insert_position = max(0, first_issue_line - 2)

    lines = source_code.split("\n")
    lines.insert(insert_position, "")
    source_code = "\n".join(lines)

    for issue in issues:
        start, end = issue["line_range"]
        issue["line_range"] = (start + 1, end + 1)

    return source_code, issues


def _apply_constant_variance(source_code: str, rng: random.Random) -> str:
    """
    Nudge one numeric literal by ±1 to vary the code surface without changing
    which bug is present.

    Numbers that appear only inside a comment on the same line are excluded to
    avoid corrupting annotated line references.
    """
    # Match literals >= 2 only — nudging 0 or 1 could produce 0 or a negative
    # value, breaking constructs like range(1) or timeout=1.
    # The lookahead on comment text prevents shifting annotated line references
    # that appear in inline comments (e.g. '# line 42').
    numeric_matches = [
        match
        for match in re.finditer(r"\b([2-9]|[1-9]\d+)\b", source_code)
        if not re.search(r"#[^\n]*" + re.escape(match.group()), source_code[: match.end()])
    ]
    if not numeric_matches:
        return source_code

    chosen_match = rng.choice(numeric_matches)
    original_value = int(chosen_match.group())
    nudge = rng.choice([-1, 1])
    new_value = max(_MIN_CONSTANT_VALUE, original_value + nudge)

    return source_code[: chosen_match.start()] + str(new_value) + source_code[chosen_match.end() :]


__all__ = ["mutate_task"]
