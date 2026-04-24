"""
Cross-episode memory for PRobe.

Maintains a JSON log of past episode findings so the agent can receive
"prior_findings_hint" observations on tasks it has encountered before.
This models the real-world reality that a security reviewer builds a mental
model of recurring vulnerability patterns across many code reviews.

Design:
  - One JSON file per environment instance (keyed by task_id).
  - Records: task_id → list of issue_ids solved in past episodes.
  - On reset, any task that has prior findings injects a summary hint into
    the initial observation's context_hints list.
  - Memory file is optional: if it cannot be read/written the environment
    degrades gracefully (no hints, no crash).
  - File location: configurable, defaults to a temp directory so tests
    remain isolated without explicit cleanup.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["EpisodeMemory"]

log = logging.getLogger(__name__)

_DEFAULT_MEMORY_DIR = Path(tempfile.gettempdir()) / "probe_memory"


class EpisodeMemory:
    """
    Lightweight JSON-backed store of cross-episode findings.

    Parameters
    ----------
    memory_dir:
        Directory where per-instance memory files are stored.
        Defaults to a system temp directory so tests are isolated.
    instance_id:
        Unique identifier for this environment instance (used as filename).
        Defaults to "default".
    """

    def __init__(
        self,
        memory_dir: Path | str | None = None,
        instance_id: str = "default",
    ) -> None:
        self._dir = Path(memory_dir) if memory_dir else _DEFAULT_MEMORY_DIR
        self._file = self._dir / f"probe_memory_{instance_id}.json"
        self._data: dict[str, list[str]] = {}  # task_id (str) → [issue_id, ...]
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load existing memory from disk; silently ignore missing/corrupt files."""
        try:
            if self._file.exists():
                raw = json.loads(self._file.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = {k: list(v) for k, v in raw.items() if isinstance(v, list)}
        except Exception as exc:  # noqa: BLE001
            log.warning("EpisodeMemory: could not load %s — %s", self._file, exc)
            self._data = {}

    def _save(self) -> None:
        """Persist memory to disk; silently ignore write errors."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._file.write_text(
                json.dumps(self._data, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("EpisodeMemory: could not save %s — %s", self._file, exc)

    # ── Public API ────────────────────────────────────────────────────────

    def record(self, task_id: int, issues_found: list[str]) -> None:
        """
        Record which issues were found for a given task after episode end.

        Merges with any previously recorded findings (deduplicates).
        """
        key = str(task_id)
        existing = set(self._data.get(key, []))
        existing.update(issues_found)
        self._data[key] = sorted(existing)
        self._save()

    def prior_hint(self, task_id: int, task: dict[str, Any]) -> str | None:
        """
        Return a context hint string summarising past findings for this task,
        or None if this task has never been seen before.

        The hint is intentionally vague — it tells the agent *categories* of
        issues found previously, not exact line numbers, so the agent must
        still do the work of locating them in the (potentially mutated) code.
        """
        key = str(task_id)
        prior_ids = self._data.get(key, [])
        if not prior_ids:
            return None

        # Map issue_id → category from the task definition
        id_to_cat: dict[str, str] = {
            iss["id"]: iss.get("category", "unknown")
            for iss in task.get("issues", [])
        }
        categories: list[str] = sorted(
            {id_to_cat[i] for i in prior_ids if i in id_to_cat}
        )
        cat_str = ", ".join(categories) if categories else "various"
        count = len(prior_ids)
        task_name = task.get("name", f"task {task_id}")

        return (
            f"=== PRIOR KNOWLEDGE: {task_name} ===\n"
            f"In a previous review of this file you found {count} issue(s) "
            f"in the following categories: {cat_str}.\n"
            "The code may have changed slightly — verify each issue at its "
            "current location before commenting."
        )

    def clear(self, task_id: int | None = None) -> None:
        """
        Clear memory for a specific task (or all tasks if task_id is None).
        Useful for test isolation.
        """
        if task_id is None:
            self._data = {}
        else:
            self._data.pop(str(task_id), None)
        self._save()
