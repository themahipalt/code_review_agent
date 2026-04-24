"""
Import compatibility shim for the PRobe server package.

Resolves the three execution contexts the server modules run in:

  1. Installed package  — relative imports work (``from .grader import …``)
  2. ``python -m``      — ``server.*`` absolute imports work
  3. Bare script / test — importlib path-based loading as last resort

Consumers import everything they need from here instead of repeating the
try/except ladder in every server module.
"""

from __future__ import annotations

# Relative imports (context 1 — installed package / pytest via pyproject.toml)
try:
    from .graders import CodeReviewGrader, FALSE_POSITIVE_PENALTY, LINE_TOLERANCE  # noqa: F401
    from .mutator import mutate_task  # noqa: F401
    from .tasks import TASKS  # noqa: F401
    from .episode_memory import EpisodeMemory  # noqa: F401
    from .scanner import run_scanner  # noqa: F401

except ImportError:
    # Absolute imports (context 2 — python -m environment.app)
    try:
        from environment.graders import (  # type: ignore[no-redef]
            CodeReviewGrader,
            FALSE_POSITIVE_PENALTY,
            LINE_TOLERANCE,
        )
        from environment.mutator import mutate_task  # type: ignore[no-redef]
        from environment.tasks import TASKS  # type: ignore[no-redef]
        from environment.episode_memory import EpisodeMemory  # type: ignore[no-redef]
        from environment.scanner import run_scanner  # type: ignore[no-redef]

    except ModuleNotFoundError:
        # Path-based loading (context 3 — bare script, ad-hoc test runner)
        import importlib.util
        import pathlib

        _SERVER_DIR = pathlib.Path(__file__).resolve().parent

        def _load_module(module_name: str):
            module_path = _SERVER_DIR / f"{module_name}.py"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            return module

        _grader_module = _load_module("grader")
        _mutator_module = _load_module("mutator")
        _tasks_module = _load_module("tasks")
        _memory_module = _load_module("episode_memory")
        _scanner_module = _load_module("scanner")

        CodeReviewGrader = _grader_module.CodeReviewGrader  # type: ignore[misc]
        FALSE_POSITIVE_PENALTY = _grader_module.FALSE_POSITIVE_PENALTY  # type: ignore[misc]
        LINE_TOLERANCE = _grader_module.LINE_TOLERANCE  # type: ignore[misc]
        mutate_task = _mutator_module.mutate_task  # type: ignore[misc]
        TASKS = _tasks_module.TASKS  # type: ignore[misc]
        EpisodeMemory = _memory_module.EpisodeMemory  # type: ignore[misc]
        run_scanner = _scanner_module.run_scanner  # type: ignore[misc]
