"""
Async LLM baseline for the PRobe environment.

Runs an AsyncOpenAI model against all tasks and produces:
  - Per-step JSONL log  (--output path.jsonl)
  - Episode summary JSON (--summary path.json)
  - Console table

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py

    # Custom options
    python baseline.py --model gpt-4o --temperature 0.3 --episodes 2 --task-id 1

    # All tasks, 3 episodes each, save logs
    python baseline.py --episodes 3 --output logs/run.jsonl --summary logs/summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "probe"))

from openai import AsyncOpenAI

from probe.server.probe_environment import ProbeEnvironment
from probe.models import ActionType, ProbeAction

SYSTEM_PROMPT = """\
You are a senior software engineer performing a pull-request code review.
You interact with the review environment by emitting a single JSON action per turn.

Available actions:

1. ADD_COMMENT — annotate a specific line:
   {"action_type": "add_comment", "line_number": <int>, "comment": "<text>",
    "severity": "<info|warning|error|critical>", "category": "<bug|security|performance|style|design>"}

2. GET_CONTEXT — reveal ±5 lines around a suspicious line (free near issues, -0.01 elsewhere):
   {"action_type": "get_context", "line_number": <int>}

3. REQUEST_CHANGES — signal the PR needs work (after adding all comments):
   {"action_type": "request_changes", "comment": "<brief summary>"}

4. APPROVE — approve the PR (only when no significant issues remain):
   {"action_type": "approve"}

5. SUBMIT_REVIEW — finalise and submit the review (ends the episode):
   {"action_type": "submit_review"}

Strategy:
- Read the code carefully. Use GET_CONTEXT on any suspicious line before commenting.
- Read every CONTEXT HINT you receive — they reveal deeper system context.
- Add one ADD_COMMENT for every issue you find (line number + severity + category).
- Decide REQUEST_CHANGES if issues exist, APPROVE if the code is clean.
- Always end with SUBMIT_REVIEW.

Reply with ONLY a valid JSON object — no markdown fences, no explanation.\
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_action(text: str) -> ProbeAction:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(ln for ln in lines if not ln.startswith("```")).strip()
    data = json.loads(text)
    return ProbeAction(**data)


def _format_user_message(obs, step: int) -> str:
    history_lines = ""
    if obs.review_history:
        recent = obs.review_history[-6:]
        history_lines = "\n\nYour review so far:\n" + "\n".join(
            f"  [{e.get('type')}] line={e.get('line')} — {str(e.get('text', ''))[:90]}"
            for e in recent
        )
    hints_section = ""
    if obs.context_hints:
        hints_section = "\n\n" + "\n\n".join(
            f"CONTEXT HINT {i + 1}:\n{hint}"
            for i, hint in enumerate(obs.context_hints)
        )
    return (
        f"File: {obs.file_name}  |  Task: {obs.task_difficulty.upper()}\n"
        f"Objective: {obs.task_description}\n"
        f"Progress: step {step}, issues found {obs.issues_found_count}/{obs.total_issues}\n\n"
        f"```python\n{obs.code_snippet}```"
        f"{hints_section}"
        f"{history_lines}\n\n"
        "What is your next action?"
    )


# ── Single episode ────────────────────────────────────────────────────────────

async def run_episode(
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    task_id: int,
    episode_idx: int,
    jsonl_file,
) -> dict:
    env = ProbeEnvironment()

    # Cycle env resets to land on the target task
    for _ in range(task_id + 1):
        obs = await env.async_reset()

    total_issues = obs.total_issues
    episode_id = obs.metadata.get("episode_id", "?")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    cumulative_reward = 0.0
    step = 0
    t0 = time.perf_counter()

    print(f"\n{'─'*60}")
    print(f"Task {task_id} [{obs.task_difficulty}] ep={episode_idx}  "
          f"{obs.file_name}  ({total_issues} issues)")
    print(f"{'─'*60}")

    while True:
        messages.append({"role": "user", "content": _format_user_message(obs, step)})

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            assistant_text = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": assistant_text})

            action = _parse_action(assistant_text)
            obs, reward_obj, done, info = await env.async_step(action)
            step += 1
            cumulative_reward += reward_obj.total

            # ── JSONL: one record per step ────────────────────────────────
            step_record = {
                "episode_id": episode_id,
                "episode_idx": episode_idx,
                "task_id": task_id,
                "task_difficulty": obs.task_difficulty,
                "step": step,
                "action_type": action.action_type.value,
                "line_number": action.line_number,
                "comment": (action.comment or "")[:120],
                "reward_total": reward_obj.total,
                "reward_components": reward_obj.components,
                "reward_passed": reward_obj.passed,
                "reward_explanation": reward_obj.explanation,
                "cumulative_reward": round(cumulative_reward, 4),
                "issues_found": obs.issues_found_count,
                "total_issues": total_issues,
                "done": done,
            }
            if jsonl_file:
                jsonl_file.write(json.dumps(step_record) + "\n")
                jsonl_file.flush()

            print(
                f"  {step:2d}. {action.action_type.value:<20}"
                f" reward={reward_obj.total:+.4f}"
                f"  cum={cumulative_reward:+.4f}"
                f"  issues={obs.issues_found_count}/{total_issues}"
                f"  {'DONE' if done else ''}"
            )

            if done:
                break

        except json.JSONDecodeError as exc:
            print(f"  [WARN] JSON parse error at step {step}: {exc} — forcing submit")
            obs, reward_obj, done, _ = await env.async_step(
                ProbeAction(action_type=ActionType.SUBMIT_REVIEW)
            )
            cumulative_reward += reward_obj.total
            break

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            break

    elapsed = time.perf_counter() - t0
    coverage = obs.issues_found_count / total_issues if total_issues else 0.0
    print(
        f"\n  → cumulative_reward={cumulative_reward:+.4f}"
        f"  coverage={coverage:.0%}"
        f"  steps={step}"
        f"  elapsed={elapsed:.1f}s"
    )
    return {
        "episode_id": episode_id,
        "episode_idx": episode_idx,
        "task_id": task_id,
        "difficulty": obs.task_difficulty,
        "file_name": obs.file_name,
        "cumulative_reward": round(cumulative_reward, 4),
        "steps": step,
        "issues_found": obs.issues_found_count,
        "total_issues": total_issues,
        "coverage": round(coverage, 3),
        "elapsed_s": round(elapsed, 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def async_main(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY environment variable is not set.")

    client = AsyncOpenAI(api_key=api_key)
    print(f"PRobe LLM Baseline  |  model={args.model}  temp={args.temperature}")

    from probe.server.tasks import TASKS
    task_ids = [args.task_id] if args.task_id is not None else list(range(len(TASKS)))

    jsonl_file = open(args.output, "a") if args.output else None  # noqa: SIM115
    all_results: list[dict] = []

    try:
        for task_id in task_ids:
            for ep_idx in range(args.episodes):
                result = await run_episode(
                    client=client,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    task_id=task_id,
                    episode_idx=ep_idx,
                    jsonl_file=jsonl_file,
                )
                all_results.append(result)
    finally:
        if jsonl_file:
            jsonl_file.close()

    avg_reward = sum(r["cumulative_reward"] for r in all_results) / len(all_results)
    avg_coverage = sum(r["coverage"] for r in all_results) / len(all_results)

    print(f"\n{'═'*65}")
    print("BASELINE SUMMARY")
    print(f"{'═'*65}")
    print(f"  {'Task':<6} {'Diff':<12} {'Ep':>3} {'Reward':>8} {'Coverage':>10} {'Steps':>6}")
    print(f"  {'─'*53}")
    for r in all_results:
        print(
            f"  {r['task_id']:<6} {r['difficulty']:<12} {r['episode_idx']:>3}"
            f" {r['cumulative_reward']:>+8.4f} {r['coverage']:>9.0%}  {r['steps']:>5}"
        )
    print(f"  {'─'*53}")
    print(f"  {'avg':<22} {avg_reward:>+8.4f} {avg_coverage:>9.0%}")

    output_data = {
        "model": args.model,
        "temperature": args.temperature,
        "results": all_results,
        "summary": {
            "avg_cumulative_reward": round(avg_reward, 4),
            "avg_coverage": round(avg_coverage, 3),
            "total_episodes": len(all_results),
        },
    }
    summary_path = args.summary or os.path.join(os.path.dirname(__file__), "baseline_results.json")
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSummary → {summary_path}")
    if args.output:
        print(f"Step log → {args.output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async LLM baseline for PRobe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=350, dest="max_tokens")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per task")
    parser.add_argument("--task-id", type=int, default=None, dest="task_id",
                        help="Run a single task ID (default: all)")
    parser.add_argument("--output", default=None, help="JSONL path for per-step logs")
    parser.add_argument("--summary", default=None, help="JSON path for episode summary")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(_parse_args()))
