# PRobe — Design Notes

See the top-level [README](../README.md) for the full environment description,
reward function breakdown, and task catalogue.

## Repository layout

```
repo-root/
├── agent/           # Client API (ProbeEnv, ProbeAction, ProbeObservation)
├── environment/     # FastAPI server + RL environment logic
├── training/        # GRPO training and baseline evaluation scripts
├── tests/           # pytest suite
├── outputs/         # logs, reward curves, artefacts (git-ignored)
└── docs/            # design notes (this file)
```

## Environment entry point

`environment/app.py` — FastAPI app mounted at `/ui/` (static frontend) and `/docs` (API).  
`openenv.yaml` → `app: environment.app:app`.

## Reward function

See `environment/graders.py` for the deterministic keyword+line-range grader.

## Training

`training/train_grpo.py` — single-turn GRPO via HuggingFace TRL.  
`training/baseline.py` — zero-shot GPT-4o-mini baseline.  
`training/scripted_baseline.py` — deterministic oracle / spammer stress-tests.
