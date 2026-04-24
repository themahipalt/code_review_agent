"""
PRobe — unified launcher.

Starts the FastAPI server which serves:
  - The interactive frontend at  http://localhost:8000/ui/
  - The REST API at              http://localhost:8000/docs
  - The WebSocket at             ws://localhost:8000/ws

Usage
-----
  uv run python run.py              # default: host=0.0.0.0, port=8000
  uv run python run.py --port 9000
  uv run python run.py --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Add the project root to sys.path so both `agent` and `environment` packages
# are importable regardless of how or from where this script is invoked.
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ── Now safe to import the app ────────────────────────────────────────────────
from environment.app import app  # noqa: E402  (import after path setup)
import uvicorn                   # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the PRobe environment server + frontend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload on code changes (dev mode)")
    args = parser.parse_args()

    frontend_url = f"http://{'localhost' if args.host == '0.0.0.0' else args.host}:{args.port}/ui/"
    api_url      = f"http://{'localhost' if args.host == '0.0.0.0' else args.host}:{args.port}/docs"

    print("\n" + "=" * 58)
    print("  PRobe — AI Code Review Training Environment")
    print("=" * 58)
    print(f"  Frontend   →  {frontend_url}")
    print(f"  API docs   →  {api_url}")
    print(f"  WebSocket  →  ws://localhost:{args.port}/ws")
    print("=" * 58 + "\n")

    uvicorn.run(
        "environment.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        # Keep uvicorn's own logging minimal so our banner stays visible
        log_level="warning",
    )


if __name__ == "__main__":
    main()
