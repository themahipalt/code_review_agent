"""
Async FastAPI server for the PRobe environment.

Endpoints:
  POST /reset              — start a new episode (HTTP session)
  POST /step               — execute one action
  GET  /state              — current episode snapshot
  GET  /health             — liveness probe
  GET  /schema             — action / observation schema
  WS   /ws                 — WebSocket session (own env per connection)

HTTP endpoints share a single env instance (sequential use).
WebSocket endpoints each spin up an isolated env instance, enabling
concurrent GRPO rollouts.

OpenEnv web interface is mounted at /web via create_app if available;
falls back to a minimal HTML redirect page.
"""

from __future__ import annotations

import json
import logging
import pathlib
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    from openenv.core.env_server.http_server import create_app as _create_openenv_app
    _OPENENV_AVAILABLE = True
except Exception:  # pragma: no cover
    _OPENENV_AVAILABLE = False

try:
    from ..agent.models import ProbeAction, ProbeObservation, RewardType
    from .probe_environment import ProbeEnvironment
except (ImportError, ModuleNotFoundError):
    from agent.models import ProbeAction, ProbeObservation, RewardType  # type: ignore
    from environment.probe_environment import ProbeEnvironment  # type: ignore

log = logging.getLogger(__name__)


# ── Shared HTTP session env ───────────────────────────────────────────────────

_http_env: ProbeEnvironment | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _http_env
    _http_env = ProbeEnvironment()
    yield
    _http_env = None


# ── Response shapes ───────────────────────────────────────────────────────────

class StepResponse:
    """Serialisable wrapper around an async_step result."""

    def __init__(
        self,
        observation: ProbeObservation,
        reward: RewardType,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.model_dump(),
            "reward": self.reward.model_dump(),
            "done": self.done,
            "info": self.info,
        }


# ── App factory ───────────────────────────────────────────────────────────────

# Resolve the frontend directory relative to this file so the app works
# regardless of the working directory it is launched from.
_FRONTEND_DIR = pathlib.Path(__file__).parent.parent / "frontend"


def _build_app() -> FastAPI:
    application = FastAPI(
        title="PRobe",
        description="OpenEnv code-review environment — async FastAPI server.",
        version="2.0.0",
        lifespan=lifespan,
    )

    # Allow the frontend (served on the same host, any port) to call the API.
    # In production, restrict allow_origins to the exact frontend URL.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── HTTP endpoints ────────────────────────────────────────────────────

    @application.post("/reset", summary="Start a new episode")
    async def reset_endpoint() -> dict[str, Any]:
        if _http_env is None:
            raise HTTPException(status_code=503, detail="Environment not initialised")
        obs = await _http_env.async_reset()
        return {"observation": obs.model_dump(), "reward": None, "done": False, "info": {}}

    @application.post("/step", summary="Execute one action")
    async def step_endpoint(action: ProbeAction) -> dict[str, Any]:
        if _http_env is None:
            raise HTTPException(status_code=503, detail="Environment not initialised")
        obs, reward, done, info = await _http_env.async_step(action)
        return StepResponse(obs, reward, done, info).to_dict()

    @application.get("/state", summary="Current episode state snapshot")
    async def state_endpoint() -> dict[str, Any]:
        if _http_env is None:
            raise HTTPException(status_code=503, detail="Environment not initialised")
        return await _http_env.async_state()

    @application.get("/health", summary="Liveness probe")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/schema", summary="Action and observation JSON schemas")
    async def schema() -> dict[str, Any]:
        return {
            "action": ProbeAction.model_json_schema(),
            "observation": ProbeObservation.model_json_schema(),
            "reward": RewardType.model_json_schema(),
        }

    # ── WebSocket endpoint (one env per connection) ───────────────────────

    @application.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        env = ProbeEnvironment()
        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                cmd = msg.get("command")

                if cmd == "reset":
                    obs = await env.async_reset()
                    await websocket.send_json(
                        {"type": "reset", "observation": obs.model_dump()}
                    )

                elif cmd == "step":
                    try:
                        action = ProbeAction(**msg["action"])
                    except Exception as exc:
                        await websocket.send_json({"type": "error", "detail": str(exc)})
                        continue
                    obs, reward, done, info = await env.async_step(action)
                    await websocket.send_json(
                        {
                            "type": "step",
                            "observation": obs.model_dump(),
                            "reward": reward.model_dump(),
                            "done": done,
                            "info": info,
                        }
                    )

                elif cmd == "state":
                    state = await env.async_state()
                    await websocket.send_json({"type": "state", "state": state})

                else:
                    await websocket.send_json(
                        {"type": "error", "detail": f"Unknown command: {cmd}"}
                    )

        except WebSocketDisconnect:
            pass

    # ── Web UI ────────────────────────────────────────────────────────────
    # /web → redirect so old links still work
    @application.get("/web", response_class=HTMLResponse, include_in_schema=False)
    async def web_redirect() -> HTMLResponse:
        return HTMLResponse(
            '<meta http-equiv="refresh" content="0;url=/ui/">',
            status_code=200,
        )

    # Mount the compiled frontend as a static site at /ui.
    # Falls back gracefully if the frontend directory has not been built yet.
    if _FRONTEND_DIR.is_dir():
        application.mount("/ui", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="ui")
        log.info("Frontend mounted at /ui from %s", _FRONTEND_DIR)
    else:
        log.warning(
            "Frontend directory not found at %s — /ui will not be available. "
            "Run the frontend build or create the 'frontend/' directory.",
            _FRONTEND_DIR,
        )

    return application


app = _build_app()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:  # noqa: S104 — bind addr is configurable via CLI
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
