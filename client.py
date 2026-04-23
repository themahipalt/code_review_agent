"""PRobe Environment Client."""

from __future__ import annotations

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ProbeAction, ProbeObservation


class ProbeEnv(EnvClient[ProbeAction, ProbeObservation, State]):
    """
    Client for the PRobe environment.

    Maintains a persistent WebSocket connection to the server.

    Example::

        with ProbeEnv(base_url="http://localhost:8000") as env:
            result = env.reset()
            print(result.observation.task_description)

            action = ProbeAction(
                action_type="add_comment",
                line_number=4,
                comment="Off-by-one: range(len+1) causes IndexError",
                severity="error",
                category="bug",
            )
            result = env.step(action)
            print(result.reward)
    """

    def _step_payload(self, action: ProbeAction) -> dict:
        payload: dict = {"action_type": action.action_type.value}
        if action.line_number is not None:
            payload["line_number"] = action.line_number
        if action.comment is not None:
            payload["comment"] = action.comment
        if action.severity is not None:
            payload["severity"] = action.severity.value
        if action.category is not None:
            payload["category"] = action.category.value
        return payload

    def _parse_result(
        self, payload: dict
    ) -> StepResult[ProbeObservation]:
        obs_data: dict = payload.get("observation", {})
        # Use model_validate so new fields added to ProbeObservation
        # are picked up automatically without changing this method.
        observation = ProbeObservation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=float(payload.get("reward") or 0.0),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
