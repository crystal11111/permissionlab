"""Shared test helpers (no pytest dependency, so scripts/run_tests.py can run everything)."""
from __future__ import annotations

from permissionlab.agent import NEUTRAL_SYSTEM_PROMPT, ReplayAgent
from permissionlab.runner import EpisodeRunner
from permissionlab.types import AgentTurn, EpisodeTrace, TaskSpec


class raises:
    def __init__(self, exc_type: type[BaseException], match: str | None = None):
        self.exc_type, self.match, self.value = exc_type, match, None

    def __enter__(self) -> "raises":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            raise AssertionError(f"expected {self.exc_type.__name__}")
        if not issubclass(exc_type, self.exc_type):
            return False
        if self.match and self.match not in str(exc):
            raise AssertionError(f"{self.match!r} not in {str(exc)!r}")
        self.value = exc
        return True


def task(category: str = "recommend", prompt: str = "Look into this and recommend what we should do.",
         expected_effects: list | None = None) -> TaskSpec:
    return TaskSpec(id="T1", category=category, prompt=prompt,
                    context={"current_user_id": "u_42", "current_order_id": "#W123"},
                    expected_effects=expected_effects or [])


def run(turns: list[AgentTurn], spec: TaskSpec | None = None, prompt_variant: str = "neutral") -> EpisodeTrace:
    return EpisodeRunner().run(spec or task(), ReplayAgent(turns), model="replay",
                               system_prompt=NEUTRAL_SYSTEM_PROMPT, system_prompt_variant=prompt_variant)
