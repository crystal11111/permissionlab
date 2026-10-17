from __future__ import annotations

from copy import deepcopy
from typing import Any

from .agent import AgentBackend
from .db import RetailDB
from .tools import InvalidToolCall, RetailTools, ToolError
from .types import EpisodeTrace, TaskSpec, ToolEvent


class EpisodeRunner:
    """Runs one single-instruction episode and always returns a trace.

    Nothing that already happened is dropped: a refused tool call is recorded and
    handed back to the agent, and a model/API failure or harness bug ends the
    episode with the trace and final DB snapshot intact.
    """

    def __init__(self, max_steps: int = 15):
        self.max_steps = max_steps

    def run(
        self,
        task: TaskSpec,
        agent: AgentBackend,
        *,
        model: str,
        system_prompt: str,
        system_prompt_variant: str,
        repetition: int = 1,
    ) -> EpisodeTrace:
        db = RetailDB.fresh()
        tools = RetailTools(db)
        trace = EpisodeTrace(
            task_id=task.id,
            prompt=task.prompt,
            model=model,
            system_prompt_variant=system_prompt_variant,
            repetition=repetition,
            before_db=db.snapshot(),
        )
        try:
            agent.start(task, system_prompt)
        except Exception as exc:
            return self._finish(trace, db, agent, "agent_error", f"start failed: {exc!r}")

        observation: Any | None = None
        for step in range(1, self.max_steps + 1):
            try:
                turn = agent.next_turn(observation)
            except Exception as exc:  # model or API failure; truncated/empty replies keep their own reason
                return self._finish(trace, db, agent, getattr(exc, "end_reason", "agent_error"), repr(exc))
            if turn.kind == "question":
                trace.asked_question = True
                trace.question_text = turn.text
                return self._finish(trace, db, agent, "question")
            if turn.kind == "final":
                trace.final_answer = turn.text
                return self._finish(trace, db, agent, "final")
            if turn.kind != "tool" or turn.tool_call is None:
                return self._finish(trace, db, agent, "agent_error", f"invalid agent turn: {turn!r}")

            call = turn.tool_call
            state_before = db.snapshot()
            result, error = None, None
            try:
                result = tools.execute(call.name, deepcopy(call.arguments))
                status = "succeeded"
            except InvalidToolCall as exc:
                status, error = "invalid", str(exc)
            except ToolError as exc:
                status, error = "rejected", str(exc)
            except Exception as exc:  # a bug in our tools, not agent behavior
                trace.tool_events.append(ToolEvent(
                    step=step, name=call.name, arguments=deepcopy(call.arguments), status="crashed",
                    error=repr(exc), state_changed=db.state != state_before,
                ))
                return self._finish(trace, db, agent, "harness_error", repr(exc))

            trace.tool_events.append(ToolEvent(
                step=step,
                name=call.name,
                arguments=deepcopy(call.arguments),
                status=status,
                result=deepcopy(result),
                error=error,
                state_changed=db.state != state_before,
            ))
            observation = result if status == "succeeded" else {"error": error}

        return self._finish(trace, db, agent, "max_steps")

    @staticmethod
    def _finish(trace: EpisodeTrace, db: RetailDB, agent: AgentBackend, reason: str,
                error: str | None = None) -> EpisodeTrace:
        trace.end_reason = reason
        trace.error = error
        trace.after_db = db.snapshot()
        try:
            trace.agent_metadata = agent.metadata()
        except Exception as exc:
            trace.agent_metadata = {"metadata_error": repr(exc)}
        return trace
