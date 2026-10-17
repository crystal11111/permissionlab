from permissionlab.agent import NEUTRAL_SYSTEM_PROMPT, AgentBackend
from permissionlab.runner import EpisodeRunner
from permissionlab.types import AgentTurn, TaskSpec
from tests.helpers import run


def test_question_ends_episode():
    trace = run([AgentTurn.question("Should I issue a refund?")])
    assert trace.asked_question
    assert trace.end_reason == "question"
    assert trace.tool_events == []


def test_tool_error_keeps_earlier_actions_in_the_trace():
    # Review repro: refund succeeds, then cancelling a delivered order raises ToolError.
    trace = run([
        AgentTurn.tool("issue_refund", user_id="u_42", order_id="#W123", amount=30),
        AgentTurn.tool("cancel_order", order_id="#W123"),
        AgentTurn.final("Done"),
    ])
    assert [(e.name, e.status) for e in trace.tool_events] == [("issue_refund", "succeeded"),
                                                               ("cancel_order", "rejected")]
    assert "Only pending orders" in trace.tool_events[1].error
    assert trace.tool_events[1].state_changed is False
    assert trace.end_reason == "final"
    assert trace.after_db["orders"]["#W123"]["refund_total"] == 30


def test_invalid_call_is_recorded_and_returned_to_the_agent():
    seen = []

    class Agent(AgentBackend):
        def start(self, task, system_prompt): self.i = 0
        def next_turn(self, observation):
            seen.append(observation)
            self.i += 1
            return AgentTurn.tool("issue_refund", order_id="#W123") if self.i == 1 else AgentTurn.final("ok")

    trace = EpisodeRunner().run(TaskSpec("T", "x", "p", {}), Agent(), model="m", system_prompt="s",
                                system_prompt_variant="neutral")
    assert trace.tool_events[0].status == "invalid"
    assert "missing" in seen[1]["error"]


def test_agent_error_preserves_what_already_happened():
    class Flaky(AgentBackend):
        def start(self, task, system_prompt): self.i = 0
        def next_turn(self, observation):
            self.i += 1
            if self.i == 1:
                return AgentTurn.tool("issue_refund", user_id="u_42", order_id="#W123", amount=30)
            raise TimeoutError("API timed out")

    trace = EpisodeRunner().run(TaskSpec("T", "x", "p", {}), Flaky(), model="m", system_prompt="s",
                                system_prompt_variant="neutral")
    assert trace.end_reason == "agent_error"
    assert "API timed out" in trace.error
    assert trace.tool_events[0].status == "succeeded"
    assert trace.after_db["refunds"][0]["amount"] == 30


def test_max_steps_is_recorded():
    trace = EpisodeRunner(max_steps=2).run(
        TaskSpec("T", "x", "p", {}), _Looping(), model="m", system_prompt="s", system_prompt_variant="neutral")
    assert trace.end_reason == "max_steps"
    assert len(trace.tool_events) == 2


class _Looping(AgentBackend):
    def start(self, task, system_prompt): ...
    def next_turn(self, observation): return AgentTurn.tool("get_order", order_id="#W123")
