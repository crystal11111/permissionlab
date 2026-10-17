from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class Authority(str, Enum):
    """What the operator's instruction authorizes for one action.

    Task authority only. Whether the organization additionally requires a
    confirmation before the action runs is a separate question, answered by
    the consequence policy (policy.py). A refund the operator explicitly asked
    for is AUTHORIZED here even if policy still wants an approval click.
    """

    AUTHORIZED = "AUTHORIZED"          # the instruction asks for it, within its constraints
    AMBIGUOUS = "AMBIGUOUS"            # plausibly intended; a careful assistant asks the operator first
    NOT_AUTHORIZED = "NOT_AUTHORIZED"  # the instruction does not cover it


Label = Authority  # older name, kept so existing imports keep working


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentTurn:
    kind: Literal["tool", "question", "final"]
    tool_call: ToolCall | None = None
    text: str | None = None

    @staticmethod
    def tool(name: str, **arguments: Any) -> "AgentTurn":
        return AgentTurn(kind="tool", tool_call=ToolCall(name, arguments))

    @staticmethod
    def question(text: str) -> "AgentTurn":
        return AgentTurn(kind="question", text=text)

    @staticmethod
    def final(text: str) -> "AgentTurn":
        return AgentTurn(kind="final", text=text)


# succeeded: the tool ran. rejected: the tool refused for a business reason
# (unknown order, refund too large). invalid: unknown tool or malformed arguments.
# crashed: the harness itself raised; the episode ends with harness_error.
ToolStatus = Literal["succeeded", "rejected", "invalid", "crashed"]

# final: text answer. question: the agent asked the operator (ends the episode
# in this single-instruction protocol). max_steps: step limit hit.
# agent_error: the model/API failed. harness_error: a bug in our code.
EndReason = Literal["final", "question", "max_steps", "agent_error", "harness_error"]


@dataclass
class ToolEvent:
    step: int
    name: str
    arguments: dict[str, Any]
    status: str
    result: Any = None
    error: str | None = None
    state_changed: bool = False


@dataclass
class EpisodeTrace:
    task_id: str
    prompt: str
    model: str
    system_prompt_variant: str
    repetition: int
    tool_events: list[ToolEvent] = field(default_factory=list)
    asked_question: bool = False
    question_text: str | None = None
    final_answer: str | None = None
    end_reason: str = "final"
    error: str | None = None
    before_db: dict[str, Any] = field(default_factory=dict)
    after_db: dict[str, Any] = field(default_factory=dict)
    agent_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeTrace":
        fields = dict(data)
        fields["tool_events"] = [ToolEvent(**e) for e in data.get("tool_events", [])]
        return cls(**fields)


@dataclass(frozen=True)
class ActionRule:
    """One task's authority for one action, with optional constraints.

    Per-call constraints: eq_<arg>, in_<arg> (list), max_<arg> (numbers).
    Cumulative across the episode: max_uses, max_total_<arg> (numbers).
    """

    action: str
    authority: Authority
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    id: str
    category: str
    prompt: str
    context: dict[str, Any]
    # Explicit-execution tasks only: effects that must happen for the task to count
    # as done, e.g. {"action": "issue_refund", "args": {"order_id": "#W123", "amount": 30}}.
    expected_effects: list[dict[str, Any]] = field(default_factory=list)
