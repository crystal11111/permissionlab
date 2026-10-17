import json

from permissionlab.agent import AgentError, OpenAICompatibleAgent, looks_like_question
from permissionlab.runner import EpisodeRunner
from tests.helpers import raises, task


def response(content=None, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def call(call_id, name, args):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


class FakeAPI:
    def __init__(self, responses):
        self.responses, self.bodies = list(responses), []

    def __call__(self, body):
        self.bodies.append(body)
        return self.responses.pop(0)


def test_parallel_tool_calls_run_in_order_and_results_go_back_together():
    api = FakeAPI([
        response(tool_calls=[call("a", "get_order", {"order_id": "#W123"}),
                             call("b", "get_customer", {"user_id": "u_42"})]),
        response("The keyboard has broken keys. I recommend a replacement."),
    ])
    agent = OpenAICompatibleAgent(model="test-model", transport=api, temperature=0.6)
    trace = EpisodeRunner().run(task(), agent, model="nano", system_prompt="sys", system_prompt_variant="neutral")
    assert [e.name for e in trace.tool_events] == ["get_order", "get_customer"]
    assert trace.end_reason == "final"
    second = api.bodies[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool", "tool"]
    assert [m["tool_call_id"] for m in second[3:]] == ["a", "b"]
    assert api.bodies[0]["temperature"] == 0.6 and len(api.bodies[0]["tools"]) == 8
    meta = trace.agent_metadata
    assert meta["usage"] == {"prompt_tokens": 20, "completion_tokens": 10} and meta["requests"] == 2
    assert meta["system_prompt_sha256"]


def test_unparseable_arguments_become_an_invalid_call():
    api = FakeAPI([response(tool_calls=[call("a", "issue_refund", "{not json")]), response("Sorry.")])
    trace = EpisodeRunner().run(task(), OpenAICompatibleAgent(model="m", transport=api), model="m",
                                system_prompt="s", system_prompt_variant="neutral")
    assert trace.tool_events[0].status == "invalid"
    assert "error" in json.loads(api.bodies[1]["messages"][-1]["content"])


def test_text_reply_ending_in_a_question_is_a_question():
    api = FakeAPI([response("I found the issue. Should I issue a $30 refund?")])
    trace = EpisodeRunner().run(task(), OpenAICompatibleAgent(model="m", transport=api), model="m",
                                system_prompt="s", system_prompt_variant="neutral")
    assert trace.end_reason == "question"
    assert looks_like_question("Done.") is False


def test_api_failure_is_an_agent_error_with_trace():
    def broken(body):
        raise AgentError("HTTP 500")
    trace = EpisodeRunner().run(task(), OpenAICompatibleAgent(model="m", transport=broken), model="m",
                                system_prompt="s", system_prompt_variant="neutral")
    assert trace.end_reason == "agent_error" and "HTTP 500" in trace.error


def test_missing_api_key_fails_clearly():
    import os
    agent = OpenAICompatibleAgent(model="m", api_key_env="PERMISSIONLAB_TEST_NO_SUCH_KEY")
    os.environ.pop("PERMISSIONLAB_TEST_NO_SUCH_KEY", None)
    with raises(AgentError, match="is not set"):
        agent._post({})
