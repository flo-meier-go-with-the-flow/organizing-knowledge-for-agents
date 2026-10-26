"""Agent loop tests against a scripted fake OpenAI client (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from openai.types.responses import ResponseFunctionToolCall, ResponseReasoningItem, ResponseUsage

from organizing_knowledge_for_agents.agent import BUDGET_EXHAUSTED, NUDGE_FINISH, run_agent
from organizing_knowledge_for_agents.settings import AgentConfig
from organizing_knowledge_for_agents.tools import FINISH, ToolResult

USAGE = ResponseUsage.model_validate(
    {
        "input_tokens": 100,
        "input_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 0},
        "output_tokens": 10,
        "output_tokens_details": {"reasoning_tokens": 5},
        "total_tokens": 110,
    }
)


def call(name: str, args: dict, call_id: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(type="function_call", name=name, arguments=json.dumps(args), call_id=call_id)


def response(*output, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(output=list(output), usage=USAGE, output_text=text)


class FakeClient:
    def __init__(self, responses):
        self.queue = list(responses)
        self.requests: list[dict] = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "input": list(kwargs["input"])})
        return self.queue.pop(0)


class FakeToolBox:
    corpus = {"dsid_a": object(), "dsid_b": object()}
    schemas: list[dict] = []

    def execute(self, name, arguments):
        return ToolResult(f"results for {name}", ["dsid_a"])


def test_search_then_finish():
    reasoning = ResponseReasoningItem(type="reasoning", id="rs_1", summary=[], encrypted_content="opaque")
    client = FakeClient(
        [
            response(reasoning, call("search", {"query": "pto policy"}, "c1")),
            response(call(FINISH, {"answer": "20 days", "doc_ids": ["dsid_a", "dsid_a", "dsid_zzz"]}, "c2")),
        ]
    )
    result = run_agent(client, AgentConfig(), FakeToolBox(), "How much PTO?")

    assert result.answer == "20 days"
    assert result.doc_ids == ["dsid_a"]
    assert result.invalid_doc_ids == ["dsid_zzz"]
    assert result.finished_via == "finish"
    assert result.tool_calls == {"search": 1}
    assert result.usage.requests == 2 and result.usage.cached_tokens == 80

    second_input = client.requests[1]["input"]
    assert second_input[1]["type"] == "reasoning" and second_input[1]["encrypted_content"] == "opaque"
    assert second_input[-1] == {"type": "function_call_output", "call_id": "c1", "output": "results for search"}
    assert client.requests[0]["include"] == ["reasoning.encrypted_content"]
    assert all("encrypted_content" not in item for item in result.trace)


def test_budget_forces_finish():
    client = FakeClient(
        [
            response(call("search", {"query": "a"}, "c1")),
            # Parallel calls overshoot the budget of 2: the third must not execute.
            response(call("search", {"query": "b"}, "c2"), call("search", {"query": "c"}, "c3")),
            response(call(FINISH, {"answer": "best guess", "doc_ids": []}, "c4")),
        ]
    )
    result = run_agent(client, AgentConfig(max_tool_calls=2), FakeToolBox(), "q")

    assert result.tool_calls == {"search": 2}
    assert result.finished_via == "forced_finish"
    last = client.requests[2]
    assert last["tool_choice"] == {"type": "function", "name": FINISH}
    outputs = [i for i in last["input"] if i.get("type") == "function_call_output"]
    assert outputs[-1] == {"type": "function_call_output", "call_id": "c3", "output": BUDGET_EXHAUSTED}
    assert last["input"][-1] == {"role": "user", "content": BUDGET_EXHAUSTED}


def test_text_answer_is_nudged_once():
    client = FakeClient([response(text="It is 20 days."), response(text="Still 20 days.")])
    result = run_agent(client, AgentConfig(reasoning_effort=""), FakeToolBox(), "q")

    assert result.finished_via == "text"
    assert result.answer == "Still 20 days."
    assert client.requests[1]["input"][-1] == {"role": "user", "content": NUDGE_FINISH}
    assert "reasoning" not in client.requests[0]
