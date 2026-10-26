"""The agent loop, built on the OpenAI Responses API.

Uses the Responses API (not Chat Completions) and passes every output item back,
including reasoning items, so reasoning models keep their chain of thought across
tool calls. With store=False, reasoning only survives as encrypted_content,
which is why it is requested via `include`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from openai import OpenAI

from organizing_knowledge_for_agents.settings import AgentConfig
from organizing_knowledge_for_agents.tools import FINISH, ToolBox

SYSTEM_PROMPT = """\
You answer questions using a company's internal Confluence wiki. You can only learn about the wiki through your tools.

- Use `search` to find relevant pages. If results look off-target, rephrase and search again; internal codenames and acronyms often matter.
- Use `read_document` when you need the full text of a page.
- When you are done, call `finish` with your answer and the doc_ids of the pages that support it. Include every page you relied on and no others.
- If the wiki does not contain the answer, say so in `finish` instead of guessing.

You have a budget of {max_tool_calls} tool calls (not counting `finish`)."""

BUDGET_EXHAUSTED = "You have used your tool-call budget. Call `finish` now with your best answer and the supporting doc_ids."
NUDGE_FINISH = "Call `finish` to submit your answer and the supporting doc_ids."


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, usage) -> None:
        self.requests += 1
        if usage is None:
            return
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        if usage.input_tokens_details:
            self.cached_tokens += usage.input_tokens_details.cached_tokens or 0
        if usage.output_tokens_details:
            self.reasoning_tokens += usage.output_tokens_details.reasoning_tokens or 0


@dataclass
class AgentResult:
    answer: str
    doc_ids: list[str]
    finished_via: str  # finish | forced_finish | text | max_turns
    usage: Usage = field(default_factory=Usage)
    tool_calls: dict[str, int] = field(default_factory=dict)
    docs_seen: list[str] = field(default_factory=list)  # every doc whose content was shown, in order
    invalid_doc_ids: list[str] = field(default_factory=list)
    seconds: float = 0.0
    trace: list[dict] = field(default_factory=list)

    def metrics(self) -> dict:
        return {
            "finished_via": self.finished_via,
            "n_tool_calls": sum(self.tool_calls.values()),
            "tool_calls": self.tool_calls,
            "n_docs_seen": len(set(self.docs_seen)),
            "n_doc_ids": len(self.doc_ids),
            "invalid_doc_ids": self.invalid_doc_ids,
            "seconds": round(self.seconds, 2),
            **self.usage.__dict__,
        }


def _dump(item) -> dict:
    return item.model_dump(mode="json", exclude_unset=True, by_alias=True)


def _trace_item(item: dict) -> dict:
    # Encrypted reasoning is opaque and large; keep it out of traces.
    return {k: v for k, v in item.items() if k != "encrypted_content"}


def run_agent(client: OpenAI, cfg: AgentConfig, toolbox: ToolBox, question: str) -> AgentResult:
    start = time.monotonic()
    usage = Usage()
    tool_calls: dict[str, int] = {}
    docs_seen: list[str] = []
    trace: list[dict] = []
    items: list[dict] = [{"role": "user", "content": question}]
    budget_spent = nudged = False

    def result(answer: str, doc_ids: list[str], finished_via: str) -> AgentResult:
        valid = [d for d in dict.fromkeys(doc_ids) if d in toolbox.corpus]
        invalid = [d for d in doc_ids if d not in toolbox.corpus]
        return AgentResult(
            answer, valid, finished_via, usage, tool_calls, docs_seen, invalid, time.monotonic() - start, trace
        )

    # Each turn is one model request; the cap only guards against a model that never stops.
    for _ in range(cfg.max_tool_calls + 5):
        kwargs: dict = {
            "model": cfg.model,
            "instructions": SYSTEM_PROMPT.format(max_tool_calls=cfg.max_tool_calls),
            "input": items,
            "tools": toolbox.schemas,
            "store": False,
            "max_output_tokens": cfg.max_output_tokens,
        }
        if cfg.reasoning_effort:
            kwargs["reasoning"] = {"effort": cfg.reasoning_effort}
            kwargs["include"] = ["reasoning.encrypted_content"]
        if budget_spent:
            kwargs["tool_choice"] = {"type": "function", "name": FINISH}

        response = client.responses.create(**kwargs)
        usage.add(response.usage)
        output = [_dump(item) for item in response.output]
        items.extend(output)
        trace.extend(_trace_item(item) for item in output)

        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            if nudged:
                return result(response.output_text, [], "text")
            nudged = True
            items.append({"role": "user", "content": NUDGE_FINISH})
            continue

        for call in calls:
            if call.name == FINISH:
                try:
                    args = json.loads(call.arguments)
                except json.JSONDecodeError:
                    args = {"answer": call.arguments, "doc_ids": []}
                return result(args.get("answer", ""), args.get("doc_ids", []), "forced_finish" if budget_spent else FINISH)

            if sum(tool_calls.values()) >= cfg.max_tool_calls:
                # Parallel calls can overshoot the budget; every call still needs an output item.
                output_item = {"type": "function_call_output", "call_id": call.call_id, "output": BUDGET_EXHAUSTED}
                items.append(output_item)
                trace.append(output_item)
                continue

            tool_result = toolbox.execute(call.name, call.arguments)
            tool_calls[call.name] = tool_calls.get(call.name, 0) + 1
            docs_seen.extend(tool_result.doc_ids)
            output_item = {"type": "function_call_output", "call_id": call.call_id, "output": tool_result.output}
            items.append(output_item)
            trace.append({**output_item, "output": f"<{len(tool_result.output)} chars>", "doc_ids": tool_result.doc_ids})

        if sum(tool_calls.values()) >= cfg.max_tool_calls and not budget_spent:
            budget_spent = True
            items.append({"role": "user", "content": BUDGET_EXHAUSTED})

    return result("", [], "max_turns")
