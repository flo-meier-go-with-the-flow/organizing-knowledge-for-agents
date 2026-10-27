"""Tools exposed to the agent, and how their results are rendered.

Grid conditions are expressed as ToolsConfig toggles so every cell runs the
same loop. The baseline is: search returns full documents, no metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from organizing_knowledge_for_agents.corpus import Doc
from organizing_knowledge_for_agents.search import Searcher
from organizing_knowledge_for_agents.settings import ToolsConfig

FINISH = "finish"


def _function(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        "strict": True,
    }


@dataclass
class ToolResult:
    output: str
    doc_ids: list[str] = field(default_factory=list)  # docs whose content the agent was shown


class ToolBox:
    def __init__(self, searcher: Searcher, cfg: ToolsConfig):
        self.searcher = searcher
        self.cfg = cfg
        self.corpus = searcher.corpus

    @property
    def schemas(self) -> list[dict]:
        search_returns = (
            "the full text of each matching page"
            if self.cfg.result_view == "full"
            else "a short preview of each matching page; use read_document for the full text"
        )
        return [
            _function(
                "search",
                f"Search the wiki. Returns the top {self.searcher.cfg.k} pages with {search_returns}.",
                {"query": {"type": "string", "description": "Search query"}},
            ),
            _function(
                "read_document",
                "Read the full text of one wiki page by its doc_id.",
                {"doc_id": {"type": "string", "description": "A doc_id from search results (dsid_...)"}},
            ),
            _function(
                FINISH,
                "Submit the final answer and the doc_ids of every page that supports it. Ends the task.",
                {
                    "answer": {"type": "string", "description": "The final answer to the question"},
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "doc_ids of the pages the answer relies on",
                    },
                },
            ),
        ]

    def _render(self, doc: Doc, full: bool) -> str:
        lines = [f"doc_id: {doc.doc_id}", f"title: {doc.title}"]
        if self.cfg.show_metadata:
            for key, value in doc.metadata.items():
                lines.append(f"{key}: {', '.join(value) if isinstance(value, list) else value}")
        body = doc.content if full else doc.content[: self.cfg.preview_chars].rstrip() + " ..."
        return "\n".join(lines) + "\n---\n" + body

    def execute(self, name: str, arguments: str) -> ToolResult:
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError as e:
            return ToolResult(f"Error: arguments are not valid JSON ({e})")

        if name == "search":
            hits = self.searcher.search(args["query"])
            if not hits:
                return ToolResult("No results.")
            full = self.cfg.result_view == "full"
            blocks = [f"[{i + 1}] {self._render(hit.doc, full)}" for i, hit in enumerate(hits)]
            return ToolResult("\n\n".join(blocks), [hit.doc.doc_id for hit in hits])

        if name == "read_document":
            doc = self.corpus.get(args["doc_id"])
            if doc is None:
                return ToolResult(f"Error: unknown doc_id {args['doc_id']!r}")
            return ToolResult(self._render(doc, full=True), [doc.doc_id])

        return ToolResult(f"Error: unknown tool {name!r}")
