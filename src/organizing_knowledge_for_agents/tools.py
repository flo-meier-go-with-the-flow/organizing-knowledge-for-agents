"""Tools exposed to the agent, and how their results are rendered.

Grid conditions are expressed as ToolsConfig toggles so every cell runs the
same loop. The baseline is: search returns full documents, no metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from organizing_knowledge_for_agents.corpus import Doc, date_to_int
from organizing_knowledge_for_agents.search import Searcher
from organizing_knowledge_for_agents.settings import ToolsConfig

FINISH = "finish"

# Search filters offered with tools.metadata_filters, for the fields the corpus has. Closed
# vocabularies become enums so the agent can't invent values; list fields match if they contain it.
ENUM_FILTERS = (
    "space",
    "status",
    "owner_team",
    "confidentiality",
    "document_type",
    "doc_classification",
    "lifecycle_status",
    "volatility",
    "data_classification",
)
CONTAINS_FILTERS = ("audience_groups", "product_area")
DATE_FILTERS = {"updated_after": "$gte", "updated_before": "$lte"}  # on last_updated


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


def _field_values(corpus: dict[str, Doc]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for doc in corpus.values():
        for key, value in doc.metadata.items():
            values.setdefault(key, set()).update(value if isinstance(value, list) else [value])
    return values


class ToolBox:
    def __init__(self, searcher: Searcher, cfg: ToolsConfig):
        self.searcher = searcher
        self.cfg = cfg
        self.corpus = searcher.corpus
        self.filters = self._filter_properties() if cfg.metadata_filters else {}

    def _filter_properties(self) -> dict:
        values = _field_values(self.corpus)
        self.authors = {name.lower(): name for name in values.get("author", ())}
        props = {}
        for key in ENUM_FILTERS + CONTAINS_FILTERS:
            if key in values:
                verb = "includes" if key in CONTAINS_FILTERS else "is"
                props[key] = {
                    "type": ["string", "null"],
                    "enum": [*sorted(values[key]), None],
                    "description": f"Only pages whose {key} {verb} this value",
                }
        if "author" in values:
            props["author"] = {"type": ["string", "null"], "description": "Only pages by this author (full name)"}
        if "last_updated" in values:
            for arg, op in DATE_FILTERS.items():
                when = "on or after" if op == "$gte" else "on or before"
                props[arg] = {
                    "type": ["string", "null"],
                    "description": f"Only pages last updated {when} this date (YYYY-MM-DD)",
                }
        return props

    def _where(self, args: dict) -> dict | str | None:
        """Chroma `where` clause for the filter arguments; a string is an error message for the agent."""
        clauses = []
        for key in self.filters:
            value = args.get(key)
            if not value:
                continue
            if key in CONTAINS_FILTERS:
                clauses.append({key: {"$contains": value}})
            elif key == "author":
                author = self.authors.get(value.strip().lower())
                if author is None:
                    return f"No page has the author {value!r}."
                clauses.append({"author": author})
            elif key in DATE_FILTERS:
                as_int = date_to_int(value)
                if as_int is None:
                    return f"Error: {key} must be a date (YYYY-MM-DD), got {value!r}."
                clauses.append({"last_updated": {DATE_FILTERS[key]: as_int}})
            else:
                clauses.append({key: value})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    @property
    def schemas(self) -> list[dict]:
        search_returns = (
            "the full text of each matching page"
            if self.cfg.result_view == "full"
            else "a short preview of each matching page; use read_document for the full text"
        )
        search_description = f"Search the wiki. Returns the top {self.searcher.cfg.k} pages with {search_returns}."
        if self.filters:
            search_description += " Optional filters restrict results to pages whose metadata matches; use null for filters you don't need."
        return [
            _function(
                "search",
                search_description,
                {"query": {"type": "string", "description": "Search query"}, **self.filters},
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
            where = self._where(args)
            if isinstance(where, str):
                return ToolResult(where)
            header = ""
            if where:
                active = ", ".join(f"{key}={args[key]}" for key in self.filters if args.get(key))
                n_matching = len(self.searcher.matching_ids(where))
                if not n_matching:
                    return ToolResult(f"No pages match the filters ({active}).")
                header = f"Filters ({active}) match {n_matching} pages.\n\n"
            hits = self.searcher.search(args["query"], where=where)
            if not hits:
                return ToolResult(header + "No results.")
            full = self.cfg.result_view == "full"
            blocks = [f"[{i + 1}] {self._render(hit.doc, full)}" for i, hit in enumerate(hits)]
            return ToolResult(header + "\n\n".join(blocks), [hit.doc.doc_id for hit in hits])

        if name == "read_document":
            doc = self.corpus.get(args["doc_id"])
            if doc is None:
                return ToolResult(f"Error: unknown doc_id {args['doc_id']!r}")
            return ToolResult(self._render(doc, full=True), [doc.doc_id])

        return ToolResult(f"Error: unknown tool {name!r}")
