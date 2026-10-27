"""Metadata in search: fusion, recency pushdown, BM25 metadata text and filters (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from organizing_knowledge_for_agents.corpus import Doc, metadata_text
from organizing_knowledge_for_agents.search import fuse, recency_ranking
from organizing_knowledge_for_agents.settings import ToolsConfig
from organizing_knowledge_for_agents.tools import ToolBox

DOCS = {
    "dsid_old": Doc("dsid_old", "PTO policy 2025", "...", "people-ops/pto-2025.json",
                    {"space": "people-ops", "author": "Ana Ruiz", "last_updated": "2025-02-03",
                     "audience_groups": ["all_employees"], "related_pages": ["/confluence/x"]}),
    "dsid_new": Doc("dsid_new", "PTO policy 2026", "...", "people-ops/pto-2026.json",
                    {"space": "people-ops", "author": "Ben Ode", "last_updated": "2026-09-12"}),
}


def test_recency_pushdown_lifts_the_newer_lookalike():
    fused = fuse([(1.0, ["dsid_old", "dsid_new"]), (1.0, ["dsid_old", "dsid_new"])], rrf_k=60)
    newest_first = recency_ranking(fused, {"dsid_old": 20250203, "dsid_new": 20260912}, depth=20)
    assert newest_first == ["dsid_new", "dsid_old"]
    for doc_id, score in fuse([(1.0, newest_first)], rrf_k=60).items():
        fused[doc_id] += score
    # Recency is one more vote: it doesn't overturn two retrievers that agree...
    assert fused["dsid_new"] < fused["dsid_old"]
    # ...but it decides between two that disagree.
    fused = fuse([(1.0, ["dsid_old", "dsid_new"]), (1.0, ["dsid_new", "dsid_old"]), (1.0, newest_first)], rrf_k=60)
    assert fused["dsid_new"] > fused["dsid_old"]


def test_recency_ignores_undated_pages():
    assert recency_ranking({"a": 0.3, "b": 0.2}, {"b": 20260101}, depth=20) == ["b"]


def test_metadata_text_spells_out_dates_and_skips_links():
    text = metadata_text(DOCS["dsid_old"])
    assert "last updated: 2025-02-03 February 2025" in text
    assert "audience groups: all employees" in text
    assert "/confluence/x" not in text


def toolbox(metadata_filters: bool = True) -> ToolBox:
    searcher = SimpleNamespace(corpus=DOCS, cfg=SimpleNamespace(k=5))
    return ToolBox(searcher, ToolsConfig(metadata_filters=metadata_filters))


def test_filters_follow_the_corpus_fields():
    props = toolbox().schemas[0]["parameters"]["properties"]
    assert set(props) == {"query", "space", "audience_groups", "author", "updated_after", "updated_before"}
    assert props["space"]["enum"] == ["people-ops", None]
    assert set(toolbox(metadata_filters=False).schemas[0]["parameters"]["properties"]) == {"query"}


def test_filters_become_a_chroma_where_clause():
    tb = toolbox()
    args = {"query": "pto", "space": "people-ops", "author": "ana ruiz", "audience_groups": "all_employees",
            "updated_after": "2025-01-01", "updated_before": None}
    assert tb._where(args) == {"$and": [
        {"space": "people-ops"},
        {"audience_groups": {"$contains": "all_employees"}},
        {"author": "Ana Ruiz"},
        {"last_updated": {"$gte": 20250101}},
    ]}
    assert tb._where({"query": "pto", "space": None}) is None
    assert tb._where({"query": "pto", "author": "Nobody"}) == "No page has the author 'Nobody'."
    assert tb._where({"query": "pto", "updated_after": "last spring"}).startswith("Error: updated_after")


def test_search_reports_filter_matches():
    calls = []

    class FakeSearcher:
        corpus = DOCS
        cfg = SimpleNamespace(k=5)

        def matching_ids(self, where):
            return {"dsid_new"} if where == {"space": "people-ops"} else set()

        def search(self, query, where=None):
            calls.append(where)
            return [SimpleNamespace(doc=DOCS["dsid_new"])]

    tb = ToolBox(FakeSearcher(), ToolsConfig(metadata_filters=True))
    result = tb.execute("search", json.dumps({"query": "pto", "space": "people-ops"}))
    assert result.output.startswith("Filters (space=people-ops) match 1 pages.")
    assert result.doc_ids == ["dsid_new"] and calls == [{"space": "people-ops"}]
    empty = tb.execute("search", json.dumps({"query": "pto", "author": "Ben Ode"}))
    assert empty.output == "No pages match the filters (author=Ben Ode)." and empty.doc_ids == []
