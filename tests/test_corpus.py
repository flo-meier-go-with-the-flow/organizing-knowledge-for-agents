"""Checks against the committed corpus (data/confluence) and question sets (data/questions)."""

from __future__ import annotations

from organizing_knowledge_for_agents.corpus import chroma_metadata, load_corpus
from organizing_knowledge_for_agents.questions import load_question_set
from organizing_knowledge_for_agents.search import Searcher
from organizing_knowledge_for_agents.settings import RetrievalConfig


def test_corpus_is_clean():
    corpus = load_corpus()
    assert len(corpus) == 5189
    for doc in corpus.values():
        assert "original_location" not in doc.metadata
        assert "dataset_noise_document" not in doc.metadata
        # The word itself appears in some page bodies; the leak would be a rendered field section.
        assert not doc.content.startswith("original_location:\n")
        assert "\n\noriginal_location:\n" not in doc.content
        meta = chroma_metadata(doc)
        assert all(v not in (None, "", []) for v in meta.values())


def test_question_sets():
    sizes = {name: len(load_question_set(name)) for name in ("confluence", "not_found", "metadata")}
    assert sizes == {"confluence": 64, "not_found": 20, "metadata": 12}

    corpus = load_corpus()
    questions = load_question_set("confluence")
    assert all(d in corpus for q in questions for d in q["expected_doc_ids"])


def test_bm25_search_finds_gold_doc():
    # A basic question should retrieve its gold doc lexically; no API keys needed.
    searcher = Searcher(RetrievalConfig(mode="bm25", k=10))
    questions = load_question_set("confluence")
    q = next(q for q in questions if q["question_type"] == "basic")
    hits = [hit.doc.doc_id for hit in searcher.search(q["question"])]
    assert q["expected_doc_ids"][0] in hits
