"""Hybrid search: OpenAI-embedding dense retrieval (Chroma) + BM25, fused with reciprocal rank fusion.

Metadata can enter in three ways, each a RetrievalConfig/ToolsConfig toggle:
- filters: a Chroma `where` clause restricts both retrievers to the matching pages
- metadata_in_bm25: BM25 indexes each page's metadata next to its text
- recency_weight: a newest-first ranking of the top candidates is fused in (recency pushdown)
"""

from __future__ import annotations

from dataclasses import dataclass

import bm25s
import numpy as np

from organizing_knowledge_for_agents.corpus import Doc, date_to_int, load_corpus, metadata_text, render_text
from organizing_knowledge_for_agents.embeddings import Embedder
from organizing_knowledge_for_agents.index import get_collection
from organizing_knowledge_for_agents.settings import RetrievalConfig


@dataclass
class Hit:
    doc: Doc
    score: float


def fuse(rankings: list[tuple[float, list[str]]], rrf_k: int) -> dict[str, float]:
    """Weighted reciprocal rank fusion of (weight, ranked doc_ids) pairs."""
    fused: dict[str, float] = {}
    for weight, ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (rrf_k + rank + 1)
    return fused


def recency_ranking(fused: dict[str, float], updated: dict[str, int], depth: int) -> list[str]:
    """The top `depth` fused candidates that have a date, newest first."""
    top = sorted(fused, key=fused.get, reverse=True)[:depth]
    return sorted((d for d in top if d in updated), key=updated.get, reverse=True)


class Searcher:
    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self.corpus = load_corpus(cfg.corpus)
        self.doc_ids = list(self.corpus)
        self.position = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        self.updated = {
            doc_id: as_int
            for doc_id, doc in self.corpus.items()
            if (as_int := date_to_int(doc.metadata.get("last_updated", ""))) is not None
        }

        if cfg.mode in ("hybrid", "bm25"):
            texts = [self._bm25_text(self.corpus[d]) for d in self.doc_ids]
            self.bm25 = bm25s.BM25()
            self.bm25.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)

        self.collection = None
        if cfg.mode in ("hybrid", "dense"):
            self.collection = get_collection(cfg.corpus, cfg.text_variant, cfg.embedding_model)
            if self.collection.count() != len(self.corpus):
                raise RuntimeError(
                    f"Chroma collection {self.collection.name} has {self.collection.count()} docs, "
                    f"corpus has {len(self.corpus)}. Run `python -m organizing_knowledge_for_agents.index "
                    f"--corpus {cfg.corpus}` first."
                )
            # Exact search over all vectors instead of Chroma's HNSW, which returns only ~95% of the exact
            # top 50 here and builds a different graph per collection: corpora with identical vectors
            # would rank differently. 5k pages take a few milliseconds per query.
            stored = self.collection.get(include=["embeddings"])
            row = {doc_id: i for i, doc_id in enumerate(stored["ids"])}
            self.vectors = np.asarray([stored["embeddings"][row[d]] for d in self.doc_ids], dtype=np.float32)
            self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)
            self.embedder = Embedder(cfg.embedding_model)

    def _bm25_text(self, doc: Doc) -> str:
        text = render_text(doc, self.cfg.text_variant)
        if self.cfg.metadata_in_bm25 and doc.metadata:
            text += "\n\n" + metadata_text(doc)
        return text

    def matching_ids(self, where: dict) -> set[str]:
        """Pages whose Chroma metadata matches a `where` clause."""
        if self.collection is None:
            raise RuntimeError("Metadata filters need the Chroma collection (retrieval.mode hybrid or dense)")
        return set(self.collection.get(where=where, include=[])["ids"])

    def _top(self, scores: np.ndarray, allowed: np.ndarray | None) -> list[str]:
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        top = np.argsort(-scores, kind="stable")[: self.cfg.candidates]
        return [self.doc_ids[i] for i in top if np.isfinite(scores[i])]

    def _bm25_ranking(self, query: str, allowed: np.ndarray | None) -> list[str]:
        tokens = bm25s.tokenize(query, stopwords="en", return_ids=False, show_progress=False)[0]
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        return self._top(np.where(scores > 0, scores, -np.inf), allowed)

    def _dense_ranking(self, query: str, allowed: np.ndarray | None) -> list[str]:
        embedding = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        return self._top(self.vectors @ (embedding / np.linalg.norm(embedding)), allowed)

    def search(self, query: str, k: int | None = None, where: dict | None = None) -> list[Hit]:
        k = k or self.cfg.k
        allowed = None
        if where:
            matching = self.matching_ids(where)
            if not matching:
                return []
            allowed = np.zeros(len(self.doc_ids), dtype=bool)
            allowed[[self.position[d] for d in matching]] = True

        rankings = []
        if self.cfg.mode in ("hybrid", "bm25"):
            rankings.append((1.0, self._bm25_ranking(query, allowed)))
        if self.cfg.mode in ("hybrid", "dense"):
            rankings.append((1.0, self._dense_ranking(query, allowed)))
        fused = fuse(rankings, self.cfg.rrf_k)

        if self.cfg.recency_weight and self.updated:
            newest_first = recency_ranking(fused, self.updated, self.cfg.recency_depth)
            for doc_id, score in fuse([(self.cfg.recency_weight, newest_first)], self.cfg.rrf_k).items():
                fused[doc_id] += score

        best = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
        return [Hit(self.corpus[doc_id], score) for doc_id, score in best]
