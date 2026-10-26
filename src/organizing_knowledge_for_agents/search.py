"""Hybrid search: Nemotron dense retrieval (Chroma) + BM25, fused with reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass

import bm25s
import numpy as np

from organizing_knowledge_for_agents.corpus import Doc, load_corpus, render_text
from organizing_knowledge_for_agents.embeddings import Embedder
from organizing_knowledge_for_agents.index import get_collection
from organizing_knowledge_for_agents.settings import RetrievalConfig


@dataclass
class Hit:
    doc: Doc
    score: float


class Searcher:
    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self.corpus = load_corpus()
        self.doc_ids = list(self.corpus)

        if cfg.mode in ("hybrid", "bm25"):
            texts = [render_text(self.corpus[d], cfg.text_variant) for d in self.doc_ids]
            self.bm25 = bm25s.BM25()
            self.bm25.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)

        if cfg.mode in ("hybrid", "dense"):
            self.collection = get_collection(cfg.text_variant, cfg.embedding_model)
            if self.collection.count() != len(self.corpus):
                raise RuntimeError(
                    f"Chroma collection {self.collection.name} has {self.collection.count()} docs, "
                    f"corpus has {len(self.corpus)}. Run `python -m organizing_knowledge_for_agents.index` first."
                )
            self.embedder = Embedder(cfg.embedding_model)

    def _bm25_ranking(self, query: str) -> list[str]:
        tokens = bm25s.tokenize(query, stopwords="en", return_ids=False, show_progress=False)[0]
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = np.argsort(-scores)[: self.cfg.candidates]
        return [self.doc_ids[i] for i in top if scores[i] > 0]

    def _dense_ranking(self, query: str) -> list[str]:
        embedding = self.embedder.embed([query], input_type="query")[0]
        result = self.collection.query(query_embeddings=[embedding], n_results=self.cfg.candidates, include=[])
        return result["ids"][0]

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or self.cfg.k
        rankings = []
        if self.cfg.mode in ("hybrid", "bm25"):
            rankings.append(self._bm25_ranking(query))
        if self.cfg.mode in ("hybrid", "dense"):
            rankings.append(self._dense_ranking(query))

        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.cfg.rrf_k + rank + 1)
        best = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
        return [Hit(self.corpus[doc_id], score) for doc_id, score in best]
