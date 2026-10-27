"""Embeddings from OpenAI text-embedding-3-large (or a company/Azure deployment of it).

The model is symmetric, so documents and queries are embedded the same way.
Its per-input limit is 8,191 tokens; the longest Confluence page is about
6,600 tokens, so nothing is truncated.
"""

from __future__ import annotations

from openai import OpenAI

from organizing_knowledge_for_agents.settings import EMBED_API_KEY, EMBED_BASE_URL


class Embedder:
    def __init__(self, model: str, client: OpenAI | None = None):
        # Unset EMBED_* values fall back to OPENAI_BASE_URL / OPENAI_API_KEY, the agent's endpoint.
        self.client = client or OpenAI(base_url=EMBED_BASE_URL or None, api_key=EMBED_API_KEY or None, max_retries=5)
        self.model = model
        self.total_tokens = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts, encoding_format="float")
        if response.usage:
            self.total_tokens += response.usage.total_tokens
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
