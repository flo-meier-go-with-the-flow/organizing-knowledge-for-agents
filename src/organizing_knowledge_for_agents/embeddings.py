"""Embeddings from an NVIDIA NIM (Nemotron Embed) via its OpenAI-compatible /v1/embeddings API.

Nemotron Embed is asymmetric: index documents with input_type="passage" and
embed search queries with input_type="query". Inputs longer than the NIM's max
sequence length (4,096 tokens for nemotron-3-embed-1b on NIM 2.2) are cut at
the end via truncate="END"; about 2% of the Confluence pages are affected.
"""

from __future__ import annotations

from typing import Literal

from openai import OpenAI

from organizing_knowledge_for_agents.settings import EMBED_API_KEY, EMBED_BASE_URL

InputType = Literal["query", "passage"]


class Embedder:
    def __init__(self, model: str, client: OpenAI | None = None):
        if client is None:
            if not EMBED_BASE_URL:
                raise RuntimeError("Set EMBED_BASE_URL to the NIM endpoint, e.g. http://<host>:8000/v1")
            # Pass api_key explicitly: otherwise the OpenAI client falls back to OPENAI_API_KEY
            # and would send the agent's key to the embedding host.
            client = OpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY or "unused", max_retries=5)
        self.client = client
        self.model = model
        self.total_tokens = 0

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "END"},
        )
        if response.usage:
            self.total_tokens += response.usage.total_tokens
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
