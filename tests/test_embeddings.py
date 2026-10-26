"""Embedding client and index naming tests (no network)."""

from __future__ import annotations

from types import SimpleNamespace

from openai.types import CreateEmbeddingResponse

from organizing_knowledge_for_agents.embeddings import Embedder
from organizing_knowledge_for_agents.index import collection_name


class FakeEmbeddingsClient:
    def __init__(self):
        self.requests: list[dict] = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        # Return items out of order: the embedder must sort by index.
        data = [{"object": "embedding", "index": i, "embedding": [float(i)]} for i in range(len(kwargs["input"]))]
        return CreateEmbeddingResponse.model_validate(
            {
                "object": "list",
                "model": kwargs["model"],
                "data": list(reversed(data)),
                "usage": {"prompt_tokens": 7, "total_tokens": 7},
            }
        )


def test_embed_sends_nim_fields_and_keeps_order():
    client = FakeEmbeddingsClient()
    embedder = Embedder("nvidia/nemotron-3-embed-1b", client=client)

    vectors = embedder.embed(["a", "b", "c"], input_type="passage")

    assert vectors == [[0.0], [1.0], [2.0]]
    assert client.requests[0]["extra_body"] == {"input_type": "passage", "truncate": "END"}
    assert client.requests[0]["model"] == "nvidia/nemotron-3-embed-1b"
    assert embedder.total_tokens == 7


def test_collection_name_is_chroma_safe():
    assert collection_name("m0", "nvidia/nemotron-3-embed-1b") == "confluence-m0-nvidia-nemotron-3-embed-1b"
