"""Embed the corpus with Voyage and store it in a persistent Chroma collection.

Usage:
    uv run python -m organizing_knowledge_for_agents.index [--variant m0] [--model voyage-4-large] [--rebuild]

Resumable: documents already in the collection are skipped. BM25 is built in
memory at search time (5k docs index in a few seconds), so only dense
embeddings are persisted.
"""

from __future__ import annotations

import argparse

import chromadb
import voyageai
from tqdm import tqdm

from organizing_knowledge_for_agents.corpus import Doc, chroma_metadata, load_corpus, render_text
from organizing_knowledge_for_agents.settings import CHROMA_DIR, VOYAGE_MODEL

# Voyage caps tokens per request; batching by characters keeps us well under it
# without a tokenizer round-trip (~4 chars per token).
MAX_BATCH_CHARS = 300_000
MAX_BATCH_DOCS = 64


def collection_name(variant: str, model: str) -> str:
    return f"confluence-{variant}-{model}"


def get_collection(variant: str, model: str, create: bool = False) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = collection_name(variant, model)
    if create:
        return client.get_or_create_collection(
            name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
        )
    return client.get_collection(name, embedding_function=None)


def _batches(docs: list[Doc], variant: str):
    batch, chars = [], 0
    for doc in docs:
        text = render_text(doc, variant)
        if batch and (chars + len(text) > MAX_BATCH_CHARS or len(batch) >= MAX_BATCH_DOCS):
            yield batch
            batch, chars = [], 0
        batch.append((doc, text))
        chars += len(text)
    if batch:
        yield batch


def build_index(variant: str, model: str, rebuild: bool = False) -> None:
    if rebuild:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        name = collection_name(variant, model)
        if name in [c.name for c in client.list_collections()]:
            client.delete_collection(name)

    collection = get_collection(variant, model, create=True)
    corpus = load_corpus()
    existing = set(collection.get(include=[])["ids"]) if collection.count() else set()
    todo = [doc for doc in corpus.values() if doc.doc_id not in existing]
    print(f"{len(corpus)} docs, {len(existing)} already indexed, {len(todo)} to embed with {model}")

    vo = voyageai.Client(max_retries=5)
    with tqdm(total=len(todo), unit="doc") as progress:
        for batch in _batches(todo, variant):
            result = vo.embed([text for _, text in batch], model=model, input_type="document")
            collection.add(
                ids=[doc.doc_id for doc, _ in batch],
                embeddings=result.embeddings,
                documents=[text for _, text in batch],
                metadatas=[chroma_metadata(doc) for doc, _ in batch],
            )
            progress.update(len(batch))
            progress.set_postfix(tokens=result.total_tokens)

    print(f"Collection {collection.name} now has {collection.count()} docs at {CHROMA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="m0", help="Document rendering to index (corpus.render_text)")
    parser.add_argument("--model", default=VOYAGE_MODEL, help="Voyage embedding model")
    parser.add_argument("--rebuild", action="store_true", help="Drop and re-create the collection")
    args = parser.parse_args()
    build_index(args.variant, args.model, args.rebuild)


if __name__ == "__main__":
    main()
