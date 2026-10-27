"""Embed the corpus with OpenAI text-embedding-3-large and store it in a persistent Chroma collection.

Usage:
    uv run python -m organizing_knowledge_for_agents.index [--variant m0] [--model MODEL] [--batch-size 32] [--rebuild]

Resumable: documents already in the collection are skipped. BM25 is built in
memory at search time (5k docs index in a few seconds), so only dense
embeddings are persisted.
"""

from __future__ import annotations

import argparse
import re

import chromadb
from tqdm import tqdm

from organizing_knowledge_for_agents.corpus import chroma_metadata, load_corpus, render_text
from organizing_knowledge_for_agents.embeddings import Embedder
from organizing_knowledge_for_agents.settings import CHROMA_DIR, EMBED_MODEL


def collection_name(variant: str, model: str) -> str:
    # Chroma names allow only [a-zA-Z0-9._-]; some model or deployment IDs contain '/' or spaces.
    return re.sub(r"[^a-zA-Z0-9._-]", "-", f"confluence-{variant}-{model}")


def get_collection(variant: str, model: str, create: bool = False) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = collection_name(variant, model)
    if create:
        return client.get_or_create_collection(
            name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
        )
    return client.get_collection(name, embedding_function=None)


def build_index(variant: str, model: str, batch_size: int, rebuild: bool = False) -> None:
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

    embedder = Embedder(model)
    with tqdm(total=len(todo), unit="doc") as progress:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            texts = [render_text(doc, variant) for doc in batch]
            collection.add(
                ids=[doc.doc_id for doc in batch],
                embeddings=embedder.embed(texts),
                documents=texts,
                metadatas=[chroma_metadata(doc) for doc in batch],
            )
            progress.update(len(batch))
            progress.set_postfix(tokens=embedder.total_tokens)

    print(f"Collection {collection.name} now has {collection.count()} docs at {CHROMA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="m0", help="Document rendering to index (corpus.render_text)")
    parser.add_argument("--model", default=EMBED_MODEL, help="Embedding model (on Azure: the deployment name)")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Docs per request (OpenAI caps a request at 300k tokens in total)"
    )
    parser.add_argument("--rebuild", action="store_true", help="Drop and re-create the collection")
    args = parser.parse_args()
    build_index(args.variant, args.model, args.batch_size, args.rebuild)


if __name__ == "__main__":
    main()
