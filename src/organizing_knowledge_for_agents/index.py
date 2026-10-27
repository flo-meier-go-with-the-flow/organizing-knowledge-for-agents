"""Embed the corpus with OpenAI text-embedding-3-large and store it in a persistent Chroma collection.

Usage:
    uv run python -m organizing_knowledge_for_agents.index [--corpus original] [--variant m0] [--model MODEL]
        [--batch-size 32] [--rebuild]

One collection per corpus, text variant and model. Before calling the API, vectors
are copied from any other collection of the same model that stores the identical
text, so corpora that differ only in metadata cost no embedding calls.

Resumable: documents already in the collection are skipped. BM25 is built in
memory at search time (5k docs index in a few seconds), so only dense
embeddings are persisted.
"""

from __future__ import annotations

import argparse
import re

import chromadb
from tqdm import tqdm

from organizing_knowledge_for_agents.corpus import Doc, chroma_metadata, load_corpus, render_text
from organizing_knowledge_for_agents.embeddings import Embedder
from organizing_knowledge_for_agents.settings import CHROMA_DIR, CORPORA, EMBED_MODEL

COPY_BATCH = 500


def _safe(name: str) -> str:
    # Chroma names allow only [a-zA-Z0-9._-]; some model or deployment IDs contain '/' or spaces.
    return re.sub(r"[^a-zA-Z0-9._-]", "-", name)


def collection_name(corpus: str, variant: str, model: str) -> str:
    return _safe(f"{corpus}-{variant}-{model}")


def get_collection(corpus: str, variant: str, model: str, create: bool = False) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = collection_name(corpus, variant, model)
    if create:
        return client.get_or_create_collection(
            name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
        )
    return client.get_collection(name, embedding_function=None)


def copy_embeddings(collection: chromadb.Collection, docs: list[Doc], variant: str, model: str) -> list[Doc]:
    """Copy vectors for identical text from other collections of `model`; return the docs still to embed."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    todo = {doc.doc_id: doc for doc in docs}
    for source in client.list_collections():
        if not todo or source.name == collection.name or not source.name.endswith(f"-{_safe(model)}"):
            continue
        source = client.get_collection(source.name, embedding_function=None)
        ids = list(todo)
        copied = 0
        for start in range(0, len(ids), COPY_BATCH):
            got = source.get(ids=ids[start : start + COPY_BATCH], include=["embeddings", "documents"])
            same = [
                (doc_id, vector, text)
                for doc_id, vector, text in zip(got["ids"], got["embeddings"], got["documents"])
                if text == render_text(todo[doc_id], variant)
            ]
            if not same:
                continue
            collection.add(
                ids=[doc_id for doc_id, _, _ in same],
                embeddings=[vector for _, vector, _ in same],
                documents=[text for _, _, text in same],
                metadatas=[chroma_metadata(todo[doc_id]) for doc_id, _, _ in same],
            )
            for doc_id, _, _ in same:
                del todo[doc_id]
            copied += len(same)
        if copied:
            print(f"Copied {copied} vectors from {source.name}")
    return list(todo.values())


def build_index(corpus_name: str, variant: str, model: str, batch_size: int, rebuild: bool = False) -> None:
    if rebuild:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        name = collection_name(corpus_name, variant, model)
        if name in [c.name for c in client.list_collections()]:
            client.delete_collection(name)

    collection = get_collection(corpus_name, variant, model, create=True)
    corpus = load_corpus(corpus_name)
    existing = set(collection.get(include=[])["ids"]) if collection.count() else set()
    todo = [doc for doc in corpus.values() if doc.doc_id not in existing]
    print(f"{corpus_name}: {len(corpus)} docs, {len(existing)} already indexed")
    todo = copy_embeddings(collection, todo, variant, model)
    print(f"{len(todo)} to embed with {model}")

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
    parser.add_argument("--corpus", default="original", choices=CORPORA, help="Corpus version under data/")
    parser.add_argument("--variant", default="m0", help="Document rendering to index (corpus.render_text)")
    parser.add_argument("--model", default=EMBED_MODEL, help="Embedding model (on Azure: the deployment name)")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Docs per request (OpenAI caps a request at 300k tokens in total)"
    )
    parser.add_argument("--rebuild", action="store_true", help="Drop and re-create the collection")
    args = parser.parse_args()
    build_index(args.corpus, args.variant, args.model, args.batch_size, args.rebuild)


if __name__ == "__main__":
    main()
