"""Paths, environment, and experiment config loading."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/<package>/settings.py -> repo root
load_dotenv(PROJECT_ROOT / ".env")

# Corpus and question sets (with gold answers) are committed to this repo. The corpus comes in
# three versions under DATA_DIR with identical titles and bodies; they differ only in metadata.
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data")).resolve()
CORPORA = ("original", "generated-metadata", "removed-metadata")
QUESTIONS_DIR = Path(os.environ.get("QUESTIONS_DIR", PROJECT_ROOT / "data" / "questions")).resolve()

CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", PROJECT_ROOT / "data" / "chroma")).resolve()
RUNS_DIR = Path(os.environ.get("RUNS_DIR", PROJECT_ROOT / "runs")).resolve()

# Embeddings: OpenAI text-embedding-3-large. Set EMBED_BASE_URL / EMBED_API_KEY only if
# embeddings live on a different endpoint than the agent model.
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")


@dataclass
class AgentConfig:
    model: str = "gpt-5-mini"
    # Empty string disables the `reasoning` parameter (for non-reasoning models).
    reasoning_effort: str = "medium"
    max_tool_calls: int = 15
    max_output_tokens: int = 16000


@dataclass
class RetrievalConfig:
    corpus: str = "original"  # one of CORPORA, loaded from DATA_DIR/<corpus>
    mode: str = "hybrid"  # hybrid | dense | bm25
    k: int = 5
    candidates: int = 50  # per retriever, before fusion
    rrf_k: int = 60
    text_variant: str = "m0"  # which document rendering was indexed (see corpus.render_text)
    embedding_model: str = EMBED_MODEL
    metadata_in_bm25: bool = False  # BM25 also indexes each page's metadata; embeddings are unchanged
    # Recency pushdown: a newest-first ranking of the top fused candidates, fused in with this
    # weight. Pages without a last_updated date get no boost; 0 turns it off.
    recency_weight: float = 0.0
    recency_depth: int = 20


@dataclass
class ToolsConfig:
    result_view: str = "full"  # full | preview
    preview_chars: int = 400
    show_metadata: bool = False
    metadata_filters: bool = False  # search takes optional filters on the corpus's metadata fields


@dataclass
class ExperimentConfig:
    name: str
    agent: AgentConfig = field(default_factory=AgentConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return ExperimentConfig(
        name=raw["name"],
        agent=AgentConfig(**raw.get("agent", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        tools=ToolsConfig(**raw.get("tools", {})),
    )
