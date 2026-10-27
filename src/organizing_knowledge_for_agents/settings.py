"""Paths, environment, and experiment config loading."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/<package>/settings.py -> repo root
load_dotenv(PROJECT_ROOT / ".env")

# The Confluence corpus is committed to this repo. Questions and the evaluator
# still come from an EnterpriseRAG-Bench checkout.
CONFLUENCE_DIR = Path(os.environ.get("CONFLUENCE_DIR", PROJECT_ROOT / "data" / "confluence")).resolve()
ERB_ROOT = Path(os.environ.get("ERB_ROOT", PROJECT_ROOT.parent / "EnterpriseRAG-Bench")).resolve()

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
    mode: str = "hybrid"  # hybrid | dense | bm25
    k: int = 5
    candidates: int = 50  # per retriever, before fusion
    rrf_k: int = 60
    text_variant: str = "m0"  # which document rendering was indexed (see corpus.render_text)
    embedding_model: str = EMBED_MODEL


@dataclass
class ToolsConfig:
    result_view: str = "full"  # full | preview
    preview_chars: int = 400
    show_metadata: bool = False


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
