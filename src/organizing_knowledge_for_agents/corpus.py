"""Load and clean the Confluence slice of EnterpriseRAG-Bench.

The corpus comes in three versions (settings.CORPORA) with the same pages, titles
and bodies: `original` has the native metadata, `generated-metadata` adds
LLM-generated tags, and `removed-metadata` has none.

Only whitelisted fields survive loading. Generator artifacts such as
`original_location` (the pre-noise folder) and `dataset_noise_document` are
dropped here so no downstream component can leak them to the agent.
"""

from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

from organizing_knowledge_for_agents.settings import CORPORA, DATA_DIR

METADATA_FIELDS = (
    # native Confluence metadata
    "space",
    "author",
    "owner_team",
    "status",
    "created_at",
    "last_updated",
    "reviewers",
    "labels",
    "related_pages",
    "confidentiality",
    # generated tags (generated-metadata only)
    "document_type",
    "doc_classification",
    "lifecycle_status",
    "volatility",
    "data_classification",
    "audience_groups",
    "product_area",
)
LIST_FIELDS = {"reviewers", "labels", "related_pages", "audience_groups", "product_area"}
DATE_FIELDS = {"created_at", "last_updated"}


@dataclass
class Doc:
    doc_id: str  # dataset_doc_uuid (dsid_...), the ID the eval scores against
    title: str
    content: str
    path: str  # relative to the confluence root, e.g. eng-sre/runbooks/foo.json
    metadata: dict = field(default_factory=dict)

    @property
    def folder(self) -> str:
        return str(Path(self.path).parent)


def _render_content(raw: dict) -> str:
    # Mirrors EnterpriseRAG-Bench's extract_document_content so text matches the official export.
    names = raw["content_field_names"]
    if len(names) == 1:
        return str(raw[names[0]])
    parts = []
    for name in names:
        value = raw.get(name, "")
        if isinstance(value, list):
            value = "\n".join(str(v) for v in value)
        parts.append(f"{name}:\n{value}")
    return "\n\n".join(parts)


def _clean_metadata(raw: dict) -> dict:
    meta = {}
    for key in METADATA_FIELDS:
        if key in raw["content_field_names"]:
            continue  # part of the body on a few pages (related_pages on 3), already in the content
        value = raw.get(key)
        if value is None or value == "" or value == []:
            continue
        if key in LIST_FIELDS:
            value = [str(v) for v in value] if isinstance(value, list) else [str(value)]
        else:
            value = str(value)
        meta[key] = value
    return meta


def load_docs(root: Path) -> list[Doc]:
    docs = []
    for path in sorted(root.rglob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        docs.append(
            Doc(
                doc_id=raw["dataset_doc_uuid"],
                title=str(raw.get(raw["title_field_name"], "")),
                content=_render_content(raw),
                path=str(path.relative_to(root)),
                metadata=_clean_metadata(raw),
            )
        )
    return docs


@lru_cache(maxsize=None)
def load_corpus(corpus: str = "original") -> dict[str, Doc]:
    if corpus not in CORPORA:
        raise ValueError(f"Unknown corpus {corpus!r}, expected one of {CORPORA}")
    docs = {doc.doc_id: doc for doc in load_docs(DATA_DIR / corpus)}
    if not docs:
        raise RuntimeError(f"No pages found in {DATA_DIR / corpus}")
    return docs


def render_text(doc: Doc, variant: str = "m0") -> str:
    """Text that gets embedded and BM25-indexed.

    m0: title + content only (no metadata). Add metadata/enrichment variants here.
    """
    if variant == "m0":
        return f"{doc.title}\n\n{doc.content}"
    raise ValueError(f"Unknown text variant: {variant}")


def metadata_text(doc: Doc) -> str:
    """A page's metadata as words for BM25, with dates also as month and year ("March 2026").

    related_pages is left out: its paths would make a page match its neighbours' topics.
    Underscores become spaces so tags like `sre_oncall` match the words in a query.
    """
    lines = []
    for key, value in doc.metadata.items():
        if key == "related_pages":
            continue
        if key in DATE_FIELDS:
            as_int = date_to_int(value)
            if as_int is not None:
                value = f"{value} {calendar.month_name[as_int // 100 % 100]} {as_int // 10000}"
        elif isinstance(value, list):
            value = ", ".join(value)
        lines.append(f"{key}: {value}".replace("_", " "))
    return "\n".join(lines)


def date_to_int(value: str) -> int | None:
    """'2026-03-08' -> 20260308, for numeric range filters. None if unparseable."""
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", value)
    if not match:
        return None
    try:
        parsed = date(*(int(g) for g in match.groups()))
    except ValueError:
        return None
    return parsed.year * 10000 + parsed.month * 100 + parsed.day


def chroma_metadata(doc: Doc) -> dict:
    """Metadata in the shape Chroma accepts: scalars or non-empty lists, dates as ints."""
    meta: dict = {"folder": doc.folder, "path": doc.path}
    for key, value in doc.metadata.items():
        if key in DATE_FIELDS:
            as_int = date_to_int(value)
            if as_int is not None:
                meta[key] = as_int
        else:
            meta[key] = value
    return meta
