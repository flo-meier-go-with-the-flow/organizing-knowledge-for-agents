"""Question sets with gold answers, committed under data/questions/.

The `metadata` set's IDs overlap with the other sets' IDs, so each set is
always run and evaluated on its own.
"""

from __future__ import annotations

import json

from organizing_knowledge_for_agents.settings import QUESTIONS_DIR

QUESTION_SETS = {
    "confluence": "Questions whose gold docs are all Confluence pages (64)",
    "not_found": "Unanswerable questions, used as an abstention control (20)",
    "metadata": "Metadata-dependent Confluence questions (12)",
}


def load_question_set(name: str) -> list[dict]:
    if name not in QUESTION_SETS:
        raise ValueError(f"Unknown question set {name!r}; choose from {sorted(QUESTION_SETS)}")
    with open(QUESTIONS_DIR / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
