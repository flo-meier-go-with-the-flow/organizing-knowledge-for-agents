"""Question sets drawn from EnterpriseRAG-Bench.

Each set records which questions file the bench's evaluator must be pointed at.
`metadata` comes from extra_questions.jsonl, whose IDs overlap with
questions.jsonl, so it is always run and evaluated separately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from organizing_knowledge_for_agents.settings import ERB_ROOT


@dataclass(frozen=True)
class QuestionSet:
    name: str
    questions_file: Path
    predicate: Callable[[dict], bool]
    description: str


QUESTION_SETS = {
    qs.name: qs
    for qs in (
        QuestionSet(
            "confluence",
            ERB_ROOT / "questions.jsonl",
            lambda q: q.get("source_types") == ["confluence"],
            "Questions whose gold docs are all Confluence pages (64)",
        ),
        QuestionSet(
            "not_found",
            ERB_ROOT / "questions.jsonl",
            lambda q: q["question_type"] == "info_not_found",
            "Unanswerable questions, used as an abstention control (20)",
        ),
        QuestionSet(
            "metadata",
            ERB_ROOT / "extra_questions.jsonl",
            lambda q: q.get("source_types") == ["confluence"],
            "Metadata-dependent Confluence questions from extra_questions.jsonl (12)",
        ),
    )
}


def load_question_set(name: str) -> tuple[QuestionSet, list[dict]]:
    qs = QUESTION_SETS[name]
    with open(qs.questions_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return qs, [q for q in rows if qs.predicate(q)]
