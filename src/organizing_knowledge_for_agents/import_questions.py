"""Regenerate data/questions/ from an EnterpriseRAG-Bench checkout.

Usage:
    uv run python -m organizing_knowledge_for_agents.import_questions --erb-root ../EnterpriseRAG-Bench

Rows are copied verbatim. Only needed when refreshing the questions from upstream;
everything else in this repo reads the committed files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from organizing_knowledge_for_agents.questions import QUESTION_SETS
from organizing_knowledge_for_agents.settings import QUESTIONS_DIR

# question set -> (upstream file, selection)
SOURCES = {
    "confluence": ("questions.jsonl", lambda q: q.get("source_types") == ["confluence"]),
    "not_found": ("questions.jsonl", lambda q: q["question_type"] == "info_not_found"),
    "metadata": ("extra_questions.jsonl", lambda q: q.get("source_types") == ["confluence"]),
}


def import_questions(erb_root: Path) -> None:
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for name in QUESTION_SETS:
        source, keep = SOURCES[name]
        with open(erb_root / source, encoding="utf-8") as f:
            lines = [line.rstrip("\n") + "\n" for line in f if line.strip()]
        kept = [line for line in lines if keep(json.loads(line))]
        (QUESTIONS_DIR / f"{name}.jsonl").write_text("".join(kept), encoding="utf-8")
        print(f"{name}: {len(kept)} questions from {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--erb-root", type=Path, required=True, help="Path to an EnterpriseRAG-Bench checkout")
    args = parser.parse_args()
    import_questions(args.erb_root)


if __name__ == "__main__":
    main()
