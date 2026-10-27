"""Score a run's answers the way EnterpriseRAG-Bench's metrics_based_eval does with --no-correction.

Usage:
    uv run python -m organizing_knowledge_for_agents.evaluate --run-dir runs/<config>/<model>/<set>/r0 --judge-model MODEL
        [--judge-reasoning-effort medium] [--parallelism 8] [--skip-citation-stripping] [--question-id ID]

Per question, with the bench's prompts (eval_prompts.py) and scoring rules:
    answer_correct      LLM judge: is the answer aligned with the gold answer?
    completeness_pct    share of the gold answer_facts an LLM judge finds in the answer
    document_recall_pct share of gold docs among the returned doc IDs (questions with gold docs only)
    invalid_extra_docs  returned docs that aren't gold (without the bench's correction step,
                        every non-gold doc counts)

Writes eval.jsonl (per question; re-running resumes) and eval_summary.json (overall and per
question type) into the run dir. One judge per run dir: resuming with a different judge fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from organizing_knowledge_for_agents.eval_prompts import (
    ANSWER_CITATION_STRIPPING_PROMPT,
    ANSWER_WHOLISTIC_EVALUATION_PROMPT,
    INDIVIDUAL_FACT_VALIDATOR_PROMPT,
)
from organizing_knowledge_for_agents.jsonl import read_jsonl
from organizing_knowledge_for_agents.questions import load_question_set

MAX_ATTEMPTS = 3  # the bench retries each judge call up to 3 times
DSID = re.compile(r"dsid_[a-f0-9]{20,32}")
YES = re.compile(r"\byes\b", re.IGNORECASE)


class Judge:
    def __init__(self, model: str, reasoning_effort: str = "medium", client: OpenAI | None = None):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = client or OpenAI(max_retries=5)  # reads OPENAI_API_KEY and OPENAI_BASE_URL

    def ask(self, prompt: str) -> str:
        kwargs: dict = {"model": self.model, "input": prompt, "store": False}
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        return self.client.responses.create(**kwargs).output_text.strip()


def strip_citations(judge: Judge, answer: str) -> str:
    """LLM pass that removes citations, then drop any remaining dsid_ references (as the bench does)."""
    try:
        stripped = judge.ask(ANSWER_CITATION_STRIPPING_PROMPT.format(answer_string=answer)) or answer
    except Exception:
        stripped = answer  # the bench keeps the original answer if stripping fails
    return re.sub(r" {2,}", " ", DSID.sub("", stripped)).strip()


def _json_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


def judge_correctness(judge: Judge, question: str, gold_answer: str, answer: str) -> tuple[bool, str]:
    prompt = ANSWER_WHOLISTIC_EVALUATION_PROMPT.format(query=question, gold_answer=gold_answer, candidate_answer=answer)
    for _ in range(MAX_ATTEMPTS):
        try:
            parsed = _json_object(judge.ask(prompt))
        except Exception:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("aligned"), str):
            return YES.search(parsed["aligned"]) is not None, str(parsed.get("reason", ""))
    return False, "judge output unusable"


def judge_fact(judge: Judge, answer: str, fact: str) -> bool:
    prompt = INDIVIDUAL_FACT_VALIDATOR_PROMPT.format(answer=answer, statement=fact)
    for attempt in range(MAX_ATTEMPTS):
        try:
            lines = judge.ask(prompt).splitlines()
            return bool(lines) and YES.search(lines[0]) is not None
        except Exception:
            if attempt == MAX_ATTEMPTS - 1:
                raise
    return False


def score_answer(judge: Judge, question: dict, answer_row: dict, strip: bool = True) -> dict:
    answer = answer_row.get("answer") or ""
    if answer and strip:
        answer = strip_citations(judge, answer)

    returned = set(answer_row.get("document_ids") or [])
    gold = set(question.get("expected_doc_ids") or [])
    recall = round(100 * len(returned & gold) / len(gold), 2) if gold else None
    invalid_extra = len(returned - gold) if gold else None

    correct, reason, completeness = False, "", 0.0
    if answer:
        facts = question.get("answer_facts") or []
        gold_answer = question.get("gold_answer") or ""
        with ThreadPoolExecutor(max_workers=len(facts) + 1) as pool:
            correctness = (
                pool.submit(judge_correctness, judge, question["question"], gold_answer, answer) if gold_answer else None
            )
            fact_checks = [pool.submit(judge_fact, judge, answer, fact) for fact in facts]
            try:
                completeness = 100 * sum(f.result() for f in fact_checks) / len(facts) if facts else 100.0
            except Exception:
                completeness = 0.0  # the bench scores completeness 0 if any fact check fails
            correct, reason = correctness.result() if correctness else (True, "")

    return {
        "question_id": question["question_id"],
        "question_type": question.get("question_type"),
        "answer_correct": correct,
        "correctness_reasoning": reason,
        "completeness_pct": round(completeness, 2),
        "document_recall_pct": recall,
        "invalid_extra_docs": invalid_extra,
    }


def compute_stats(rows: list[dict]) -> dict:
    """Same aggregates as the bench; recall and extra docs only over questions with gold docs."""
    n = len(rows)
    if n == 0:
        return {"count": 0}
    recalls = [r["document_recall_pct"] for r in rows if r["document_recall_pct"] is not None]
    extras = [r["invalid_extra_docs"] for r in rows if r["invalid_extra_docs"] is not None]
    return {
        "count": n,
        "average_correctness_pct": round(100 * sum(r["answer_correct"] for r in rows) / n, 2),
        "average_completeness_pct": round(sum(r["completeness_pct"] for r in rows) / n, 2),
        "combined_correctness_completeness_score": round(
            sum(r["completeness_pct"] for r in rows if r["answer_correct"]) / n, 2
        ),
        "average_recall_pct": round(sum(recalls) / len(recalls), 2) if recalls else None,
        "average_invalid_extra_docs": round(sum(extras) / len(extras), 2) if extras else None,
    }


def evaluate_run(run_dir: Path, judge: Judge, parallelism: int = 8, strip: bool = True, question_id: str | None = None) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    question_set = config["question_set"]
    questions = {q["question_id"]: q for q in load_question_set(question_set)}
    answers = read_jsonl(run_dir / "answers.jsonl")
    if question_id:
        answers = [a for a in answers if a["question_id"] == question_id]

    eval_path = run_dir / "eval.jsonl"
    previous = read_jsonl(eval_path)
    judges = {(r["judge_model"], r["judge_reasoning_effort"]) for r in previous}
    if judges - {(judge.model, judge.reasoning_effort)}:
        raise SystemExit(f"{eval_path} was scored with {sorted(judges)}; use the same judge or delete the file.")
    done = {r["question_id"] for r in previous}
    todo = [a for a in answers if a["question_id"] not in done]
    print(f"{run_dir}: {len(answers)} answers, {len(done)} already scored, {len(todo)} to score with {judge.model}")

    lock = threading.Lock()

    def work(answer_row: dict) -> None:
        row = score_answer(judge, questions[answer_row["question_id"]], answer_row, strip)
        row.update(judge_model=judge.model, judge_reasoning_effort=judge.reasoning_effort)
        with lock, open(eval_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = {pool.submit(work, a): a["question_id"] for a in todo}
        for future in tqdm(as_completed(futures), total=len(futures), unit="q"):
            if future.exception():
                print(f"  {futures[future]}: {future.exception()!r} (retried on the next run)")

    rows = list({r["question_id"]: r for r in read_jsonl(eval_path)}.values())
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["question_type"], []).append(r)
    summary = {
        "question_set": question_set,
        "judge_model": judge.model,
        "judge_reasoning_effort": judge.reasoning_effort,
        "answers": len(read_jsonl(run_dir / "answers.jsonl")),
        "scored": len(rows),
        "overall": compute_stats(rows),
        "by_question_type": {t: compute_stats(by_type[t]) for t in sorted(by_type)},
    }
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["overall"], indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="A run directory containing answers.jsonl")
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL"), help="Judge model (default: $JUDGE_MODEL)")
    parser.add_argument("--judge-reasoning-effort", default="medium", help='Reasoning effort; "" for non-reasoning models')
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--skip-citation-stripping", action="store_true")
    parser.add_argument("--question-id", help="Score a single question")
    args = parser.parse_args()
    if not args.judge_model:
        parser.error("set --judge-model or JUDGE_MODEL (pin one judge for the whole study)")

    judge = Judge(args.judge_model, args.judge_reasoning_effort)
    evaluate_run(args.run_dir, judge, args.parallelism, not args.skip_citation_stripping, args.question_id)


if __name__ == "__main__":
    main()
