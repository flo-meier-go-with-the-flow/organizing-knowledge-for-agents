"""Run the agent over a question set and write files the bench's evaluator can score.

Usage:
    uv run python -m organizing_knowledge_for_agents.run --config configs/baseline.toml --questions confluence
        [--repeat 0] [--parallelism 4] [--limit N] [--question-id qst_XXXX] [--model NAME]

Writes to runs/<config name>/<model>/<question set>/r<repeat>/:
    answers.jsonl   {question_id, answer, document_ids}, the evaluator's input format
    metrics.jsonl   per-question efficiency metrics (tokens, tool calls, latency)
    errors.jsonl    questions that raised, and are retried on the next invocation
    traces/<qid>.json
    config.json, summary.json

Re-running the same command resumes: questions already in answers.jsonl are skipped.
"""

from __future__ import annotations

import argparse
import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from organizing_knowledge_for_agents.agent import run_agent
from organizing_knowledge_for_agents.jsonl import read_jsonl
from organizing_knowledge_for_agents.questions import QUESTION_SETS, load_question_set
from organizing_knowledge_for_agents.search import Searcher
from organizing_knowledge_for_agents.settings import RUNS_DIR, ExperimentConfig, load_config
from organizing_knowledge_for_agents.tools import ToolBox


def summarize(run_dir: Path, n_questions: int) -> dict:
    # Keep the last row per question: a crash between writes can leave a duplicate.
    metrics = list({m["question_id"]: m for m in read_jsonl(run_dir / "metrics.jsonl")}.values())
    errors = read_jsonl(run_dir / "errors.jsonl")
    done = {m["question_id"] for m in metrics}
    summary: dict = {
        "questions": n_questions,
        "completed": len(done),
        "open_errors": len({e["question_id"] for e in errors} - done),
    }
    if metrics:
        for key in ("n_tool_calls", "n_docs_seen", "n_doc_ids", "seconds", "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"):
            summary[f"mean_{key}"] = round(sum(m[key] for m in metrics) / len(metrics), 2)
        summary["total_input_tokens"] = sum(m["input_tokens"] for m in metrics)
        summary["total_output_tokens"] = sum(m["output_tokens"] for m in metrics)
        finished: dict[str, int] = {}
        for m in metrics:
            finished[m["finished_via"]] = finished.get(m["finished_via"], 0) + 1
        summary["finished_via"] = finished
    return summary


def run(cfg: ExperimentConfig, question_set: str, repeat: int, parallelism: int, limit: int | None, question_id: str | None) -> Path:
    questions = load_question_set(question_set)
    if question_id:
        questions = [q for q in questions if q["question_id"] == question_id]
    if limit:
        questions = questions[:limit]

    run_dir = RUNS_DIR / cfg.name / cfg.agent.model / question_set / f"r{repeat}"
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({**cfg.to_dict(), "question_set": question_set}, indent=2)
    )

    done = {row["question_id"] for row in read_jsonl(run_dir / "answers.jsonl")}
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"{cfg.name} / {cfg.agent.model} / {question_set} / r{repeat}: {len(questions)} questions, {len(done)} done, {len(todo)} to run")

    toolbox = ToolBox(Searcher(cfg.retrieval), cfg.tools)
    client = OpenAI(max_retries=5)  # reads OPENAI_API_KEY and OPENAI_BASE_URL
    lock = threading.Lock()

    def append(name: str, row: dict) -> None:
        with lock, open(run_dir / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def work(q: dict) -> None:
        qid = q["question_id"]
        try:
            res = run_agent(client, cfg.agent, toolbox, q["question"])
        except Exception as e:
            append("errors.jsonl", {"question_id": qid, "error": repr(e), "traceback": traceback.format_exc()})
            return
        (run_dir / "traces" / f"{qid}.json").write_text(
            json.dumps({"question_id": qid, "question": q["question"], "answer": res.answer, "doc_ids": res.doc_ids, "trace": res.trace}, indent=2)
        )
        append("metrics.jsonl", {"question_id": qid, "question_type": q["question_type"], **res.metrics()})
        # Written last: a row here marks the question as done for resume.
        append("answers.jsonl", {"question_id": qid, "answer": res.answer, "document_ids": res.doc_ids})

    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = [pool.submit(work, q) for q in todo]
        for _ in tqdm(as_completed(futures), total=len(futures), unit="q"):
            pass

    summary = summarize(run_dir, len(questions))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Experiment config (TOML)")
    parser.add_argument("--questions", default="confluence", choices=sorted(QUESTION_SETS))
    parser.add_argument("--repeat", type=int, default=0, help="Repeat index; runs are nondeterministic, so repeat each cell")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Only run the first N questions (pilots)")
    parser.add_argument("--question-id", help="Run a single question")
    parser.add_argument("--model", help="Override agent.model from the config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.agent.model = args.model
    run(cfg, args.questions, args.repeat, args.parallelism, args.limit, args.question_id)


if __name__ == "__main__":
    main()
