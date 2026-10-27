"""Evaluator tests with a scripted judge (no network)."""

from __future__ import annotations

import json

import pytest

from organizing_knowledge_for_agents.evaluate import compute_stats, evaluate_run, score_answer
from organizing_knowledge_for_agents.questions import load_question_set

QUESTION = {
    "question_id": "qst_x",
    "question_type": "basic",
    "question": "How many PTO days do employees get?",
    "expected_doc_ids": ["dsid_a", "dsid_b"],
    "gold_answer": "Employees get 20 PTO days, approved by their manager.",
    "answer_facts": ["Employees get 20 PTO days.", "PTO must be approved by the employee's manager."],
}


class FakeJudge:
    model = "fake-judge"
    reasoning_effort = "medium"

    def __init__(self, fail_facts: bool = False):
        self.fail_facts = fail_facts
        self.prompts: list[str] = []

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if prompt.startswith("Strip all citations"):
            answer = prompt.split("```")[1].strip()
            return answer.replace(" [1]", "")
        if prompt.startswith("You are a wholistic"):
            return 'Here you go: {"reason": "same count", "aligned": "yes"}'
        if prompt.startswith("You are an answer validator"):
            if self.fail_facts:
                raise RuntimeError("judge unavailable")
            statement = prompt.split("## Statement")[1]
            return "yes" if "20 PTO days" in statement else "no\nthe answer never mentions approval"
        raise AssertionError(f"unexpected prompt: {prompt[:40]}")


def test_scores_follow_bench_rules():
    judge = FakeJudge()
    answer = {"answer": "Employees get 20 PTO days [1] (dsid_" + "a" * 32 + ").", "document_ids": ["dsid_a", "dsid_c", "dsid_c"]}

    row = score_answer(judge, QUESTION, answer)

    assert row["answer_correct"] is True
    assert row["completeness_pct"] == 50.0
    assert row["document_recall_pct"] == 50.0
    assert row["invalid_extra_docs"] == 1
    fact_prompts = [p for p in judge.prompts if p.startswith("You are an answer validator")]
    assert all("[1]" not in p and "dsid_aaaa" not in p for p in fact_prompts)


def test_question_without_gold_docs_has_no_recall():
    row = score_answer(FakeJudge(), {**QUESTION, "expected_doc_ids": []}, {"answer": "20 PTO days", "document_ids": ["dsid_a"]})
    assert row["document_recall_pct"] is None and row["invalid_extra_docs"] is None


def test_empty_answer_scores_zero_without_judge_calls():
    judge = FakeJudge()
    row = score_answer(judge, QUESTION, {"answer": "", "document_ids": []})
    assert (row["answer_correct"], row["completeness_pct"], row["document_recall_pct"]) == (False, 0.0, 0.0)
    assert judge.prompts == []


def test_failed_fact_check_zeroes_completeness():
    row = score_answer(FakeJudge(fail_facts=True), QUESTION, {"answer": "20 PTO days", "document_ids": ["dsid_a"]})
    assert row["completeness_pct"] == 0.0 and row["answer_correct"] is True


def test_compute_stats():
    rows = [
        {"answer_correct": True, "completeness_pct": 100.0, "document_recall_pct": 100.0, "invalid_extra_docs": 0},
        {"answer_correct": False, "completeness_pct": 50.0, "document_recall_pct": None, "invalid_extra_docs": None},
    ]
    assert compute_stats(rows) == {
        "count": 2,
        "average_correctness_pct": 50.0,
        "average_completeness_pct": 75.0,
        "combined_correctness_completeness_score": 50.0,
        "average_recall_pct": 100.0,
        "average_invalid_extra_docs": 0.0,
    }


def test_evaluate_run_resumes_and_pins_the_judge(tmp_path):
    question = load_question_set("confluence")[0]
    (tmp_path / "config.json").write_text(json.dumps({"question_set": "confluence"}))
    (tmp_path / "answers.jsonl").write_text(
        json.dumps({"question_id": question["question_id"], "answer": "an answer", "document_ids": question["expected_doc_ids"]}) + "\n"
    )

    summary = evaluate_run(tmp_path, FakeJudge(), parallelism=1)
    assert summary["scored"] == 1 and summary["overall"]["average_recall_pct"] == 100.0

    resumed = FakeJudge()
    evaluate_run(tmp_path, resumed, parallelism=1)
    assert resumed.prompts == []  # nothing left to score

    other = FakeJudge()
    other.model = "another-judge"
    with pytest.raises(SystemExit):
        evaluate_run(tmp_path, other, parallelism=1)
