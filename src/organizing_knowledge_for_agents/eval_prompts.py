"""Judge prompts, copied verbatim from EnterpriseRAG-Bench (src/prompts/answer_evaluation.py, commit d36685e).

Copyright (c) 2026 DanswerAI, Inc. MIT License, see licenses/EnterpriseRAG-Bench-LICENSE.
Kept identical so scores stay comparable with the bench's own evaluator.
"""

ANSWER_CITATION_STRIPPING_PROMPT = """
Strip all citations and reference sections from the provided string without modifying any other content. Citations of all forms such as [number], [number][number], [number](link), etc. should all be removed. \
Anything that is clearly a citation, regardless of format should be removed. Also remove any reference footers or intermediate sections that are clearly references. \
Output the stripped string with no additional text or formatting changes.

String to strip:
```
{answer_string}
```

CRITICAL: Output only the stripped string and nothing else. Only citations should be removed, everything else should be left unchanged.
""".strip()


ANSWER_WHOLISTIC_EVALUATION_PROMPT = """
You are a wholistic and detail-oriented answer evaluator. Given a query, a gold answer, and a candidate answer, evaluate if the candidate answer aligned with the gold answer.
Use the following metrics for evaluating the answer:
- The candidate answer must provide loosely the same information as the gold answer. The core aspects directly asked by the query must be addressed in the candidate answer and they must not conflict with the gold answer.
- If there are any specific quantities mentioned in both answers, they must match.
- The candidate answer is not required to contain all of the same details as the gold answer.
- The candidate answer must address the key parts of the query, if it is missing anything critical to the question, it is misaligned.
- The candidate answer may contain more details, richer information, or other helpful relevant information than the gold answer, this is ok.
- The candidate answer may offer up additional loosely related information that adds to the context of the answer, this is ok as long as it does not lead the user to an incorrect conclusion (compared to the gold answer).
- Do not penalize the candidate answer for stylistic differences. If the candidate answer offers follow up questions, asks additional clarifications to the user, or offers additional context, \
this is ok as long as it contains the necessary information to answer the question.

There is a separate check for answer completeness, this is not in scope for this evaluation. However, if there are core parts of the question being left out, this is misaligned.

## Query
```
{query}
```

## Gold Answer
```
{gold_answer}
```

## Candidate Answer
```
{candidate_answer}
```

## Output Format
Output a JSON with "reason" and "aligned" fields. The "reason" field should be a as concise as possible (max 1 sentence) explanation of why the candidate answer is aligned or misaligned with the gold answer. \
The "aligned" field should be a simple "yes" or "no", use only those two strings literally and nothing else.

CRITICAL: Output only a JSON object with the following fields in the order shown below (with no additional text or formatting):
{{
  "reason": "reason for the classification",
  "aligned": "yes or no"
}}
""".strip()


INDIVIDUAL_FACT_VALIDATOR_PROMPT = """
You are an answer validator. Given an answer and a statement, determine if the answer is consistent with and contains the information in the statement. \
The answer may contain more details or richer information than the statement but as long as it does not contradict the statement, this is valid. \
If there are negative statements such as "The answer must not say...", it is valid if the answer mentions the statement with caveats or qualifications. \
It is valid if additional context is shared for completeness however hallucinations are not allowed. \
Output a simple yes or no for if the answer is consistent with and contains the information in the statement.

## Answer
```
{answer}
```

## Statement
```
{statement}
```

CRITICAL: output only a simple yes if the answer is consistent with the statement or a no if the answer does not contain the information in the statement or contradicts the statement.
""".strip()
