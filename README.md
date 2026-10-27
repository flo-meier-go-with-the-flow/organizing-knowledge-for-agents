# organizing-knowledge-for-agents

Experiments on how data structure (metadata, progressive disclosure, folder layout) affects agentic retrieval, using the Confluence slice of [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench).

The baseline is an agent built on the OpenAI Responses API. Its search tool does hybrid retrieval: OpenAI `text-embedding-3-large` vectors in Chroma plus in-memory BM25, fused with reciprocal rank fusion. Every grid cell runs the same agent loop; conditions differ only in config toggles for the tools and the indexed text.

Goals, experiment grid, decision log and current status: [docs/embed-and-basic-agent-loop.md](docs/embed-and-basic-agent-loop.md).

## Baseline results

First baseline, 27 October 2026: `configs/baseline.toml` with `GPT-6-luna` (medium reasoning) on the `confluence` question set, one run (`r0`).

| Question type | Questions | Correct | Completeness | Combined | Doc recall | Non-gold docs cited | Tool calls | Input tokens |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **All** | **64** | **71.9%** | **79.6%** | **65.6** | **82.8%** | **0.58** | **3.5** | **99k** |
| Basic | 19 | 89.5% | 89.6% | 84.3 | 94.7% | 0.26 | 2.1 | 36k |
| Constrained | 11 | 90.9% | 92.8% | 87.7 | 95.5% | 0.45 | 2.8 | 43k |
| Semantic | 15 | 60.0% | 64.6% | 51.3 | 66.7% | 0.67 | 4.6 | 203k |
| Completeness | 11 | 36.4% | 74.6% | 30.9 | 75.2% | 1.18 | 6.5 | 173k |
| Other | 8 | 75.0% | 72.4% | 65.2 | 78.1% | 0.50 | 1.8 | 25k |

- Values are means per question. The metrics are defined in [3. Evaluate](#3-evaluate). Combined is the bench's headline score: completeness, counted only for answers judged correct.
- Other groups the small types: miscellaneous (3), intra-document reasoning (2), project-related (2) and conflicting info (1).
- Single-page lookups (basic, constrained) are close to solved. Completeness questions need 6.2 gold pages on average, against 2.1 overall, and only 36% of them are answered correctly.
- Of the 18 wrong answers, 6 never saw a gold page in any tool result, and 9 saw every gold page but were still judged wrong.

These numbers are provisional:
- **The judge was `GPT-6-luna`, the agent model itself**, which likely flatters the scores. Rescore with a stronger judge once one is deployed.
- **One run only.** Agent runs are nondeterministic, so differences of a few points between conditions need repeats.
- **The `not_found` and `metadata` sets haven't been run yet.**

### Findings

**Question types.** EnterpriseRAG-Bench generates each type to test a different retrieval skill (definitions in its `methodology.md`):
- **Basic:** one page, phrased without long exact matches.
- **Semantic:** loose phrasing with few keywords or qualifiers, worded differently from the page that answers it.
- **Constrained:** many pages look relevant, and qualifiers (a date, product, region) rule out all but a few.
- **Completeness:** the answer is spread over several pages (6.2 on average here), and missing one breaks it.
- **Other:** miscellaneous (off-topic pages in odd folders), intra-document reasoning (two distant parts of one page), project-related (several pages within one project) and conflicting info (a newer page supersedes an older one).

**Every gold page is indexed.** All 134 gold references in `confluence` and all 12 in `metadata` are in the corpus and in Chroma, so no miss comes from a missing page. Six `confluence` gold pages are misfiled (`original_location` is set) and four are flagged as noise. The baseline answered 6 of 6 and 3 of 4 of those correctly, because search ignores folders. Report them separately in folder-navigation conditions.

**Semantic questions mostly fail on outdated look-alikes.** 6 of the 15 were wrong:
- In 4, the agent answered from an older page on the same topic, in the same folder. The gold page is the newer one in every case. For example, the agent used the Optimize Conductor 2.0 release notes instead of 4.0, and a 2025 policy ("2 business days") instead of the 2026 one ("72 business hours").
- In those 4, the gold page never reached the top 5 for the agent's queries. The agent also can't see dates (`show_metadata = false`), so it couldn't have told current from outdated anyway. This is what the metadata-visible condition tests.
- The agent's query rewrites can hurt. In `qst_0252`, the original question ranks the gold page #2, but the agent's rewrites (quoted phrases, keyword lists) ranked it 10th or lower.
- In the other 2, the agent cited the right page and was judged wrong on a detail. These are borderline and may flip with a stronger judge.

## How the agent works

A deliberately minimal harness: one model, three tools, a fixed tool budget.

```mermaid
flowchart TD
    Q([Question]) --> M["Model turn<br/>OpenAI Responses API"]
    M -->|"search(query)"| S["search<br/>BM25 + dense vectors in Chroma, fused with RRF<br/>returns the top 5 full pages"]
    M -->|"read_document(doc_id)"| R["read_document<br/>returns one full page"]
    M -->|"finish(answer, doc_ids)"| F(["Done<br/>answers.jsonl + metrics + trace"])
    M -.->|"plain text, no tool call"| N["Nudge: call finish<br/>(only once, a second text reply counts as the answer)"]
    S --> C["Append the tool outputs and the model's<br/>reasoning items to the conversation"]
    R --> C
    C --> B{"15 tool calls used?"}
    B -->|no| M
    B -->|yes| X["Next turn may only call finish<br/>(forced via tool_choice)"]
    X --> M
    N --> M
```

- **One request per turn.** The model can call several tools in parallel within a turn, and each call is run and answered separately.
- **The conversation only grows.** Tool outputs are appended, and so are the model's reasoning items (encrypted, since requests use `store=False`). A reasoning model therefore keeps its train of thought across tool calls.
- **The budget is enforced per call.** If parallel calls would exceed it, the extra calls get a "budget exhausted" reply instead of running. Set the budget with `agent.max_tool_calls`.
- **Where the code lives:**
  - `agent.py`: the loop
  - `tools.py`: tool definitions and how results are rendered
  - `search.py`: hybrid search

## Setup

Requires [uv](https://docs.astral.sh/uv/), which also installs the pinned Python 3.14 if needed.

```bash
uv sync
cp .env.example .env   # fill in OPENAI_API_KEY, JUDGE_MODEL (and OPENAI_BASE_URL for a company endpoint)
```

Everything needed is in this repo:
- the corpus in `data/confluence/`
- the question sets with gold answers in `data/questions/`
- the evaluator

No EnterpriseRAG-Bench checkout is required. Both datasets come from [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) (MIT); see [`data/questions/README.md`](data/questions/README.md) for provenance.

## 1. Build the index (once per text variant)

```bash
uv run python -m organizing_knowledge_for_agents.index            # text-embedding-3-large, variant m0 (title + content)
```

This embeds 5,189 docs (about 11.5M tokens) into `data/chroma/`. If it's interrupted, re-running the same command resumes.

Embeddings use the same OpenAI endpoint and key as the agent, unless `EMBED_BASE_URL` / `EMBED_API_KEY` are set:
- On Azure, set `EMBED_MODEL` to the deployment name.
- The longest page is about 6,600 tokens, under the model's 8,191-token input limit, so nothing is truncated.
- `text-embedding-3-large` is also what EnterpriseRAG-Bench's own vector baseline uses, so dense-retrieval results are comparable.

## 2. Run the agent

```bash
# pilot first: measure cost per question before committing to the grid
uv run python -m organizing_knowledge_for_agents.run --config configs/baseline.toml --questions confluence --limit 10

uv run python -m organizing_knowledge_for_agents.run --config configs/baseline.toml --questions confluence --repeat 0
uv run python -m organizing_knowledge_for_agents.run --config configs/baseline.toml --questions not_found  --repeat 0
uv run python -m organizing_knowledge_for_agents.run --config configs/baseline.toml --questions metadata   --repeat 0
```

| Question set | File | Count | What it tests |
|---|---|---|---|
| `confluence` | `data/questions/confluence.jsonl` | 64 | Questions whose gold docs are all Confluence pages |
| `not_found` | `data/questions/not_found.jsonl` | 20 | Unanswerable questions (abstention control) |
| `metadata` | `data/questions/metadata.jsonl` | 12 | Questions that depend on page metadata (space, author, status, dates) |

Output goes to `runs/<config>/<model>/<question set>/r<repeat>/`:

- `answers.jsonl`: the evaluator's input format.
- `metrics.jsonl`: tokens, cached tokens, tool calls, docs seen and latency per question.
- `summary.json`: aggregates over the run.
- `traces/`: full agent transcripts, minus encrypted reasoning.

Runs resume, and questions that errored are retried on the next invocation. Agent runs are nondeterministic, so repeat each cell (`--repeat 1`, `--repeat 2`, ...). `runs/` holds generated outputs and is gitignored.

## 3. Evaluate

```bash
uv run python -m organizing_knowledge_for_agents.evaluate --run-dir runs/baseline/GPT-6-luna/confluence/r0
```

The judge model comes from `JUDGE_MODEL` in `.env`, or from `--judge-model`. Pin one judge for the whole study, and don't use the small agent model as the judge. The evaluator refuses to mix judges within a run directory. It uses the same OpenAI endpoint as the agent.

Scoring reproduces EnterpriseRAG-Bench's `metrics_based_eval` with `--no-correction`. It uses the bench's judge prompts verbatim (`eval_prompts.py`) and the same rules:

| Metric | How it's computed |
|---|---|
| `answer_correct` | An LLM judge decides whether the answer agrees with the gold answer, after citations are stripped. |
| `completeness_pct` | Share of the gold `answer_facts` that the judge finds in the answer. |
| `document_recall_pct` | Share of gold docs among the returned doc IDs. Only for questions with gold docs. |
| `invalid_extra_docs` | Returned docs that aren't gold. |

The gold set is never rewritten, so every cell is scored against the same ground truth. Results go to `eval.jsonl` (per question; re-running resumes) and `eval_summary.json` (overall and per question type) in the run directory.

## Layout

```
src/organizing_knowledge_for_agents/
  settings.py   paths, env, experiment config (TOML -> dataclasses)
  corpus.py     load + clean docs (whitelisted metadata only), text variants
  questions.py  question sets (data/questions/)
  embeddings.py OpenAI embeddings client (text-embedding-3-large)
  index.py      embeddings -> Chroma indexing (resumable)
  search.py     hybrid search (dense + BM25, RRF)
  tools.py      agent tools and their rendering; grid toggles live here
  agent.py      Responses API loop with tool budget and forced finish
  run.py        runner: parallel, resumable, writes answers/metrics/traces
  evaluate.py   scores a run: correctness, completeness, doc recall (bench-compatible)
  eval_prompts.py  the bench's judge prompts, verbatim
  jsonl.py      JSONL helper
  import_questions.py  regenerates data/questions/ from a bench checkout (rarely needed)
configs/        one TOML per grid cell
data/confluence the corpus (raw EnterpriseRAG-Bench JSON)
data/questions  question sets with gold answers
licenses/       EnterpriseRAG-Bench MIT license (data and judge prompts)
tests/          agent loop, evaluator, embeddings and corpus tests (no API keys needed)
```

## Adding a grid condition

1. Copy `configs/baseline.toml` and give it a unique `name`.
2. Change one dimension:
   - **Progressive disclosure:** `tools.result_view = "preview"`. Search then returns previews, and the agent must call `read_document`.
   - **Metadata visible:** `tools.show_metadata = true`.
   - **New indexed text** (e.g. metadata in the embedding): add a variant in `corpus.render_text`, index it with `--variant`, and set `retrieval.text_variant`.
   - **New tools** (folder navigation, metadata filters, outlines): add them in `tools.py` behind a `ToolsConfig` flag.

The raw JSON in `data/confluence/` contains generator artifacts. `original_location` is the pre-noise folder of a misfiled doc, and `dataset_noise_document` marks noise docs. `corpus.load_docs` keeps only whitelisted metadata fields, so neither reaches the agent. Keep it that way, unless a condition deliberately uses `original_location` as an oracle.

```bash
uv run pytest            # no API keys or network needed
```
