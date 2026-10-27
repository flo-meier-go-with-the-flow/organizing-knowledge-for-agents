# Task

## Goal

Use EnterpriseRAG-Bench to answer:
- What are best practices for structuring data so that agentic retrieval works best?
- What knowledge organization is optimal for agentic retrieval?

Directions:
- progressive disclosure
- better metadata for documents/chunks
- a folder structure that helps agents navigate

Approach: evaluate on the Confluence pages only. Build an agentic-retrieval baseline, then vary metadata, retrieval strategy (baseline vs progressive disclosure) and folder structure along a grid.

## Data

- **Questions:**
  - 64 questions with `source_types == ["confluence"]`: basic 19, semantic 15, constrained 11, completeness 11, other types 1–3 each.
  - 12 metadata questions (`extra_questions.jsonl`).
  - 20 `info_not_found` questions as an abstention control.
- **Corpus:** 5,189 docs in `data/confluence/`, organized as 19 spaces with one level of subfolders. Docs average about 2.5k tokens, so one doc ≈ one chunk.
- **Native metadata:** about 99% complete: space, author, owner_team, status, dates, reviewers, labels, related_pages, confidentiality.
- **Noise:** 7.6% of docs are misfiled. 8.8% have a `space` field that disagrees with their folder.
- **Reference point:** BM25 alone puts at least one gold doc in the top 5 for 47 of the 64 questions.

## Pitfalls

1. **Leak fields.** `original_location` (each misfiled doc's correct folder) and `dataset_noise_document` are generator artifacts. The loader strips them. Use `original_location` only as a deliberate oracle condition.
2. **Filenames confound folders.** Filenames are LLM-written summaries. Hold them constant, or ablate them on purpose.
3. **Ceiling effects.** The search space is 5k docs, and lexical retrieval is already strong. Treat efficiency as a primary metric, and use tool budgets or a weaker model to separate the conditions.
4. **Gold drift.** Always evaluate with `--no-correction`. Otherwise cells end up scored against different gold sets.
5. **Statistics.** With n=64, the margin of error is about ±10pp. Use the same questions in every cell, run ≥3 repeats, and use paired bootstrap/McNemar tests. No per-category conclusions.
6. **Overfitting.** Freeze the grid before the first full run, or hold out about 20 questions as a dev set.
7. **Restructuring breaks links.** `related_pages` and in-body links are paths, so rewrite them when moving files. The metadata questions name spaces explicitly.
8. **Folders matter only if the agent navigates.** Decide whether per-folder README/index files belong to the folder dimension or the progressive-disclosure dimension.
9. **Metadata has two axes:** which fields exist, and how the agent can use them (inline text vs search filters). Report the enrichment cost per doc.
10. **External validity.** The corpus is LLM-generated and unusually uniform. Don't call the result an "ideal structure".

## Grid (staged, not full factorial)

- **R (retrieval):**
  - R0: baseline, where search returns full docs
  - R1: progressive disclosure (previews, then read/outline)
- **M (metadata):**
  - M0: title and body only
  - M1: M0 plus native metadata
  - M2: M1 plus LLM enrichment (summary, doc_type, entities, supersedes)
- **F (folders):**
  - F0: flat
  - F1: as-is
  - F2: de-noised via `original_location` (oracle)
  - F3: LLM-built taxonomy with folder READMEs

Order: pilot (10 questions) → baseline → one-factor-at-a-time sweeps → small factorial on the best levels.

**Metrics:**
- correctness, completeness and doc recall from the bench's evaluator
- efficiency: tool calls, tokens, $/question, latency, docs seen
- abstention on `info_not_found`
- accuracy vs cost as a Pareto plot

## Decisions

- **Stack:**
  - Chroma for vectors, OpenAI `text-embedding-3-large` for embeddings (same model as the bench's vector baseline), in-memory BM25, fused with RRF.
  - No truncation is needed: the longest page is about 6,600 tokens, under the 8,191-token input limit.
  - A small OpenAI model from the company deployment runs the agent loop (Responses API) and writes answers.
- **Baseline tools:**
  - `search`: top 5, full text, no metadata
  - `read_document`
  - `finish(answer, doc_ids)`
  - 15 tool calls, then a forced finish.
- **Grid cells are config toggles** in `configs/*.toml`. The agent loop is identical everywhere.
- **Eval:** the bench's `metrics_based_eval` with `--no-correction` and one pinned judge model that is not the agent model.
- **Hosting:** GitHub is the source of truth. The company GitLab mirrors it (pull mirror, or a scheduled CI job on the Free tier). Contributions go through GitHub PRs.

## TODO

Setup
- [ ] Choose the agent model and endpoint (OpenAI vs Azure → `OPENAI_BASE_URL`) and set it in `configs/baseline.toml`.
- [ ] Confirm `text-embedding-3-large` is deployed on the company endpoint, and note the deployment name if it's Azure.
- [ ] Choose and pin the judge model.
- [ ] Decide repo visibility, and whether the benchmark corpus stays committed in a public repo.
- [ ] Set up the GitLab mirror from GitHub.
- [ ] Build the index: `uv run python -m organizing_knowledge_for_agents.index`.

Baseline
- [ ] Pilot on 10 questions: read traces, measure cost per question, classify failures.
- [ ] Full baseline on all three question sets, 3 repeats each. Evaluate.
- [ ] Reference run with the largest deployable model.

Grid
- [ ] Freeze the grid and metrics (optionally hold out a dev set).
- [ ] Progressive disclosure: `result_view = "preview"`, plus an outline/read-section tool.
- [ ] Metadata: `show_metadata`, search filters, a frontmatter text variant, LLM enrichment via batch.
- [ ] Folders: a `list_dir` tool and paths in results. Build F0–F3, rewriting path links.
- [ ] Analysis: paired tests across cells, Pareto plot.
