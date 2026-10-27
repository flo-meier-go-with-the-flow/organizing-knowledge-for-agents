# Project plan: organizing knowledge for agents

*Retroactive plan, written on 2026-10-27 after the first two days of work. It records the goals, the approach, the decisions taken so far and why, and what comes next. Update the status section and the decision log as the project moves on.*

## 1. Goal

Find out how to structure company knowledge so that agentic retrieval works best. Two questions:

1. What are best practices for structuring data for agentic retrieval?
2. What knowledge organization is optimal for agentic retrieval?

Three directions to test:

- **Progressive disclosure:** the agent sees summaries first and opens full documents only when needed.
- **Metadata:** which metadata documents carry, and how the agent can use it.
- **Folder structure:** which folder layout helps an agent navigate.

## 2. Scope

**Dataset.** [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench), restricted to its Confluence pages: polished wiki documentation, the closest match to the "organized knowledge" this project is about.

| Question set | Source | Count | Purpose |
|---|---|---|---|
| `confluence` | `questions.jsonl`, `source_types == ["confluence"]` | 64 | Main test set: basic 19, semantic 15, constrained 11, completeness 11, other types 1–3 each |
| `metadata` | `extra_questions.jsonl`, Confluence only | 12 | Directly tests the metadata dimension (filters on space, author, status, dates) |
| `not_found` | `questions.jsonl`, `info_not_found` | 20 | Abstention control: does more structure make the agent confidently wrong? |

**Corpus** (`data/confluence/`):
- 5,189 pages in 19 top-level spaces with one level of subfolders (103 in total).
- Pages are short: median 2,180 tokens, longest 6,596. One page fits in one chunk.
- Native metadata is about 99% complete: space, author, owner_team, status, dates, reviewers, labels, related_pages, confidentiality.
- Built-in noise: 7.6% of pages are misfiled, and 8.8% have a `space` field that disagrees with their folder.

**Reference point.** BM25 alone, without an agent, puts at least one gold page in its top 5 for 47 of the 64 questions. Lexical retrieval is already strong on this corpus.

## 3. Approach

1. Build one agentic-retrieval baseline.
2. Vary one structural dimension at a time, keeping the agent, models and evaluation fixed.
3. Compare cells on answer quality and on efficiency.

### Baseline system

- **Retrieval:** hybrid search. Dense vectors from OpenAI `text-embedding-3-large` stored in Chroma, plus in-memory BM25, fused with reciprocal rank fusion. Search returns the top 5 pages.
- **Agent:** a small OpenAI model from the company deployment, in our own loop on the Responses API. Tools:
  - `search`: full text of the top 5 pages, no metadata.
  - `read_document`: one page by ID.
  - `finish(answer, doc_ids)`: submits the answer.

  The budget is 15 tool calls, after which the agent must call `finish`.
- **Runner:** runs a question set with parallelism and resume, and writes `answers.jsonl` in the format the bench's evaluator expects. It also logs per-question efficiency metrics and full transcripts.
- **Evaluation:** the bench's `metrics_based_eval` with `--no-correction` and one pinned judge model that is not the agent model.

### Experiment grid

| Dimension | Levels |
|---|---|
| **R: retrieval** | R0 baseline (search returns full pages) · R1 progressive disclosure (previews first, then read/outline) |
| **M: metadata** | M0 title and body only · M1 plus native metadata · M2 plus LLM enrichment (summary, doc_type, entities, supersedes) |
| **F: folders** | F0 flat · F1 as-is · F2 de-noised via `original_location` (oracle upper bound) · F3 LLM-built taxonomy with folder READMEs |

Each cell is a TOML file in `configs/` that toggles tools or the indexed text. A full factorial would be about 36 cells × 96 questions × 3 repeats ≈ 10k agent runs, so the grid runs in stages:

1. Pilot on 10 questions, to measure cost per question and read failure modes.
2. Full baseline.
3. One-factor-at-a-time sweeps from the baseline.
4. A small factorial on the best levels of each factor.

### Metrics and analysis

- **Quality:** correctness, completeness and document recall from the bench's evaluator.
- **Efficiency:** tool calls, tokens, cost per question, latency, pages seen. Efficiency is progressive disclosure's main promise, and accuracy may saturate on a 5k-page corpus.
- **Abstention** on the `not_found` set.
- **Analysis:** paired comparisons (same questions in every cell), at least 3 repeats per cell, paired bootstrap or McNemar tests, and an accuracy-vs-cost Pareto plot. No per-category conclusions, since the categories are too small.

## 4. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-26 | Evaluate on the 64 Confluence-only questions, plus the 12 metadata and 20 not-found questions | Keeps the corpus small enough to iterate on. The extra sets test metadata and abstention at no extra cost. |
| 2026-10-26 | Write our own agent loop on the OpenAI Responses API instead of using the bench's shell agent | Every cell runs the identical loop, and grid conditions become tool toggles. The Responses API keeps a reasoning model's reasoning across tool calls. |
| 2026-10-26 | Agent and answer writing: a small OpenAI model from the company deployment | Deployable inside the company. A weaker model also makes structure effects easier to see, given the ceiling risk. |
| 2026-10-26 | Chroma for vectors; hybrid dense + BM25 search fused with RRF | Chroma is simple and local. BM25 handles internal codenames and acronyms that embeddings miss. |
| 2026-10-26 | Embeddings: Voyage `voyage-4-large` | First choice. *Superseded the same day.* |
| 2026-10-26 | Embeddings: NVIDIA `nemotron-3-embed-1b` via a self-hosted NIM | Planned company deployment. It would have needed truncation at the NIM's 4,096-token limit. *Superseded on 2026-10-27.* |
| 2026-10-27 | Embeddings: OpenAI `text-embedding-3-large` | Runs on the same endpoint as the agent. It's the model the bench's own vector baseline uses, so dense results are comparable. The longest page (6,596 tokens) fits its 8,191-token limit, so nothing is truncated. |
| 2026-10-26 | Strip generator artifacts in the loader and keep only whitelisted metadata | `original_location` reveals each misfiled page's correct folder, and `dataset_noise_document` marks noise. Neither may reach the agent. `original_location` is used only for the F2 oracle. |
| 2026-10-26 | Evaluate with `--no-correction` and one pinned judge that differs from the agent | Without it, the evaluator can rewrite gold sets per run, so cells would be scored against different ground truths. A separate judge avoids the agent grading itself. |
| 2026-10-26 | Grid as config toggles, run in stages instead of a full factorial | A full factorial is about 10k agent runs before judging. |
| 2026-10-26 | GitHub is the source of truth, and the company GitLab mirrors it | Contributors without GitLab access can work via GitHub pull requests. A firewalled GitLab can pull from GitHub, but GitHub can't push into it. |
| 2026-10-26 | Corpus committed under `data/confluence/`; questions and the evaluator stay in a bench checkout | Gold questions and answers stay out of this repo, and out of public view. |

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Leaked generator fields give away answers or the clean folder layout | Whitelist-based loader; tests check that the leak fields never reach the agent |
| Filenames are LLM-written summaries, confounding the folder experiment | Hold filenames constant, or ablate them on purpose |
| Ceiling effects: small corpus, strong lexical baseline | Treat efficiency as a primary metric; use tool budgets and a weaker agent model; do a reference run with a large model |
| Gold sets drifting between cells | Always `--no-correction` |
| Low statistical power (n=64, about ±10pp) | Paired design, at least 3 repeats, paired tests |
| Overfitting the design to the test questions | Freeze the grid before the first full run, or hold out about 20 questions as a dev set |
| Moving files breaks path-based links (`related_pages`, in-body links) | Rewrite links when building F0–F3. Note that the metadata questions name spaces explicitly. |
| Folders and progressive disclosure blur together | Folder structure only matters if the agent can navigate. Decide which dimension owns per-folder README/index files. |
| Metadata conflates "which fields" with "how they're used" | Treat them as two axes (inline text vs search filters); report enrichment cost per page |
| Limited external validity: the corpus is LLM-generated and uniform | State this in the write-up; don't claim an "ideal structure" |

## 6. Status and next steps

*As of 2026-10-27.*

Done
- [x] Scope, grid, metrics and evaluation protocol defined.
- [x] Baseline scaffold (branch `feat/baseline-scaffold`): corpus loader, hybrid search, agent loop, runner and tests.
- [x] Embedding model settled: `text-embedding-3-large`.

Setup
- [ ] Choose the agent model and endpoint, and set it in `configs/baseline.toml`. For Azure, set `OPENAI_BASE_URL`.
- [ ] Confirm `text-embedding-3-large` is deployed on the company endpoint, and note the deployment name if it's Azure.
- [ ] Choose and pin the judge model.
- [ ] Decide repo visibility, and whether the corpus stays committed in a public repo.
- [ ] Set up the GitLab mirror from GitHub.
- [ ] Build the index: `uv run python -m organizing_knowledge_for_agents.index`.

Baseline
- [ ] Pilot on 10 questions: read transcripts, measure cost per question, sort failures into tool-use mechanics vs retrieval.
- [ ] Full baseline on all three question sets, 3 repeats each, then evaluate.
- [ ] Reference run with the largest deployable model.

Grid
- [ ] Freeze the grid and metrics (optionally hold out a dev set).
- [ ] Progressive disclosure: `result_view = "preview"`, plus an outline/read-section tool.
- [ ] Metadata: `show_metadata`, search filters, a frontmatter text variant, LLM enrichment via batch.
- [ ] Folders: a `list_dir` tool and paths in results. Build F0–F3, rewriting path links.
- [ ] Analysis: paired tests across cells, Pareto plot, write-up.
