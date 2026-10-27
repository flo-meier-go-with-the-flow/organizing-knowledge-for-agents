# Confluence content-metadata taxonomy — Stage 0 proposal

Sample: **80 docs**, round-robin across all 19 spaces (seed 0), judged from title + body only (bodies cut at 12,000 chars; 3 docs were cut).
Per-doc results: `confluence.jsonl` (same folder). Corpus-wide counts are regex checks over all 5,189 docs.
Docs 0–39 were the first sample; docs 40–79 were added with the same prompt to check how stable the results are.

## Headline findings (stable across both halves)

| | docs 0–39 | docs 40–79 | all 80 |
|---|---|---|---|
| nature = process | 22 | 24 | 46 (58%) |
| nature = none | 5 | 5 | 10 (13%) |
| lifecycle current / vision / unclear / **archived** | 35 / 4 / 1 / **0** | 36 / 3 / 1 / **0** | 71 / 7 / 2 / **0** |
| sensitivity internal / confidential / **public** | 28 / 12 / **0** | 34 / 6 / **0** | 62 / 18 / **0** |
| sensitivity agrees with declared confidentiality | 33 | 37 | 70 (88%) |
| `audience` suggested | 38 | 37 | 75 (94%) |

1. **"Playbook" is the corpus's house style, not a type.** 1,516 titles (29%) contain it; in the sample it labels runbooks, finance procedures, HR guides, manager guides and sales material. The classifier must be told to ignore the word.
2. **`classification` mixes two axes**, confirmed by the second half. `process` swallows ~58%, and every `none` is a plan, requirements doc, roadmap, sales material or workplace guideline. *fact vs changing* is volatility; *architecture / process / decision / narrative* is genre.
3. **`status` = archived never occurs** (0/80; 1 title corpus-wide). Lifecycle is ~89% current, ~9% vision.
4. **`data_classification` mostly agrees with the declared value.** It diverges on 10/80: 7 declared-`internal` docs read as confidential (pricing, invoicing, cash-flow, security playbooks), and 3 `restricted` docs read as internal. `public` never occurs.
5. **Sales material is labelled inconsistently** (fact / process / none across 5 docs), so the prompt needs an explicit rule for it.

## 1. document_type (closed, 13 values incl. `other`)

Coverage check: 37 of the 40 new docs map cleanly onto the list proposed after the first 40. The 3 that didn't were all guides for internal tools, which is why `tool_guide` was added.

| value | definition | absorbs sampled labels (80 docs) |
|---|---|---|
| `runbook` | step-by-step operation of a system, incident, upgrade or test | runbook, incident_runbook, test_runbook, ops/operator/oncall playbooks, game_day_playbook, migration_playbook, upgrade_strategy_guide, security_forensics_playbook |
| `design_doc` | how a system works or is built | design_doc, architecture_overview, design_pattern_and_runbook, design_and_ops_playbook, technical_reference, framework_guide, evaluation_framework |
| `technical_spec` | normative contract, schema, standard, SLO, requirements, catalog or benchmark | technical_spec, technical_contract, technical_standard, observability_contract, audit_event_schema_spec, slo_definition, network_requirements_spec, benchmark_spec, compatibility_catalog, design_spec |
| `tool_guide` | how to use an internal tool, simulator or library | tooling_guide, tool_guide, tooling_catalog |
| `decision_record` | recorded decision with options and rationale (ADR) | adr (3) |
| `policy` | rules, compliance procedures, exception processes | policy (5), finance_policy, compliance_procedure, exception_process |
| `business_procedure` | non-engineering how-to: finance, CS, HR, manager processes, checklists | finance_playbook, handoff_playbook, hr_playbook, hr_process_playbook, manager_playbook, sop, checklist |
| `onboarding_guide` | orientation for new hires, new tenants, first-run users | onboarding_guide, onboarding_playbook, onboarding_access_playbook, employee_handbook, quickstart_guide |
| `sales_enablement` | pitch, pricing, competitive or buyer material for sellers | sales_playbook, sales_enablement_guide, sales_enablement_brief |
| `proposal_or_plan` | proposed product/pricing/launch plan, PRD, roadmap | pricing_proposal, launch_playbook, prd, roadmap |
| `postmortem` | incident review, root cause, retrospective | retrospective (1); 84 corpus titles (1.6%) say postmortem/incident review |
| `faq` | question-and-answer reference | 0 sampled; 30 corpus titles (0.6%) contain FAQ |
| `informal_notes` | office lore, workplace guidelines, loose tips | miscellaneous_notes, informal_tips, facilities_guideline, workplace_guidelines |
| `other` | none of the above | — |

`postmortem` and `faq` are kept from corpus counts. Both are rare enough (1.6% and 0.6%) that ~1 or fewer would be expected in 80 docs. `meeting_notes` (4 corpus titles) and `announcement` stay dropped.

## 2. classification: split into two fields

**`classification` (genre, 7 values):**
- `reference`: stable facts, contracts, schemas, catalogs, **and sales/enablement material**
- `procedure`: how-to, runbook, checklist
- `policy`: rules people must follow, including workplace guidelines
- `design`: system or architecture description
- `decision`: a decision and its rationale
- `plan`: proposal, PRD, roadmap, launch plan
- `narrative`: retros, notes, informal discussion

On the 80 docs, all 10 `none` answers get a home: `plan` takes 5 (product_proposal, product_requirements, roadmap, feature_launch_plan, requirements_specification), the sales rule 2, `narrative` 2 (mixed_informal_notes, office_lore) and `policy` 1 (workplace_policy). `policy` also splits off the ~9 policy/compliance docs that were counted under `process`.

**`volatility` (new, 2 values):** `stable` (definitions, contracts, policies, decisions) vs `changing` (thresholds, configs, contacts, tooling, runbooks likely to drift).

## 3. status and data_classification

**status (current / archived / vision).** Confirmed: archived is 0/80. Expect ~89% current / ~9% vision / ~2% unclear unless archived docs are added. Prompt cues:
- vision: "proposal", "PRD", "roadmap", "target state", "pilot", "v1 draft", future-dated plans. 108 corpus titles match proposal/roadmap/draft.
- archived: only when the doc says *it* is deprecated, superseded or retired. A doc describing a decommission or migration is current.

**data_classification (public / internal / confidential).** Keep. Confidential cues: pricing/discounts, invoicing and revenue figures, offer/compensation rationale, security-exception and forensics paths, customer-specific network/deploy details. `public` stays unused (0/80).

## 4. Additional metadata groups

Coverage after merging synonyms, over all 80 docs:

| group (merged facets) | docs | vocabulary | verdict |
|---|---|---|---|
| `audience` (audience) | 75 | closed, multi-valued: `all_employees`, `new_hires`, `people_managers`, `engineering`, `sre_oncall`, `ml_engineering`, `sdk_developers`, `customer_success`, `sales`, `finance_legal`, `security_compliance`, `customers_external` | **add**: near-universal, not in existing metadata |
| `product_area` (product_area, deployment_mode, deployment_tier, deployment_type) | 42 | closed: `inference_runtime`, `sdk`, `dedicated_capacity`, `private_deployment`, `evals`, `observability`, `billing_metering`, `gpu_fleet`, `none` | **add**: grew from 18/40 to 42/80 |
| `deployment_mode` (part of the above) | 18 | closed: `hosted_shared`, `hosted_dedicated`, `private_vpc`, `air_gapped`, `edge_onprem`, `any` | optional; or fold into `product_area` |
| `system` (service, tool, component) | 23 | **open**, normalized snake_case service names | optional; strongest handle for "which doc covers service X" |
| team family (team, owning_team, owner_team, department, related_team) | 66 | — | **reject for retrieval**: duplicates existing `owner_team` / `space`; use only as a Stage 4 drift check |
| `topic` | 33 | — | **reject**: free keywords, duplicates `labels` |
| `time_period` (time_period, cadence) | 11 | — | reject: sparse, covered by `created_at` |
| `compliance_area`, `incident_type` | 7, 5 | — | reject for now: only relevant inside security / on-call spaces |
| one-offs (`hardware`, `buyer_persona`, `environment`, `adr_number`, …) | ≤2 | — | reject: too specific, already in body text |

## 5. Open questions for you

1. Split `classification` into genre + `volatility` (recommended), or keep a single field?
2. Archived is 0/80. Accept that, or create or tag archived docs so `status` has signal?
3. Which additional groups to add: `audience` + `product_area` (recommended), plus `system` and/or `deployment_mode`?
4. `audience` as a list (recommended: most docs name 2+ audiences) or a single primary audience?
5. Sales material: genre `reference` (as proposed), or its own `enablement` genre?
