# RAGauge — Product Requirements (PRD)

> Eval-first RAG over SEC 10-K filings. The headline is the **evaluation
> harness**, not the chatbot. This PRD opens with the founding epic; subsequent
> sections (features, milestones, acceptance criteria) hang off it.
>
> See [`DESIGN.md`](./DESIGN.md) for architecture, retrieval design, and the
> eval-harness flow.

---

## Epic: Eval-first RAG over SEC 10-K filings

### Summary
Build a retrieval-augmented QA system over SEC 10-K filings whose defining
feature is a **measurement harness** that proves the system is reliable: a
hand-verified golden set, retrieval + groundedness metrics, an ablation that
shows each retrieval stage earns its complexity, a cost/latency/quality
dashboard, and a CI gate that blocks quality regressions. The chatbot is the
*test subject*; the evidence that it works is the *product*.

### Business value

This is a **job-search artifact**. Its job is to make a hiring panel believe one
specific claim: *this person can make an LLM system measurably reliable.* That is
a different — and rarer — claim than *this person can wire up a chatbot.*

- **Differentiation in a crowded field.** The market is saturated with RAG demos
  that wrap a vector DB and an LLM call. They look identical and prove nothing.
  RAGauge leads with measurement, which immediately reads as production
  experience rather than tutorial completion.
- **Proof of the skill that production teams actually pay for.** Companies
  deploying LLMs are bottlenecked on *reliability and evaluation*, not on getting
  a model to emit text. A portfolio that demonstrates golden sets, groundedness
  scoring, abstention, ablation, and a regression gate speaks directly to that
  pain.
- **Evidence over assertion.** Every claim in the writeup is backed by a number:
  recall@5, groundedness rate, unsupported-claim rate, $ per eval run, p95
  latency. "I made it reliable" becomes "here is the ablation table and the CI
  gate that blocks regressions."
- **Interview surface area.** The design is engineered to invite the *right*
  questions (chunking trade-offs, when BM25 beats dense, who judges the judge,
  CI thresholds) and to have a defensible answer for each — turning the project
  into a 45-minute technical conversation the candidate controls.
- **Honest scope.** A small, finished, deeply-measured system beats a sprawling,
  half-built one. Shipping a defensible MVP is itself the signal.

### User stories

#### (a) Recruiter / hiring manager reviewing the repo

- **As a hiring manager skimming the README in 90 seconds,** I want to see the
  eval results — ablation table, groundedness, cost/latency — up front, so that I
  can tell within a minute that this candidate measures their systems rather than
  just demos them.
- **As a technical reviewer reading the code,** I want clean component boundaries
  (ingest / retrieve / generate / eval) and typed data contracts, so that I can
  trust the candidate writes maintainable systems, not notebook spaghetti.
- **As an interviewer preparing questions,** I want a visible ablation proving
  each retrieval stage earns its complexity, so that I can probe the candidate's
  judgment about when *not* to add a stage.
- **As a skeptic of LLM demos,** I want to see a first-class "insufficient
  evidence" path and an unsupported-claim-rate metric, so that I believe the
  candidate understands hallucination is the failure mode that matters.
- **As an engineering manager,** I want a CI gate that fails the build on quality
  regression, so that I see the candidate thinks in terms of production guardrails,
  not one-off testing.
- **As a reviewer worried about integrity,** I want evidence the golden set was
  hand-verified, so that I trust the metrics aren't measuring against AI-generated
  noise.

#### (b) End user asking questions of SEC filings

- **As an analyst,** I want to ask a natural-language question about a 10-K and
  get an answer **with inline citations to the source chunks**, so that I can
  verify the answer against the filing instead of trusting it blindly.
- **As a user asking about a specific figure,** I want exact financial terms and
  numbers retrieved correctly (not just semantically-close prose), so that I get
  the right line item, not an adjacent one.
- **As a user asking something the filings don't answer,** I want the system to
  say **"insufficient evidence"** instead of inventing a plausible number, so
  that I am never confidently misled.
- **As a user with a cross-section question,** I want the system to combine
  evidence across sections/filings (multi-hop), so that I can answer questions
  that no single passage covers.
- **As a user comparing companies,** I want section-aware answers (a risk-factors
  question answered from risk factors, not MD&A), so that the context is correct.

### Success metrics

The project is successful when these are **measured, reported, and gated** — not
merely achieved once.

| Metric | What it proves | Target / treatment |
|---|---|---|
| **recall@5** | Retrieval surfaces the gold evidence | Reported per ablation config; a CI floor enforced |
| **Groundedness / supported-claim rate** | Answers are backed by cited evidence | Reported via LLM-as-judge (structured output) |
| **Unsupported-claim rate** | The hallucination signal | Trend toward zero; a CI ceiling enforced |
| **Unanswerable-precision** | Abstains correctly (no LLM needed) | Reported on the ~15% unanswerable rows |
| **$ per eval run** | Real cost via provider token-counting | Reported in the dashboard; tracked across model sweep |
| **p95 latency** | Per-stage and end-to-end responsiveness | Reported; rerank's latency cost made explicit |
| **Recall lift per stage** | Each retrieval stage earns its complexity | The ablation table is the headline artifact |

Definition of done for the epic: a reviewer can open the repo and, from the
README alone, see the ablation table, the groundedness/cost/latency dashboard,
and a passing (and demonstrably blocking) CI gate.

### Subtask checklist (living progress tracker)

This is the working backlog — **update it as work lands** (check the box; add
`(partial: …)` or `(deviation: …)` notes inline). It decomposes the build order
([`DESIGN.md`](./DESIGN.md) §12) into focused-session-sized tasks. Each carries
**Acceptance** (done-when), **Metric** (the headline number it moves, or `—` for
plumbing), and **Depends on**.

**Ordering principle — fastest signal first.** Tasks are sequenced so the first
*demonstrable* artifact a hiring manager values appears as early as the
dependency graph allows: a real **recall@5 baseline number (T10)** lands before
any generation code, and the **ablation table (T20)** — the thesis made visible —
is reached on the shortest viable path. `★` marks a task that produces a
reviewer-facing artifact or headline number; the rest are the plumbing that earns it.

**Status legend:** `- [ ]` pending · `- [x]` done · annotate partials inline.
**Progress at a glance:** 0 / 23 done — next up: **T1**.

#### Phase 0 — Foundations (clean boundaries = architect signal)

- [ ] **T1 · Project scaffolding & dependencies.** Python project layout
  (`pyproject`/lockfile), `.env` loading, component dirs (`ingest/ retrieve/
  generate/ eval/`), test runner wired.
  - *Acceptance:* `pytest` runs green on a placeholder test; deps install clean from a fresh checkout; `.env` read without committing secrets.
  - *Metric:* — · *Depends on:* none
- [ ] **T2 · Data-contract types.** Implement the typed records that cross
  boundaries: `Chunk`, `RetrievedChunk`, `Answer`, `GoldRow`, `RunReport`
  (Pydantic), incl. the content-derived **stable `chunk_id`** scheme.
  - *Acceptance:* types importable + round-trip (de)serialize; `chunk_id` is deterministic from content and stable across re-ingestion (unit test proves it); `stage_provenance` modelled on `RetrievedChunk`.
  - *Metric:* — (load-bearing for attributable ablations) · *Depends on:* T1

#### Phase 1 — Ingest

- [ ] **T3 · Acquire & vendor 2–3 static 10-Ks.** Download once, store raw under
  the gitignored data dir; record source + corpus hash.
  - *Acceptance:* 2–3 filings on disk across ≥2 companies; a manifest notes accession/URL + corpus hash; raw files gitignored.
  - *Metric:* — · *Depends on:* T1
- [ ] **T4 · Section segmentation (Item 1 / 1A / 7 / 8).** Heading/anchor
  detection that splits a filing into Items before any size-based chunking.
  - *Acceptance:* each chosen filing splits into the target Items with correct `section` labels; validated by eyeball across all filings (boundary robustness is a known risk).
  - *Metric:* — · *Depends on:* T3
- [ ] **T5 · Structure-aware chunker + metadata. ★** Size-bounded chunking within
  sections that keeps tables coherent and footnotes with referents; attach full
  metadata (`doc_id, company, fiscal_year, section, anchor, chunk_id`).
  - *Acceptance:* chunks carry complete metadata; ≥1 documented example of a table/footnote a naive splitter would break, preserved intact (this is the §4 interview exhibit); no mid-sentence/mid-row splits in a spot-check.
  - *Metric:* — (sets the recall ceiling measured at T10) · *Depends on:* T2, T4
- [ ] **T6 · Chunk store + inspection script.** Persist chunks keyed by
  `chunk_id`; a small CLI to dump/inspect chunks for a doc/section.
  - *Acceptance:* `inspect` prints chunks with metadata for a given filing/section; store re-loads without re-parsing. *(Demo: inspect chunks.)*
  - *Metric:* — · *Depends on:* T5

#### Phase 2 — Dense retrieval baseline

- [ ] **T7 · Embed chunks + build dense index.** Embedding model + vector index;
  index is embedding-model-versioned and rebuilt on ingest.
  - *Acceptance:* index builds over the full chunk set; build is reproducible; embedding model id recorded for the run.
  - *Metric:* — · *Depends on:* T6
- [ ] **T8 · Dense retrieval top-k (Retrieve seam). ★** Config-toggleable
  `Retrieve(query, config) → RetrievedChunk[]` returning dense top-k with scores
  + provenance — the seam the harness and CLI talk to.
  - *Acceptance:* a query returns ranked `RetrievedChunk`s with `stage_provenance="dense"`; `top_k` config-driven; eyeball relevance sane. *(Demo: query it.)*
  - *Metric:* — (enables recall@5) · *Depends on:* T7, T2

#### Phase 3 — First measurable signal (no LLM — cheap, honest, fast)

- [ ] **T9 · Golden set v0, hand-verified. ★** ~30 rows (`id, question,
  gold_answer, gold_chunk_ids, type, difficulty`); ~15% `unanswerable`, several
  `multi_hop`; **AI drafts, human verifies every row**, stratified by
  type × section × difficulty.
  - *Acceptance:* ~30 rows committed as version-controlled ground truth; every row carries verified `gold_chunk_ids` (existing `chunk_id`s); coverage table shows the type/section/difficulty spread; a note records the by-hand verification. *(Artifact: the labeled set.)*
  - *Metric:* enables every downstream metric · *Depends on:* T6 (needs real `chunk_id`s); questions can be drafted earlier
- [ ] **T10 · Retrieval-metrics harness on dense-only. ★ FIRST NUMBER.**
  recall@5, MRR, unanswerable-precision computed without any LLM; reproducible
  from a config hash.
  - *Acceptance:* one command runs the golden set through dense retrieval and reports recall@5, MRR, unanswerable-precision; output tied to (config hash, corpus hash); this is the **honest dense-only baseline**. *(Artifact: the baseline.)*
  - *Metric:* **recall@5, MRR, unanswerable-precision** · *Depends on:* T8, T9

#### Phase 4 — Generation

- [ ] **T11 · Pipeline seam (Retrieve → Generate).** Single `Pipeline` object that
  composes the stages behind one call; harness/CLI talk only to it.
  - *Acceptance:* `pipeline(question | config)` runs retrieve→generate end-to-end; internal stages never called directly by harness/CLI.
  - *Metric:* — · *Depends on:* T8
- [ ] **T12 · Grounded answer with inline citations. ★** Generation (latest Claude;
  **verify model id/pricing against the live API ref before wiring**) constrained
  to retrieved evidence, citing `chunk_id`s per claim.
  - *Acceptance:* answers cite `chunk_id`s for supported claims and use only supplied chunks; `Answer` carries token/cost/latency telemetry. *(Demo: ask it.)*
  - *Metric:* — (sets up groundedness) · *Depends on:* T11
- [ ] **T13 · Insufficient-evidence path. ★** Dual-trigger abstention:
  pre-generation evidence gate (weak/low-score retrieval short-circuits before an
  LLM call) + post-generation honesty; returns structured `abstained=true` with no
  fabricated citations.
  - *Acceptance:* unanswerable golden rows return a structured abstention; pre-gen gate avoids the generation call when evidence is too weak; no citations on abstentions.
  - *Metric:* **unanswerable-precision** (now generation-side too) · *Depends on:* T12

#### Phase 5 — Generation metrics & RunReport

- [ ] **T14 · LLM-as-judge (Pydantic schema). ★** Structured claim-level
  grounded/unsupported labels that roll up into rates; judge prompt + schema
  versioned with the run.
  - *Acceptance:* judge emits validated structured output (no free-text parsing); claim labels aggregate into groundedness + unsupported-claim rate; prompt/schema version recorded.
  - *Metric:* **groundedness / supported-claim rate, unsupported-claim rate** · *Depends on:* T12
- [ ] **T15 · Ops telemetry — cost & latency.** $/run via the **provider
  token-counting API** (not a generic tokenizer); p50/p95 latency per-stage and
  end-to-end.
  - *Acceptance:* a run reports real $/run from provider token counts and p50/p95 latency broken out by stage; rerank's latency cost is attributable.
  - *Metric:* **$ per eval run, p95 latency** · *Depends on:* T12
- [ ] **T16 · RunReport assembly + persistence.** Append-only per-config report:
  per-question results + aggregates + run metadata (config hash, corpus hash,
  model id, timestamp, cost).
  - *Acceptance:* each run persists one `RunReport` reproducible from (config hash, corpus hash, model id); history is append-only for regression tracking.
  - *Metric:* — (substrate for ablation + CI) · *Depends on:* T10, T14, T15

#### Phase 6 — Hybrid retrieval & the ablation (the headline)

- [ ] **T17 · BM25 sparse index + retrieval, toggleable.** Tokenizer-versioned
  sparse index; `bm25` on/off config toggle; provenance recorded.
  - *Acceptance:* BM25 returns top-k with `stage_provenance="bm25"`; toggles independently of dense; recall@5 reportable for BM25-only. *(Demo: stage toggles.)*
  - *Metric:* recall@5 / MRR (BM25 contribution) · *Depends on:* T8
- [ ] **T18 · Reciprocal Rank Fusion, toggleable.** Fuse dense + sparse by rank;
  `fusion` on/off; fusion constant config-driven.
  - *Acceptance:* fused ranking combines both stages by rank (no score-normalization hacks); toggle on/off; provenance shows contributing stages.
  - *Metric:* recall@5 / MRR (fusion lift) · *Depends on:* T17
- [ ] **T19 · Cross-encoder rerank, toggleable.** Re-score fused top-k on a short
  list for final top-n; `rerank` on/off; latency cost surfaced.
  - *Acceptance:* rerank reorders the fused shortlist; toggle on/off; per-stage latency delta captured (feeds the trade-off story).
  - *Metric:* recall@5 / MRR (rerank lift) + p95 latency · *Depends on:* T18, T15
- [ ] **T20 · Ablation runner + table. ★ THE THESIS ARTIFACT.** Iterate configs
  (dense → +BM25/RRF → +rerank), collect `RunReport`s, emit the table: recall@5,
  MRR, unanswerable-precision, groundedness, unsupported-rate, $/run, p95 per config.
  - *Acceptance:* one command runs the config matrix and produces the ablation table showing **lift per stage**; if a stage doesn't earn its cost, the writeup says so (willingness to cut is the signal). *(Artifact: the thesis table.)*
  - *Metric:* **recall lift per stage** (+ all of the above, per config) · *Depends on:* T16, T19

#### Phase 7 — Sweep, gate, writeup

- [ ] **T21 · Model & cost sweep. ★** Compare candidate Claude models (generator
  and judge) on the fixed golden set; report $/run, p50/p95, quality.
  - *Acceptance:* a dashboard/table compares ≥2 models with quality vs. $/run vs. latency, justifying the generator/judge picks by data, not guess. *(Artifact: the dashboard.)*
  - *Metric:* **$ per eval run, p95 latency** vs. quality · *Depends on:* T16
- [ ] **T22 · CI gate (GitHub Actions). ★** Harness runs in CI and **fails the
  build** when recall@5 floor / unsupported-claim-rate ceiling are crossed;
  thresholds defensible (baseline + tolerance). Consider cheap deterministic gate
  per-PR, full judged sweep on schedule/label (cost risk).
  - *Acceptance:* a demonstrably failing PR (metric below threshold) blocks merge and a passing one merges; thresholds documented with rationale. *(Artifact: a blocking PR.)*
  - *Metric:* gates **recall@5, unsupported-claim rate** · *Depends on:* T16 (full gate after T20)
- [ ] **T23 · README headline + portfolio writeup. ★** Surface the ablation table
  + cost/latency/groundedness dashboard and the headline metrics **up top**;
  explain each design decision and what each stage earned.
  - *Acceptance:* a reviewer sees the ablation table, dashboard, and a passing/blocking CI badge from the README alone (the epic's definition of done); headline metrics current.
  - *Metric:* surfaces all headline metrics · *Depends on:* T20, T21, T22

---
