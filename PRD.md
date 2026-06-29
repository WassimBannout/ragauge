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
**Progress at a glance:** 17 / 23 done (T1–T8 + T10, T12–T19; T9, T11, T20
partial) — **Slice 1 (ingest + dense retrieval), the eval harness, and the
hybrid retrieval ablation (T17–T19) are built and tested.** The deterministic
retrieval ablation is measured: **dense recall@5 = 0.32 → hybrid (BM25 + RRF)
0.51 (+0.19) → +rerank 0.34** (the generic cross-encoder doesn't earn its cost;
see T19). The grounded generator + dual-trigger abstention + structured
LLM-as-judge + RunReport are implemented and stub-validated, awaiting an
`ANTHROPIC_API_KEY` run for the judged numbers (which fill the ablation's
groundedness columns). Next up: **human-verify the golden set**, the
**embedding-model dimension of the ablation (T20)**, then the **sweep + CI gate
(T21–T23)**. See § Implementation status for deviations.

#### Phase 0 — Foundations (clean boundaries = architect signal)

- [x] **T1 · Project scaffolding & dependencies.** Python project layout
  (`pyproject`/lockfile), `.env` loading, component dirs (`ingest/ retrieve/
  generate/ eval/`), test runner wired.
  - *Acceptance:* `pytest` runs green on a placeholder test; deps install clean from a fresh checkout; `.env` read without committing secrets.
  - *Metric:* — · *Depends on:* none
- [x] **T2 · Data-contract types.** Implement the typed records that cross
  boundaries: `Chunk`, `RetrievedChunk`, `Answer`, `GoldRow`, `RunReport`
  (Pydantic), incl. the content-derived **stable `chunk_id`** scheme.
  - *Acceptance:* types importable + round-trip (de)serialize; `chunk_id` is deterministic from content and stable across re-ingestion (unit test proves it); `stage_provenance` modelled on `RetrievedChunk`.
  - *Metric:* — (load-bearing for attributable ablations) · *Depends on:* T1

#### Phase 1 — Ingest

- [x] **T3 · Acquire & vendor 2–3 static 10-Ks.** Download once, store raw under
  the gitignored data dir; record source + corpus hash.
  - *Acceptance:* 2–3 filings on disk across ≥2 companies; a manifest notes accession/URL + corpus hash; raw files gitignored.
  - *Metric:* — · *Depends on:* T1
- [x] **T4 · Section segmentation (Item 1 / 1A / 7 / 8).** Heading/anchor
  detection that splits a filing into Items before any size-based chunking.
  - *Acceptance:* each chosen filing splits into the target Items with correct `section` labels; validated by eyeball across all filings (boundary robustness is a known risk).
  - *Metric:* — · *Depends on:* T3
- [x] **T5 · Structure-aware chunker + metadata. ★** Size-bounded chunking within
  sections that keeps tables coherent and footnotes with referents; attach full
  metadata (`doc_id, company, fiscal_year, section, anchor, chunk_id`).
  - *Acceptance:* chunks carry complete metadata; ≥1 documented example of a table/footnote a naive splitter would break, preserved intact (this is the §4 interview exhibit); no mid-sentence/mid-row splits in a spot-check.
  - *Metric:* — (sets the recall ceiling measured at T10) · *Depends on:* T2, T4
- [x] **T6 · Chunk store + inspection script.** Persist chunks keyed by
  `chunk_id`; a small CLI to dump/inspect chunks for a doc/section.
  - *Acceptance:* `inspect` prints chunks with metadata for a given filing/section; store re-loads without re-parsing. *(Demo: inspect chunks.)*
  - *Metric:* — · *Depends on:* T5

#### Phase 2 — Dense retrieval baseline

- [x] **T7 · Embed chunks + build dense index.** Embedding model + vector index;
  index is embedding-model-versioned and rebuilt on ingest.
  - *Acceptance:* index builds over the full chunk set; build is reproducible; embedding model id recorded for the run.
  - *Metric:* — · *Depends on:* T6
- [x] **T8 · Dense retrieval top-k (Retrieve seam). ★** Config-toggleable
  `Retrieve(query, config) → RetrievedChunk[]` returning dense top-k with scores
  + provenance — the seam the harness and CLI talk to.
  - *Acceptance:* a query returns ranked `RetrievedChunk`s with `stage_provenance="dense"`; `top_k` config-driven; eyeball relevance sane. *(Demo: query it.)*
  - *Metric:* — (enables recall@5) · *Depends on:* T7, T2

#### Phase 3 — First measurable signal (no LLM — cheap, honest, fast)

- [ ] **T9 · Golden set v0, hand-verified. ★** *(partial: 30 candidate rows
  drafted + grounded at `data/golden/candidates.jsonl` with a coverage table and
  verification note; **human verification of every row still pending** — that's
  the integrity step, not yet done.)* ~30 rows (`id, question,
  gold_answer, gold_chunk_ids, type, difficulty`); ~15% `unanswerable`, several
  `multi_hop`; **AI drafts, human verifies every row**, stratified by
  type × section × difficulty.
  - *Acceptance:* ~30 rows committed as version-controlled ground truth; every row carries verified `gold_chunk_ids` (existing `chunk_id`s); coverage table shows the type/section/difficulty spread; a note records the by-hand verification. *(Artifact: the labeled set.)*
  - *Metric:* enables every downstream metric · *Depends on:* T6 (needs real `chunk_id`s); questions can be drafted earlier
- [x] **T10 · Retrieval-metrics harness on dense-only. ★ FIRST NUMBER.**
  *(done: `ragauge eval --no-judge` → `ragauge/eval/metrics.py` + `run.py`;
  measured **recall@5 = 0.32, MRR = 0.17** on the candidate golden set, tied to
  config+corpus hash, no LLM.)*
  recall@5, MRR, unanswerable-precision computed without any LLM; reproducible
  from a config hash.
  - *Acceptance:* one command runs the golden set through dense retrieval and reports recall@5, MRR, unanswerable-precision; output tied to (config hash, corpus hash); this is the **honest dense-only baseline**. *(Artifact: the baseline.)*
  - *Metric:* **recall@5, MRR, unanswerable-precision** · *Depends on:* T8, T9

#### Phase 4 — Generation

- [ ] **T11 · Pipeline seam (Retrieve → Generate).** *(deviation: the harness
  composes retrieve→generate→judge inline in `eval/run.py` rather than via a
  standalone `Pipeline` object — sufficient for the eval surface; a dedicated
  `Pipeline` can be extracted when the CLI `ask` command lands.)* Single
  `Pipeline` object that composes the stages behind one call; harness/CLI talk
  only to it.
  - *Acceptance:* `pipeline(question | config)` runs retrieve→generate end-to-end; internal stages never called directly by harness/CLI.
  - *Metric:* — · *Depends on:* T8
- [x] **T12 · Grounded answer with inline citations. ★** *(done:
  `ragauge/generate/generator.py`; `claude-opus-4-8` via `messages.parse`
  structured output `{abstained, answer, citations}`, model id/pricing verified
  against the live API ref; fabricated citations are filtered to the supplied
  evidence; `Answer` carries token/cost/latency. Needs an API-key run to exercise
  live.)* Generation (latest Claude;
  **verify model id/pricing against the live API ref before wiring**) constrained
  to retrieved evidence, citing `chunk_id`s per claim.
  - *Acceptance:* answers cite `chunk_id`s for supported claims and use only supplied chunks; `Answer` carries token/cost/latency telemetry. *(Demo: ask it.)*
  - *Metric:* — (sets up groundedness) · *Depends on:* T11
- [x] **T13 · Insufficient-evidence path. ★** *(done: pre-generation evidence
  gate (`retrieval.min_score` / empty evidence) abstains before any LLM call, +
  post-generation `abstained=true`; a refusal / unparseable output also abstains
  rather than fabricates. No citations on abstentions.)* Dual-trigger abstention:
  pre-generation evidence gate (weak/low-score retrieval short-circuits before an
  LLM call) + post-generation honesty; returns structured `abstained=true` with no
  fabricated citations.
  - *Acceptance:* unanswerable golden rows return a structured abstention; pre-gen gate avoids the generation call when evidence is too weak; no citations on abstentions.
  - *Metric:* **unanswerable-precision** (now generation-side too) · *Depends on:* T12

#### Phase 5 — Generation metrics & RunReport

- [x] **T14 · LLM-as-judge (Pydantic schema). ★** *(done:
  `ragauge/eval/judge.py`; `messages.parse` → `JudgeVerdict{supported,
  unsupported_claims, score}`, no free-text parsing; rolls up into
  groundedness/supported-rate + unsupported-claim rate; prompt+schema versions
  recorded in the RunReport; judge≥generator capability gate enforced. Needs an
  API-key run for live numbers.)* Structured claim-level
  grounded/unsupported labels that roll up into rates; judge prompt + schema
  versioned with the run.
  - *Acceptance:* judge emits validated structured output (no free-text parsing); claim labels aggregate into groundedness + unsupported-claim rate; prompt/schema version recorded.
  - *Metric:* **groundedness / supported-claim rate, unsupported-claim rate** · *Depends on:* T12
- [x] **T15 · Ops telemetry — cost & latency.** *(done: `ragauge/eval/cost.py`;
  $/run from real provider `usage` token counts (not a generic tokenizer),
  per-stage p50/p95 for retrieval and generation. Pricing table verified against
  the live API ref. Rerank's latency cost surfaces once T19 lands.)* $/run via
  the **provider token-counting API** (not a generic tokenizer); p50/p95 latency
  per-stage and end-to-end.
  - *Acceptance:* a run reports real $/run from provider token counts and p50/p95 latency broken out by stage; rerank's latency cost is attributable.
  - *Metric:* **$ per eval run, p95 latency** · *Depends on:* T12
- [x] **T16 · RunReport assembly + persistence.** *(done: `eval/run.py` writes a
  `RunReport` to `metrics.json` — per-question rows + aggregates + (config hash,
  corpus hash, embedding/generator/judge model ids, timestamp, cost). Append-only
  history file is a small follow-up once the ablation needs it.)* Append-only
  per-config report:
  per-question results + aggregates + run metadata (config hash, corpus hash,
  model id, timestamp, cost).
  - *Acceptance:* each run persists one `RunReport` reproducible from (config hash, corpus hash, model id); history is append-only for regression tracking.
  - *Metric:* — (substrate for ablation + CI) · *Depends on:* T10, T14, T15

#### Phase 6 — Hybrid retrieval & the ablation (the headline)

- [x] **T17 · BM25 sparse index + retrieval, toggleable.** Tokenizer-versioned
  sparse index; `bm25` on/off config toggle; provenance recorded.
  - *Acceptance:* BM25 returns top-k with `stage_provenance="bm25"`; toggles independently of dense; recall@5 reportable for BM25-only. *(Demo: stage toggles.)*
  - *Metric:* recall@5 / MRR (BM25 contribution) · *Depends on:* T8
  - **DONE:** in-memory Okapi BM25 (`ragauge/retrieve/bm25.py`), tokenizer-versioned
    (`TOKENIZER_VERSION`), built lazily from the same chunk store the dense index
    embeds — no separate artifact to go stale. Implemented directly (no
    `rank_bm25`), mirroring the NumPy-over-FAISS call: zero native deps, inspectable.
- [x] **T18 · Reciprocal Rank Fusion, toggleable.** Fuse dense + sparse by rank;
  `fusion` on/off; fusion constant config-driven.
  - *Acceptance:* fused ranking combines both stages by rank (no score-normalization hacks); toggle on/off; provenance shows contributing stages.
  - *Metric:* recall@5 / MRR (fusion lift) · *Depends on:* T17
  - **DONE:** pure `reciprocal_rank_fusion` (`ragauge/retrieve/fusion.py`), `rrf_k`
    config-driven; fused `RetrievedChunk`s carry one `ProvenanceEntry` per
    contributing stage. **Fusion lift: recall@5 0.32 → 0.51 (+0.19), MRR 0.17 → 0.29.**
- [x] **T19 · Cross-encoder rerank, toggleable.** Re-score fused top-k on a short
  list for final top-n; `rerank` on/off; latency cost surfaced.
  - *Acceptance:* rerank reorders the fused shortlist; toggle on/off; per-stage latency delta captured (feeds the trade-off story).
  - *Metric:* recall@5 / MRR (rerank lift) + p95 latency · *Depends on:* T18, T15
  - **DONE:** `CrossEncoderReranker` (`ragauge/retrieve/rerank.py`), lazy-loaded,
    `rerank_model` config-driven, appends a `RERANK` provenance entry. **Finding:
    the cross-encoder rerank did *not* pay for itself here — swept three
    checkpoints (two MS-MARCO sizes + the strong same-family `bge-reranker`); all
    lose to hybrid on recall@5, so it's the stage (not one weak model) that
    doesn't earn its cost:**
    - hybrid (no rerank): **recall@5 0.51**, MRR 0.29, p95 179ms
    - `ms-marco-MiniLM-L-6-v2` (default): recall@5 0.34, MRR 0.28, p95 2111ms
    - `ms-marco-MiniLM-L-12-v2`: recall@5 0.42, MRR 0.39, p95 5815ms
    - `BAAI/bge-reranker-base`: recall@5 0.42, **MRR 0.41**, p95 23607ms
    The stronger models sharpen the *first* hit (MRR up) but demote multi-hop
    second-gold chunks out of the top 5, at 10–130× retrieval latency on CPU.
    **Rerank stays off by default; the toggle remains so a finance-tuned
    cross-encoder can be re-measured later.**
- [~] **T20 · Ablation runner + table. ★ THE THESIS ARTIFACT.** Iterate configs
  (dense → +BM25/RRF → +rerank), collect `RunReport`s, emit the table: recall@5,
  MRR, unanswerable-precision, groundedness, unsupported-rate, $/run, p95 per config.
  Include an **embedding-model dimension** (baseline `bge-base-en-v1.5` vs.
  finance-domain `voyage-finance-2` vs. `text-embedding-3-large`) — a deterministic,
  LLM-free recall@5/MRR comparison per §S1.4.
  - *Acceptance:* one command runs the config matrix and produces the ablation table showing **lift per stage**; if a stage doesn't earn its cost, the writeup says so (willingness to cut is the signal). *(Artifact: the thesis table.)*
  - *Metric:* **recall lift per stage** (+ all of the above, per config) · *Depends on:* T16, T19
  - **PARTIAL:** `python -m ragauge.eval.ablation` runs the 3-rung retrieval
    matrix (dense → hybrid → hybrid+rerank) in one command, reusing one
    embedder/retriever, and emits the markdown comparison table + per-config
    `RunReport`s to `metrics_ablation.json`. Groundedness/$/run columns populate
    when an `ANTHROPIC_API_KEY` is present (deterministic recall/MRR/p95 always).
    *Still pending:* the **embedding-model dimension** (`voyage-finance-2` /
    `text-embedding-3-large`) and the judged groundedness run.

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

## Slice 1 requirements: Ingestion + dense retrieval

> **Scope of this slice.** The first vertical slice: take raw 10-K files on disk
> and end at a config-toggleable dense-retrieval seam returning ranked evidence —
> i.e. checklist tasks **T3–T8** (and the content-addressed `chunk_id` from
> **T2**). It deepens [`DESIGN.md`](./DESIGN.md) §4 (chunking), §5.1 (dense), and
> §9 (where state lives) into build-ready requirements. It deliberately stops
> *before* generation and the eval harness; the first *number* (recall@5, T10)
> sits one task past this slice and is what these requirements exist to make real.
>
> **Why this much rigor on ingest:** the metadata schema and `chunk_id` are
> load-bearing (§2.3). Get chunking or id-stability wrong and every downstream
> metric measures noise. This is the recall *ceiling* — generation and reranking
> can only recover evidence that ingest preserved and retrieval surfaced.

### S1.1 Parsing approach for messy filings

10-Ks are adversarial input (HTML/iXBRL, huge numeric tables, dense footnotes,
inconsistent Item formatting). The parser's job is to recover clean,
structure-tagged text **without stranding numbers from their labels**.

1. **Source document selection.** Parse the **primary 10-K HTML document** (the
   main `.htm`), not the SGML full-submission wrapper. Filings are inline-XBRL
   (iXBRL); the MVP parses the *rendered* text, not XBRL facts. *(Deviation note:
   XBRL fact extraction would give near-perfect number↔label binding for the
   financial statements — recorded as a post-MVP upgrade, explicitly out of scope
   here to keep the slice honest and small.)*
2. **HTML → structured blocks.** Use a permissive HTML parser (e.g. lxml /
   BeautifulSoup). Drop non-content: `<script>`/`<style>`, iXBRL hidden elements,
   page-number and running-header/footer artifacts, and the document's own Table
   of Contents. Preserve block structure: headings, paragraphs, lists, and
   **tables as tables** (do not flatten yet).
3. **Layout-table vs. data-table discrimination.** 10-Ks use `<table>` for both
   visual layout (indentation, columns of prose) and real financial data. A
   numeric-density / row×col heuristic classifies each table; layout tables are
   unwrapped to prose, **data tables are kept structured** (§S1.2.3).
4. **Table linearization that preserves binding.** A retained data table is
   serialized so every cell stays adjacent to its **row label and column header**
   (e.g. Markdown grid or `header | label | value` rows) — the make-or-break for
   finance QA (§DESIGN 14). A short caption header (`company · FY · section ·
   nearest heading`) is prepended so "$X" stays attached to "Total revenue,
   FY2023."
5. **Footnotes.** Two senses, handled distinctly: (a) **Notes to the Financial
   Statements** are large prose and segment as normal Item 8 content; (b) inline
   markers (`(1)`, superscripts) tied to a table row or sentence are **kept with
   their referent** — the marker is not orphaned from its note text (carried in
   the same chunk or linked via `anchor`).
6. **Normalization.** Collapse runaway whitespace; normalize Unicode (smart
   quotes, non-breaking spaces, currency/minus glyphs); repair hyphenated
   line-break splits; strip form-feed/page-break artifacts. Normalization is
   deterministic and applied **before** `chunk_id` hashing so ids are stable.
7. **Robustness posture.** Item boundaries and table markup vary across
   filers/years (a known §DESIGN 14 risk). The parser targets the **2–3 chosen
   filings reliably and is validated by eyeball across all of them**; a universal
   EDGAR parser is a non-goal. Fragility is documented, not hidden.

### S1.2 Structure-aware chunking strategy

Per §DESIGN 4: **structure-first, then size-bounded** — never the reverse.

1. **Section segmentation first.** Detect Item boundaries (**Item 1, 1A, 7, 8** at
   minimum) by heading/anchor patterns *before* any size split. Disambiguate real
   section headings from their Table-of-Contents echoes (ToC entries precede the
   body / are link targets; the body heading is the operative one). Each segment
   is labelled with a `section` enum; text between recognized Items inherits the
   preceding Item. Section identity is a **first-class retrieval filter** (a
   risk-factors question must not be answered from MD&A).
2. **Size-bounded chunking within a section.** Accumulate semantic units
   (paragraphs / sub-headings) into chunks up to a **token budget**, splitting
   only on unit boundaries — **never mid-sentence or mid-row**.
   - **Budget is bounded by the embedding model's max sequence length.** With a
     512-token-max encoder (the recommended default, §S1.4), target **~350–450
     content tokens** per prose chunk, leaving headroom for the caption header and
     any model-required instruction prefix. Overlap **~10–15% (~50–64 tokens)** to
     preserve cross-boundary context.
   - These numbers are a **documented starting default, not a tuned truth** — they
     are validated/retuned against recall@5 at **T10** and may move. Token
     counting uses the **embedding model's own tokenizer** (the binding
     constraint), not a generic one.
3. **Tables are atomic where possible.** A data table (or a coherent table region)
   is a **single chunk** with its caption header. A table that exceeds the model
   max is split **on row groups, repeating the column-header context** in each
   piece. `content_type=table` is recorded.
4. **Footnotes travel with their referent** (§S1.1.5); `content_type=footnote`
   where applicable.
5. **Every chunk carries full metadata and a stable `chunk_id`** (§S1.3).

> **Interview exhibit (T5 acceptance):** keep ≥1 concrete worked example of a
> table or footnote that a naive recursive-character splitter would shatter,
> preserved intact here — this is the §DESIGN 4.3 / probe-#1 artifact.

### S1.3 Per-chunk metadata schema

The atomic unit of evidence (§DESIGN 2.3). Required fields are the §4.2 set;
the rest are inspection/exhibit support.

| Field | Type | Definition | Example |
|---|---|---|---|
| `doc_id` | str | Stable filing id (company + form + FY slug, or accession no.) | `AAPL-10K-FY2023` |
| `company` | str | Canonical issuer (ticker and/or name; CIK optional) | `AAPL` |
| `fiscal_year` | int | Fiscal year the filing covers | `2023` |
| `section` | enum | `ITEM_1 \| ITEM_1A \| ITEM_7 \| ITEM_8 \| OTHER` | `ITEM_1A` |
| `anchor` | str | Human-meaningful locator: section + nearest heading + ordinal/offset, so a citation points somewhere a human can verify in the source | `ITEM_1A · "Risks Related to Supply Chain" · ¶3` |
| `chunk_id` | str | **Stable, content-derived** id (see below) | `AAPL-10K-FY2023:ITEM_1A:9f3a1c7b2e10` |
| `text` | str | Normalized chunk text (incl. caption header for tables) | … |
| `content_type` | enum | `prose \| table \| footnote` | `table` |
| `token_count` | int | Tokens under the embedding tokenizer (budget audit) | `412` |
| `char_span` | tuple? | Offsets into the normalized section (inspection) | `(10342, 11876)` |
| `source_path` | str | Relative path to the source filing (provenance) | `data/raw/AAPL-10K-2023.htm` |

**`chunk_id` derivation (the load-bearing detail).** Deterministic
content-addressing, **no UUIDs/timestamps**:

```
chunk_id = f"{doc_id}:{section}:{sha256(normalized_text).hexdigest()[:12]}"
```

- **Deterministic & reproducible:** identical input + identical chunking config →
  identical id on every rebuild (T2 stability unit test).
- **Unique across scope:** prefixing `doc_id`/`section` prevents collisions when
  boilerplate text recurs across filings.
- **Pinning caveat (state it honestly):** ids are stable *for a given chunking
  config*. Re-tuning chunk size changes boundaries and therefore ids — so the
  **golden set's `gold_chunk_ids` are pinned to `(corpus hash, chunking-config
  hash)`**, and re-tuning chunking after T9 means re-verifying affected rows. This
  is a deliberate, documented coupling, not an accident.

### S1.4 Embedding choice (with justification)

**Decision (firm).** **Local `BAAI/bge-base-en-v1.5`** (768-dim, 512-token max,
asymmetric query/passage prompting, run via `sentence-transformers`) is the
**baseline default** — *and* the embedding model is **promoted to a first-class,
measured ablation dimension** rather than a buried "maybe later." *(Verify exact
model id, dim, and max-seq at wiring time, per `CLAUDE.md`; `intfloat/e5-base-v2`
is an equivalent drop-in.)*

This is the most on-thesis choice the slice can make: the project's identity is
*"every choice is a measured decision,"* so the embedding model is treated exactly
like the retrieval stages — pick a reproducible default, then **prove** it against
alternatives with numbers.

Justification, scored against the project's own thesis (reliability + honest
measurement), **not** leaderboard rank alone:

- **Determinism / reproducibility — the decisive reason.** Embeddings run locally
  and deterministically, so a `RunReport` reproduces from `(config, corpus, model
  id)` forever. A hosted embedding API can silently re-version and make a past
  recall@5 unreproducible — fatal for a harness whose entire credibility is
  determinism everywhere except the LLM calls (§DESIGN 9).
- **Zero marginal ingest cost.** Re-ingestion happens on *every* chunking change;
  a per-token embedding bill would tax exactly the iteration loop this slice
  exists to enable, and would muddy `$ per eval run` (which should attribute to
  the LLM judge/generator, not ingest).
- **Honest, competitive baseline.** MTEB-competitive retrieval at 768-dim gives a
  real dense baseline that BM25 + RRF + rerank then have to *beat* in the
  ablation — the point is the lift, not a flattering starting point.
- **Asymmetric retrieval fit.** bge/e5 support distinct query vs. passage
  prompting, matching question→passage retrieval over 10-K prose.
- **Operational footprint.** 768-dim over a few thousand chunks is trivial; CPU
  embedding keeps the project laptop- and CI-runnable with no GPU dependency.

**The embedding model is a measured ablation row — and a cheap, high-signal one.**
It is **config-selected and index-versioned** (§S1.5, §DESIGN 9), so swapping it
is a config diff. Crucially, **comparing embedding models needs *zero* LLM
calls** — embedding quality is scored by **recall@5 / MRR on the golden set**, the
deterministic metrics that need no judge. So the comparison is nearly free yet
yields a real result. The headline comparison:

| Candidate | Hypothesis it tests | Cost to compare |
|---|---|---|
| `bge-base-en-v1.5` *(default/baseline)* | Reproducible general-purpose anchor | local, free |
| **`voyage-finance-2`** | *Does a **finance-domain** embedding beat general-purpose on 10-K retrieval?* | API, embed-only |
| OpenAI `text-embedding-3-large` | Does a larger hosted general model close the gap? | API, embed-only |

This rides the existing ablation/sweep machinery (T20/T21) as one extra
deterministic row — no new infrastructure. It produces a genuine interview
exhibit: *"finance-domain vs. general embeddings on 10-Ks, here's the recall@5
delta — and here's why the reproducible local model still anchors the baseline."*
The local default stands as the harness's deterministic backbone; the domain model
has to **earn** the swap with numbers. *(Voyage is Anthropic's recommended
embeddings provider; verify ids/dims at wiring time.)*

### S1.5 Index layout

Corpus is tiny by design (2–3 filings → ~hundreds–low-thousands of chunks), which
*simplifies* the right choice rather than complicating it.

- **Vector index — exact flat search.** Use a local file-based **exact** index
  (e.g. FAISS `IndexFlatIP` over L2-normalized vectors = cosine). At this scale,
  ANN approximation buys nothing and would **inject approximation error into the
  very recall@5 we are trying to measure** — exact search keeps the retrieval
  metric clean and deterministic. (ANN is a documented scaling concern, §DESIGN
  14, not an MVP need.)
- **Chunk store is the source of truth.** Chunks persist keyed by `chunk_id` in an
  **inspectable** store (**JSONL** preferred for the T6 dump/inspect CLI; SQLite
  acceptable). It owns text + metadata; the vector index owns only geometry.
- **Position→id sidecar.** A mapping from vector-index row → `chunk_id` joins the
  two. No text lives in the vector index.
- **Versioned, self-describing artifacts.** The built-index directory is stamped
  with **`embedding_model_id` + `corpus_hash` + `chunking_config_hash`** so a
  stale or mismatched index **cannot be silently served** (§DESIGN 9). Rebuild is
  deterministic: same inputs → byte-comparable artifacts.
- **All build state is gitignored** (raw filings + indexes, per `CLAUDE.md`); a
  manifest records source URLs/accessions + corpus hash (T3).

### S1.6 Functional requirements

- **FR1 — Acquire & vendor corpus.** Download 2–3 static 10-Ks across ≥2
  companies once; store raw under the gitignored data dir; manifest records
  accession/URL + `corpus_hash`. *(T3)*
- **FR2 — Parse.** Convert each primary HTML filing into normalized,
  structure-tagged blocks with layout/data-table discrimination and footnote
  association (§S1.1).
- **FR3 — Segment sections.** Split each filing into Item 1 / 1A / 7 / 8 (+
  `OTHER`) with correct `section` labels, ToC echoes disambiguated. *(T4)*
- **FR4 — Chunk.** Produce structure-aware, size-bounded chunks: tables coherent,
  footnotes with referents, no mid-sentence/mid-row splits, token budget under the
  embedding max. *(T5)*
- **FR5 — Metadata + `chunk_id`.** Every chunk carries the full §S1.3 schema; ids
  are deterministic and content-derived. *(T2, T5)*
- **FR6 — Chunk store + inspection.** Persist by `chunk_id`; a CLI dumps chunks
  with metadata for a given filing/section without re-parsing. *(T6)*
- **FR7 — Embed + build index.** Embed all chunks and build the exact vector
  index; record `embedding_model_id`; rebuild is reproducible. *(T7)*
- **FR8 — Dense retrieval seam.** `Retrieve(query, config) → RetrievedChunk[]`
  returns dense top-k with `score` and `stage_provenance="dense"`; `top_k` is
  config-driven; this is the *only* surface the harness/CLI call. *(T8)*
- **FR9 — Config & versioning.** Embedding model, `top_k`, and chunking params are
  declarative config; index artifacts are stamped with model/corpus/chunking
  hashes (§S1.5).

### S1.7 Non-functional requirements

- **NFR1 — Determinism/reproducibility.** Given `(corpus_hash, embedding_model_id,
  chunking_config_hash)`, ingest and retrieval are reproducible; no UUIDs,
  timestamps, or hidden state influence `chunk_id` or ranking.
- **NFR2 — No network at serve time.** Dense retrieval runs fully offline against
  local artifacts (no live EDGAR fetch, no hosted embedding call in the default).
- **NFR3 — Portability.** CPU-only; runs on a laptop and in CI with no GPU.
- **NFR4 — Performance (targets, validated not assumed).** Full ingest
  (parse→chunk→embed→index) of the corpus completes in **minutes** on CPU; a dense
  query (encode + flat search) returns top-k in **well under ~250 ms p95** locally.
- **NFR5 — Cost.** **Zero marginal embedding cost** in the default path, keeping
  `$ per eval run` attributable to LLM calls.
- **NFR6 — Security/hygiene.** No secrets committed (`.env`, gitignored); raw
  filings and built indexes gitignored; the repo stays lean.
- **NFR7 — Inspectability.** Chunks and their metadata are human-readable and
  dumpable (supports the §DESIGN 4.3 exhibit and debugging).
- **NFR8 — Validated robustness.** Section segmentation and table handling are
  eyeball-validated across **all** chosen filings before the slice is called done.

### S1.8 Acceptance criteria (slice exit)

The slice is done when:

1. **Corpus.** 2–3 filings across ≥2 companies are vendored with a manifest +
   `corpus_hash`; raw files are gitignored. *(T3)*
2. **Sections.** Each filing segments into Item 1 / 1A / 7 / 8 with correct labels,
   verified by eyeball across every filing; ToC echoes are not mistaken for
   section starts. *(T4)*
3. **Chunking exhibit.** Chunks carry the complete §S1.3 metadata; a **documented
   worked example** shows a table/footnote a naive splitter would break, preserved
   intact; a spot-check finds **no mid-sentence/mid-row splits**. *(T5)*
4. **Id stability.** A unit test proves `chunk_id` is deterministic from content
   and identical across a re-ingest under the same chunking config. *(T2)*
5. **Store + inspect.** The inspection CLI prints chunks with metadata for a given
   filing/section, reloading from the store **without re-parsing**. *(T6)*
6. **Index.** The exact vector index builds over the full chunk set, is stamped
   with `embedding_model_id` + `corpus_hash` + `chunking_config_hash`, and rebuilds
   reproducibly. *(T7)*
7. **Retrieval seam.** A query returns ranked `RetrievedChunk`s with scores and
   `stage_provenance="dense"`; `top_k` is config-driven; an eyeball check on a
   handful of questions shows sane relevance. *(T8)*
8. **Reproducibility check.** Two clean ingests of the same corpus + config yield
   identical `chunk_id`s and identical top-k for a fixed query set.

> **Out of scope for this slice (guards against scope creep):** BM25/RRF/rerank
> (T17–T19), any generation or citations (T11–T13), the LLM judge and `RunReport`
> (T14–T16), and the golden set/recall@5 number itself (T9–T10). This slice
> *earns* that first number; it does not produce it.

---

## Implementation status

> **As of 2026-06-29.** Update this at the end of every working session
> (per [`CLAUDE.md`](./CLAUDE.md)). The checklist above is the per-task tracker;
> this section is the prose snapshot a reviewer reads first.

**Phase:** **Slice 1 (ingest + dense retrieval, T1–T8), the eval harness
(T10, T12–T16), and the hybrid retrieval ablation (T17–T20) are built and
tested.** The full path exists end-to-end: raw 10-Ks → structure-aware chunks →
stamped exact dense index → config-toggleable `Retrieve` seam (dense + BM25 →
RRF → cross-encoder rerank, each its own toggle) → grounded, cited generation
(or abstention) → structured LLM-as-judge → a persisted `RunReport`, with a
one-command ablation runner that sweeps the retrieval matrix into a comparison
table. The **deterministic retrieval ablation is measured** (dense recall@5
0.32 → hybrid 0.51 → +rerank 0.34); the judged generation metrics are
implemented and stub-validated but need an `ANTHROPIC_API_KEY` run (no key in the
build environment).
**Progress:** 17 / 23 subtasks coded (T1–T8 + T10, T12–T19; T9, T11, T20
partial). **Next up: human-verify the golden set, add the embedding-model
dimension to the ablation (T20), then the model/cost sweep + CI gate (T21–T23).**

**What runs today (`ragauge` CLI):** `acquire` · `ingest` · `inspect` ·
`build-index` · `query` (Slice 1) · **`eval`** — `ragauge eval --no-judge` runs
the golden set through retrieval and reports recall@5 / MRR /
unanswerable-precision with **no LLM and no API key**, writing a `RunReport` to
`metrics.json`; `ragauge eval` (with a key) adds grounded generation + the judge.
The retrieval stack is config-toggleable end-to-end —
`python -m ragauge.eval.run --retrieval {dense|hybrid|hybrid+rerank}` picks a
rung, and **`python -m ragauge.eval.ablation`** sweeps all three in one command,
emitting the markdown comparison table + per-config `RunReport`s to
`metrics_ablation.json`. **34 unit tests green** (`pytest`), all offline.

**Measured (deterministic, no LLM), on the 30-row candidate golden set**
(`corpus_hash=60707081f218`):

| config | recall@5 | MRR | retrieval p95 ms |
|---|---|---|---|
| dense-only (bge-base) | 0.32 | 0.17 | 197 |
| **+ BM25 + RRF (hybrid)** | **0.51** | **0.29** | 179 |
| + cross-encoder rerank (off by default) | 0.34 | 0.28 | 2111 |

recall@5 = 0.32 is the **honest dense-only baseline** — bge underperforms on the
numeric/table questions that dominate a 10-K golden set. The hybrid ablation
earns that headroom back, **measured**: a BM25 sparse stage fused with dense via
RRF lifts recall@5 from **0.32 → 0.51 (+0.19) at no latency cost**, while a
generic MS-MARCO cross-encoder reranker does **not** pay for itself here (recall@5
falls to 0.34 at ~10× the retrieval latency), so it ships **off by default**. The
reranker is itself a measured call — three cross-encoders were swept and all three
lose to hybrid on recall@5, confirming it's the *stage*, not one weak checkpoint
(see T19).

### Headline-metrics status

Mirrored at the top of [`README.md`](./README.md). Retrieval metrics are now
**measured**; the judged metrics are wired and validated on stubbed calls but
remain **unmeasured** until a run with an API key — *honest numbers or none.*

| Metric | Status |
|---|---|
| **recall@5** (best config = hybrid) | ✅ **0.51** measured (T10/T18, no LLM); dense baseline 0.32 |
| **MRR** (best config = hybrid) | ✅ **0.29** measured (no LLM); dense baseline 0.17 |
| **recall lift per stage** (ablation) | ✅ measured (T20): dense 0.32 → hybrid 0.51 (**+0.19**) → +rerank 0.34 |
| **p95 latency (retrieval)** | ✅ hybrid **~179 ms**, dense **~197 ms**; rerank ~2100 ms (its cost made explicit) |
| **unanswerable-precision** | wired; n/a in retrieval-only mode (needs the generator's abstention signal — T13) |
| **groundedness / supported-claim rate** | wired (T14); needs a judged run |
| **unsupported-claim rate** | wired (T14); needs a judged run |
| **$ / eval run** | wired (T15, real provider token counts); needs a judged run |

### Completed features

**Slice 1 — ingest + dense retrieval (T1–T8), coded · tested · verified:**
- **Scaffolding & contracts (T1–T2).** `ragauge` package with the four-component
  layout (`ingest/ retrieve/ generate/ eval/`); Pydantic `Chunk / RetrievedChunk
  / Answer / GoldRow / RunReport`; content-addressed `chunk_id`
  (`{doc_id}:{section}:sha256(text)[:12]`); a declarative `PipelineConfig` whose
  retrieval stages are config toggles (only `dense` wired). `uv.lock` pins torch
  to the CPU wheel index.
- **Acquire (T3).** EDGAR fetch of the primary 10-K HTML for AAPL / MSFT / NVDA;
  manifest with accession/URL + `corpus_hash`; raw files gitignored.
- **Parse + segment + chunk (T4–T5).** HTML → blocks with layout/data-table
  discrimination and binding-preserving table linearization; Item 1/1A/7/8
  segmentation (ToC echoes, running headers, and by-reference Item 8 handled);
  structure-aware size-bounded chunking (tables atomic, no mid-sentence/mid-row
  splits) with full metadata.
- **Store + inspect (T6).** JSONL chunk store keyed by `chunk_id`; `inspect` CLI
  dumps chunks by doc/section without re-parsing.
- **Embed + index (T7).** Local `bge-base-en-v1.5` (768-dim, asymmetric
  query/passage prompting); exact flat inner-product index stamped with
  model+corpus+chunking hashes so a stale index can't be served.
- **Dense Retrieve seam (T8).** Config-toggleable `Retrieve(query, config) →
  RetrievedChunk[]` with `stage_provenance="dense"`; the only surface the harness
  / CLI call.

**Eval harness — retrieval baseline + generation + judge (T10, T12–T16), coded ·
tested:**
- **Retrieval metrics, no LLM (T10).** `ragauge/eval/metrics.py` —
  recall@5, MRR, unanswerable-precision, p50/p95 latency as pure functions of the
  ranking; reproducible from `(config_hash, corpus_hash)`. **The first number:
  recall@5 = 0.32, MRR = 0.17.**
- **Grounded generation + citations (T12).** `ragauge/generate/generator.py` —
  `claude-opus-4-8` via `messages.parse` **structured output**
  `{abstained, answer, citations}`; answers cite `chunk_id`s and use only the
  supplied evidence; citations the model invents are filtered to the retrieved
  set; `Answer` carries token / cost / latency telemetry.
- **Dual-trigger abstention (T13).** A **pre-generation evidence gate**
  (`retrieval.min_score` / empty evidence) abstains *before* any LLM call, plus
  **post-generation** `abstained=true`; a refusal / unparseable output also
  abstains rather than fabricating. No citations on abstentions.
- **LLM-as-judge, structured (T14).** `ragauge/eval/judge.py` — `messages.parse`
  → Pydantic `JudgeVerdict{supported, unsupported_claims, score}` (no free-text
  parsing); claim labels roll up into groundedness / supported-rate and
  unsupported-claim rate; prompt + schema versions recorded in the `RunReport`.
- **Ops telemetry (T15).** `ragauge/eval/cost.py` — **$/run from real provider
  `usage` token counts** (not a generic tokenizer); per-stage p50/p95 latency;
  pricing table verified against the live Claude API reference. A
  **judge ≥ generator capability gate** is enforced before any judged run.
- **RunReport assembly (T16).** `ragauge/eval/run.py` writes a `RunReport` to
  `metrics.json`: per-question rows + aggregates + `(config_hash, corpus_hash,
  embedding / generator / judge model ids, timestamp, cost)`.
- **Tests.** 34 unit tests green (19 Slice 1 + 7 covering the deterministic
  metrics, cost-from-token-counts, and the judge-capability gate + 8 covering
  BM25 ranking, RRF fusion, the rerank toggle, and per-stage provenance across
  the hybrid stack); the judged path is validated end-to-end against a stubbed
  client (citation filtering, the abstention gates, None/refusal handling,
  telemetry, and metric aggregation).

**Hybrid retrieval + the ablation (T17–T20), coded · tested:**
- **BM25 sparse retrieval, toggleable (T17).** `ragauge/retrieve/bm25.py` — an
  in-memory Okapi BM25 index, tokenizer-versioned (`TOKENIZER_VERSION`), built
  lazily from the same chunk store the dense index embeds (no separate artifact
  to go stale). Implemented directly (no `rank_bm25`), mirroring the
  NumPy-over-FAISS call: zero native deps, inspectable. Returns top-k with
  `stage_provenance="bm25"`, toggling independently of dense.
- **Reciprocal Rank Fusion, toggleable (T18).** `ragauge/retrieve/fusion.py` —
  pure rank-based fusion (no score-normalization hacks), `rrf_k` config-driven;
  fused `RetrievedChunk`s carry one provenance entry per contributing stage.
  **Fusion lift: recall@5 0.32 → 0.51 (+0.19), MRR 0.17 → 0.29.**
- **Cross-encoder rerank, toggleable (T19).** `ragauge/retrieve/rerank.py` — a
  lazy-loaded `CrossEncoderReranker`, `rerank_model` config-driven, appends a
  `RERANK` provenance entry. **Finding: it doesn't earn its cost here** — three
  checkpoints swept (two MS-MARCO sizes + `bge-reranker-base`), all lose to
  hybrid on recall@5 at 10–130× the retrieval latency on CPU, so rerank ships
  **off by default** with the toggle kept for a future finance-tuned model.
- **Ablation runner + table (T20). ★** `python -m ragauge.eval.ablation` runs
  the 3-rung retrieval matrix (dense → hybrid → hybrid+rerank) in one command,
  reusing a single embedder/retriever, and emits the markdown comparison table +
  per-config `RunReport`s to `metrics_ablation.json`. Groundedness / $-per-run
  columns populate when an `ANTHROPIC_API_KEY` is present; recall / MRR / p95 are
  always deterministic.

**Planning (pre-existing):** design & architecture ([`DESIGN.md`](./DESIGN.md)),
epic + PRD + the 23-task checklist, and the build-ready **§ Slice 1
requirements** spec.

### Partial / in progress
- **T9 — golden set (drafted, not yet verified).** 30 candidate rows are drafted
  and **grounded in real chunk text** at `data/golden/candidates.jsonl`, with a
  coverage table and verification note (`data/golden/README.md`): 20 single_doc /
  5 multi_hop / 5 unanswerable (17%), all five Item sections covered, every
  `gold_chunk_id` confirmed to exist in the corpus. **The by-hand verification of
  every row — the integrity step the whole project hinges on — is still pending.**
- **T11 — pipeline seam (deviation).** The harness composes retrieve → generate →
  judge inline in `eval/run.py` rather than via a standalone `Pipeline` object;
  sufficient for the eval surface today, extractable when a CLI `ask` command
  needs it.
- **T20 — ablation (retrieval dimension done; embedding dimension + judged run
  pending).** The 3-rung retrieval ablation (dense → hybrid → hybrid+rerank) runs
  in one command and is measured. **Still pending:** the **embedding-model
  dimension** (`voyage-finance-2` / `text-embedding-3-large` vs. the bge baseline
  — a deterministic recall@5/MRR comparison per §S1.4) and the **judged
  groundedness columns** (need an `ANTHROPIC_API_KEY`).
- _Note:_ the owner also edits these docs from a separate playbook chat, so treat
  filesystem state as truth and re-verify before relying on any summary.

### Pending
- **Judged metrics run.** Generation / groundedness / unsupported-rate / $-per-run
  are implemented but **unmeasured** until `ragauge eval` (or the ablation) runs
  with an `ANTHROPIC_API_KEY` (no key in this environment).
- **T20 — embedding-model dimension.** The retrieval-stage ablation is done and
  measured; the remaining piece is the embedding-model row (`voyage-finance-2` /
  `text-embedding-3-large` vs. bge) — an LLM-free recall@5/MRR comparison added to
  the same table.
- **T21–T23 — model & cost sweep, CI gate, README/portfolio writeup.**

### Next steps (immediate)
1. **Human-verify the golden set (finish T9):** read and correct every candidate
   row, then promote it to the verified ground-truth file the harness loads — this
   gates the integrity of every number above.
2. **Run `ragauge eval` / the ablation with a key** to fill in the judged headline
   metrics (groundedness, unsupported-claim rate, $/run) and validate the live
   generation + judge path.
3. **Add the embedding-model dimension to the ablation (finish T20):**
   `voyage-finance-2` / `text-embedding-3-large` vs. bge — a deterministic
   recall@5/MRR row, the last piece of the thesis table.
4. **T21–T23 — model/cost sweep, CI gate, README/portfolio writeup.**

### Technical decisions & deviations from plan
- **Eval harness (this session):**
  - **Structured output via `messages.parse` + Pydantic** for both the generator
    (`GenerationOutput{abstained, answer, citations}`) and the judge
    (`JudgeVerdict{supported, unsupported_claims, score}`) — verdicts are parsed
    and aggregated, never regexed out of prose (DESIGN §7.3). `temperature` /
    `budget_tokens` are omitted (removed on Opus 4.8).
  - **Models verified against the live API reference at wiring time:** generator
    and judge both default to **`claude-opus-4-8`** (judge ≥ generator holds since
    they're equal; a capability-rank gate enforces this for any override). Pricing
    `$5 / $25` per 1M (input / output) is taken from the same reference.
  - **Cost from real provider token counts.** `$/run` is computed from
    `response.usage.input_tokens / output_tokens` (the provider's own counts), not
    a generic tokenizer — the locked decision below, now realized in
    `eval/cost.py`.
  - **Deterministic metrics are LLM-free and run without the SDK or a key.** The
    `anthropic` import is **lazy**, so recall@5 / MRR / unanswerable-precision (and
    CI's cheap gate) never depend on an LLM call (DESIGN §7.3: the harness is not
    hostage to judge variance).
  - **Fail-safe on refusal / unparseable output:** the generator abstains and the
    judge scores 0 / unsupported, rather than silently fabricating or counting an
    answer grounded.
  - **`metrics.json` is gitignored**; the explicit baseline (`runs/`, `baseline.json`)
    is committed separately when the ablation lands.
- **Hybrid retrieval & ablation (this session):**
  - **BM25 implemented directly, not via `rank_bm25`.** An in-memory Okapi BM25
    mirrors the NumPy-over-FAISS decision: identical semantics, zero native deps,
    fully inspectable, built lazily from the existing chunk store so there is no
    second artifact to version-drift.
  - **RRF, not score normalization.** Dense and sparse are fused by *rank*
    (reciprocal-rank, `rrf_k` config-driven), sidestepping the brittle
    cross-scale normalization that comparing cosine against BM25 magnitudes would
    otherwise require.
  - **Rerank shipped off by default — a measured cut, not an omission.** The
    cross-encoder stage lost to hybrid on recall@5 across three swept checkpoints
    (two MS-MARCO sizes + `bge-reranker-base`) at 10–130× the CPU retrieval
    latency; the stronger models sharpen MRR (the first hit) but demote multi-hop
    second-gold chunks out of the top 5. The toggle is retained so a finance-tuned
    cross-encoder can be re-measured later. **Willingness to cut a stage that
    doesn't pay for itself is the thesis signal, not a gap.**
  - **One-command ablation.** `eval/ablation.py` reuses a single
    embedder/retriever across rungs and writes per-config `RunReport`s + a
    markdown table to `metrics_ablation.json` (gitignored).
- **Deviations made in Slice 1 (all documented, none contradict `DESIGN.md`):**
  - **Vector index is NumPy, not FAISS.** The design said exact flat search
    *"e.g. FAISS `IndexFlatIP`"*; we implement the identical semantics
    (brute-force inner product over L2-normalized vectors = cosine) in NumPy.
    Same exactness, no native dependency, fully CPU-portable (NFR3). FAISS remains
    a drop-in if scale ever demands it. §S1.5.
  - **Section segmentation hardening for real filings (NFR8).** Two real failure
    modes found and fixed by eyeball across the 3 filings: (a) **MSFT** repeats a
    bare "Item 1A" as a *running page header* on every page — a section start now
    requires a **title** after the item number, so running headers aren't
    boundaries; (b) **NVDA** files a one-line **Item 8 "by reference" stub** with
    the consolidated statements physically in the Item 15 schedules region — a
    tightly-gated post-pass (fires only when Item 8 is a stub) relabels those
    F-pages as `ITEM_8`, so finance evidence carries the right section. These are
    exactly the §DESIGN 14 robustness risks, surfaced not hidden.
  - **Embedding latency, not cost.** Building the 789-chunk dense index took
    ~6 min wall (~20 CPU-min) on a loaded laptop CPU — above NFR4's "minutes" but
    a one-time, offline, zero-$ build (NFR5 holds). Mitigations if it bites: cache
    embeddings across re-ingests, or batch on a GPU. Serving latency is fine
    (96–186 ms/query).
  - **CPU-only torch actually pinned (NFR3 fix).** `[tool.uv.sources]` pinned
    torch to the CPU wheel index, but torch was a *transitive-only* dependency
    (via `sentence-transformers`), so the pin never bound and `uv.lock` resolved
    the PyPI build with ~1.5 GB of CUDA/nvidia wheels — silently violating the
    CPU-only portability claim. Fix: declare `torch` as a **direct** dependency so
    the source pin binds; re-locking drops all 18 nvidia/cuda packages + triton
    (82 → 64) and resolves `torch 2.12.1+cpu`. The documented install path is
    `uv sync` (not `uv pip install -e .`, which ignores `[tool.uv.sources]`).
- **Decisions confirmed at wiring time (per `CLAUDE.md`):** `bge-base-en-v1.5`
  verified as 768-dim / 512-token-max; bge asymmetric query instruction wired;
  normalized embeddings → cosine via inner product. Generator/judge model ids and
  pricing verified against the live Claude API reference (above).
- **Decisions now made (recorded per `CLAUDE.md`):**
  - **Embedding model:** local **`bge-base-en-v1.5`** (768-dim, 512-token max) via
    `sentence-transformers` as the reproducible, zero-cost baseline — *and* the
    embedding model is **promoted to a first-class, LLM-free ablation dimension**
    (vs. finance-domain `voyage-finance-2` and `text-embedding-3-large`), scored by
    recall@5/MRR. See §S1.4.
  - **Vector index:** local **exact flat search** over normalized vectors — at this
    corpus size, exact search keeps recall@5 free of ANN approximation error. ANN
    deferred as a scaling concern (§DESIGN 14). §S1.5.
  - **Chunk store:** inspectable **JSONL keyed by `chunk_id`**; vector index holds
    geometry only. §S1.5.
  - **`chunk_id`:** deterministic content-addressing
    `{doc_id}:{section}:{sha256(text)[:12]}` — no UUIDs/timestamps. §S1.3.
  - **Generator + judge model:** **`claude-opus-4-8`** for both (a measured pick to
    revisit in the T21 sweep), with a judge ≥ generator capability gate.
- **Decisions still open (resolve at implementation, not before):** BM25 library,
  cross-encoder rerank model, and the final generator-vs-judge model split — the
  latter a *measured* outcome of the T21 sweep, not a guess.
- **Cost accounting** uses the **provider token counts** returned on every
  response, not a generic tokenizer (locked decision, now realized; affects
  T15/T21).

---
