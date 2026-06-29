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
**Progress at a glance:** 8 / 23 done — **Slice 1 (ingest + dense retrieval, T1–T8)
complete and verified end-to-end** on AAPL/MSFT/NVDA FY2025–26 10-Ks (789 chunks,
dense top-k working, query p95 < 200 ms). Next up: **T9** (golden set) → **T10**
(the first recall@5 number). See § Implementation status for deviations.

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
  Include an **embedding-model dimension** (baseline `bge-base-en-v1.5` vs.
  finance-domain `voyage-finance-2` vs. `text-embedding-3-large`) — a deterministic,
  LLM-free recall@5/MRR comparison per §S1.4.
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

**Phase:** **Slice 1 (ingest + dense retrieval, T1–T8) is built, tested, and
verified end-to-end.** Raw 10-Ks → structure-aware chunks → stamped exact dense
index → a config-toggleable `Retrieve(query, config)` seam returning ranked,
section-labelled evidence.
**Progress:** 8 / 23 subtasks coded. **Next up: T9 (golden set) → T10 (first
recall@5 number).** The README headline metrics remain intentionally blank: the
first *number* is T10, which sits one task past this slice. No generation, judge,
or `RunReport` code exists yet — by design.

**What runs today (`ragauge` CLI):** `acquire` (EDGAR → manifest + corpus_hash) ·
`ingest` (parse → segment → chunk → JSONL store) · `inspect` (dump chunks by
doc/section, no re-parse) · `build-index` (embed + exact flat index) · `query`
(dense top-k). **19 unit tests green** (`pytest`), all offline (a hash-based test
embedder needs no model download).

**Verified on the real corpus** (AAPL FY2025, MSFT FY2025, NVDA FY2026 — 3
filings / 3 companies, `corpus_hash=60707081f218`): **789 chunks** (667 prose /
122 table), healthy four-Item section coverage on every filing, **byte-identical
`chunk_id`s + chunk order across a re-ingest** (acceptance #8), and sane dense
relevance — e.g. a supply-chain-risk query returns an all-`ITEM_1A` top-5 at
0.73–0.76 cosine. Dense query latency **96–186 ms** (NFR4 target < 250 ms p95).

### Headline-metrics status

The headline numbers (mirrored at the top of [`README.md`](./README.md)) are
**intentionally unmeasured** until the eval harness exists — *honest numbers or
none*. Where each one lands:

| Metric | Status | Lands at |
|---|---|---|
| **recall@5** | not yet measured | **T10** (next) |
| **MRR / unanswerable-precision** | not yet measured | **T10** |
| **groundedness / supported-claim rate** | not yet measured | **T14** (LLM judge) |
| **unsupported-claim rate** | not yet measured | **T14** (LLM judge) |
| **$ / eval run** | not yet measured | **T15** (provider token-counting) |
| **p95 latency (dense retrieval)** | ✅ **~96–186 ms** measured | Slice 1; end-to-end at T15 |
| **recall lift per stage** (ablation) | not yet measured | **T20** |

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
  / CLI call. `ragauge` CLI: `acquire / ingest / inspect / build-index / query`.
- **Tests.** 19 unit tests green, all offline (hash-based test embedder).

**Planning (pre-existing):** design & architecture ([`DESIGN.md`](./DESIGN.md)),
epic + PRD + the 23-task checklist, and the build-ready **§ Slice 1
requirements** spec.

### Partial / in progress
- _Nothing partial in Slice 1._ T1–T8 are complete; T9+ not started.
- _Note:_ the owner also edits these docs from a separate playbook chat, so treat
  filesystem state as truth and re-verify before relying on any summary.

### Pending
- **T9–T23.** No golden set, generation, judge, ablation, sweep, or CI gate yet.
  Critical path to first signal: **T9 (golden set) → T10 (recall@5 baseline)**;
  the thesis artifact (ablation table) is **T20**.

### Next steps (immediate)
1. **T9 — golden set v0:** ~30 hand-verified rows pinned to the real `chunk_id`s
   now in `data/chunks.jsonl` (gold rows reference live ids; stratify by
   type × section × difficulty; ~15% unanswerable).
2. **T10 — retrieval-metrics harness:** run the golden set through the dense seam,
   report recall@5 / MRR / unanswerable-precision tied to (config hash, corpus
   hash) — **the first honest number.**
3. **(stretch) tune chunking** against T10 recall@5 before locking ids for T9.

### Technical decisions & deviations from plan
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
- **Decisions confirmed at wiring time (per `CLAUDE.md`):** `bge-base-en-v1.5`
  verified as 768-dim / 512-token-max; bge asymmetric query instruction wired;
  normalized embeddings → cosine via inner product. The other refinements below
  stand unchanged.
- **Decisions now made (Slice 1, recorded per `CLAUDE.md`):**
  - **Embedding model:** local **`bge-base-en-v1.5`** (768-dim, 512-token max) via
    `sentence-transformers` as the reproducible, zero-cost baseline — *and* the
    embedding model is **promoted to a first-class, LLM-free ablation dimension**
    (vs. finance-domain `voyage-finance-2` and `text-embedding-3-large`), scored by
    recall@5/MRR. See §S1.4. Exact ids/dims verified at wiring time.
  - **Vector index:** local **exact flat search** (e.g. FAISS `IndexFlatIP` over
    normalized vectors) — at this corpus size, exact search keeps recall@5 free of
    ANN approximation error. ANN deferred as a scaling concern (§DESIGN 14). §S1.5.
  - **Chunk store:** inspectable **JSONL keyed by `chunk_id`** (SQLite acceptable);
    vector index holds geometry only, store holds text + metadata. §S1.5.
  - **`chunk_id`:** deterministic content-addressing
    `{doc_id}:{section}:{sha256(text)[:12]}` — no UUIDs/timestamps. §S1.3.
  - **Chunking defaults (tunable, validated at T10):** ~350–450 content tokens,
    ~10–15% overlap, bounded by the embedding model's 512-token max. §S1.2.
- **Decisions still open (resolve at implementation, not before):** BM25 library,
  cross-encoder rerank model, exact Claude model ids for generator vs. judge. Per
  `DESIGN.md` §11 and `CLAUDE.md`, Claude model ids/pricing are **verified against
  the live API reference at wiring time** and the generator/judge picks are a
  *measured* outcome of the T21 sweep, not a guess.
- **Cost accounting** will use the **provider token-counting API**, not a generic
  tokenizer (locked decision; affects T15/T21).

---
