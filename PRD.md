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

### MVP-sized subtasks

Built incrementally; each milestone is independently demoable. Dense-only-first
guarantees an honest baseline before complexity is added.

1. **Ingest & structure-aware chunking** — parse 2–3 static 10-Ks into
   section-aware chunks with the metadata schema (`doc_id, company, fiscal_year,
   section, anchor, chunk_id`); preserve tables and footnotes.
2. **Dense retrieval baseline** — embed chunks, build the vector index, return
   top-k for a query.
3. **End-to-end grounded answer** — generate an answer from dense-only retrieval
   with inline `chunk_id` citations.
4. **Insufficient-evidence path** — pre-generation evidence gate + post-generation
   abstention, returning a structured "insufficient evidence" answer.
5. **Golden set (~30 rows, hand-verified)** — `id, question, gold_answer,
   gold_chunk_ids, type, difficulty`; ~15% unanswerable, several multi-hop; AI
   drafts, human verifies every row.
6. **Baseline metrics harness** — recall@5, MRR, unanswerable-precision, and
   groundedness (LLM-as-judge with Pydantic schema) on dense-only.
7. **Hybrid retrieval, toggleable** — add BM25, reciprocal-rank fusion, and a
   cross-encoder reranker, each behind a config toggle.
8. **Ablation table** — run configs (dense → +BM25/RRF → +rerank) and report
   recall lift, groundedness, and cost/latency per stage.
9. **Model & cost sweep** — compare models on the golden set; report $/run (via
   provider token-counting), p50/p95 latency, and quality.
10. **CI gate** — GitHub Actions runs the harness and fails the build when
    recall@5 / unsupported-claim-rate cross thresholds.
11. **README / portfolio writeup** — surface the ablation table and dashboard up
    top; explain the design decisions and what each stage earned.

---
