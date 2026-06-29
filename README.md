# RAGauge — *measure your RAG*

[![eval-gate](https://github.com/WassimBannout/ragauge/actions/workflows/eval-gate.yml/badge.svg)](https://github.com/WassimBannout/ragauge/actions/workflows/eval-gate.yml)

## Headline metrics

| metric | value | from |
|---|---|---|
| **recall@5** | **0.51** hybrid · 0.32 dense | retrieval ablation (deterministic, no LLM) |
| **groundedness / supported-claim rate** | **1.00** | judged CI baseline — dense, 13 of 25 answered |
| **unsupported-claim rate** | **0.00** | same judged run |
| **$ per eval run** | **$0.66** | 25-question dense judged run (provider token counts) |

*Current as of 2026-06-30 — eval harness, retrieval ablation, model sweep, a
PR-blocking CI gate (exhibited green and blocking), and golden-set verification
built. **recall@5** is the deterministic ablation winner
(hybrid; dense baseline 0.32). **groundedness / unsupported-claim rate / $-per-run**
are the first judged numbers, from the committed CI baseline
([`evals/baseline.json`](evals/baseline.json)) on **dense** retrieval: the system
answers 13 of 25 and abstains on the rest (recall@5 0.32, so it declines when the
gold chunk isn't retrieved), so groundedness/unsupported are over those 13 answered
rows. A full **hybrid + model-sweep** judged run (the richer picture) is the next
step. The 30-row golden set passes **automated verification** (structural + 25/25
answers self-consistent with their cited chunks, via `ragauge.eval.verify`); human
sign-off of the flagged judgment-calls is the last step — **honest numbers or none.***

### Retrieval ablation — recall lift per stage (`python -m ragauge.eval.ablation --no-judge`)

| config | recall@5 | MRR | retrieval p95 ms |
|---|---|---|---|
| dense-only (bge-base) | 0.32 | 0.17 | 197 |
| **+ BM25 + RRF (hybrid)** | **0.51** | **0.29** | 179 |
| + cross-encoder rerank | 0.34 | 0.28 | 2111 |

> **The defensible sentence:** *adding a BM25 sparse stage and fusing it with
> dense via RRF lifted recall@5 from 0.32 to 0.51 (+0.19) at no latency cost — but
> a generic MS-MARCO cross-encoder reranker did **not** pay for itself on 10-K
> text (recall@5 fell to 0.34, +~1900 ms p95), so it stays off by default.*
> That last row is the point of an ablation: BM25 earns its complexity here and
> the off-the-shelf reranker doesn't, and we can *prove* both with the table
> rather than assume a four-stage stack is better.
>
> Reranker model is itself a **measured** decision, not an assumed win. Three
> cross-encoders were swept — two MS-MARCO sizes plus a strong same-family
> `bge-reranker` — and **all three lose to hybrid on recall@5**, confirming it's
> the stage, not one weak checkpoint, that doesn't earn its cost here:
>
> | rerank model | recall@5 | MRR | retr p95 ms |
> |---|---|---|---|
> | *(none — hybrid)* | **0.51** | 0.29 | 179 |
> | `ms-marco-MiniLM-L-6-v2` (default) | 0.34 | 0.28 | 2111 |
> | `ms-marco-MiniLM-L-12-v2` | 0.42 | 0.39 | 5815 |
> | `BAAI/bge-reranker-base` | 0.42 | **0.41** | 23607 |
>
> The bigger/stronger cross-encoders lift MRR (they sharpen the *first* hit) but
> still drop recall@5 — they demote the second gold chunk of multi-hop questions
> out of the top 5 — at 10–130× the hybrid retrieval latency on CPU.

### Generation & model sweep (first judged numbers)

| Metric | Value | Status |
|---|---|---|
| **groundedness / supported-claim rate** | 1.00 | first judged run — CI baseline (dense, 13 answered of 25) |
| **unsupported-claim rate** | 0.00 | same run; small-n — fuller picture awaits the hybrid/sweep run |
| **$ / eval run** · **gen p95** | $0.66 · 5.7 s | 25-q dense judged baseline (provider token counts) |

> First judged numbers come from the committed CI baseline
> ([`evals/baseline.json`](evals/baseline.json)): on **dense** retrieval the system
> answers 13 of 25 and **abstains on the rest** (recall@5 is 0.32, so it correctly
> declines when the gold chunk isn't retrieved) — so groundedness 1.00 / unsupported
> 0.00 is over those 13 answered rows. The model sweep + hybrid retrieval give the
> fuller cost/quality picture below.

> A **model & cost sweep** (`python -m ragauge.eval.sweep` → `dashboard.md`) compares
> three Claude tiers (cheapest → balanced → most capable: haiku → sonnet → opus)
> on the fixed golden set — groundedness vs. **cost-per-query** vs. **p50/p95
> latency** — to pick the generator by data, not guess. It also prices **prompt
> caching on vs. off** (stable instruction prefix cached; per-question evidence
> after the breakpoint). Harness built and unit-tested; the dashboard fills in on a
> keyed run.
>
> Retrieval recall@5 / MRR / p95 are pure functions of the ranking — measured
> here with no API key. The judged metrics (groundedness, unsupported-claim rate,
> $/run) now have their first measured numbers from the committed CI baseline
> (dense, 25-q); the judged **ablation** columns (all three retrieval rungs) and the
> full model sweep are the next run. **Honest numbers or none.** See
> [DESIGN.md](DESIGN.md) §7 and
> [PRD.md → Implementation status](PRD.md#implementation-status).

An **eval-first** retrieval-augmented QA system over SEC 10-K filings. The
headline isn't the chatbot — it's the **evaluation harness**: a hand-verified
golden set, retrieval + groundedness metrics, an ablation that proves every
retrieval stage earns its complexity, a cost/latency/quality dashboard, and a
CI gate that blocks quality regressions.

> **Thesis:** *I can make an LLM system measurably reliable — and prove it with numbers.*

## What's here

- **[DESIGN.md](DESIGN.md)** — full architecture: ingest → retrieve → generate →
  eval harness, with diagrams and the signal-vs-table-stakes breakdown.
- **[PRD.md](PRD.md)** — epic, the 23-task scoped subtask checklist, and the
  living **Implementation status** snapshot.

## Status

🟢 **Slice 1 (ingest + dense retrieval), the eval harness, the hybrid retrieval
ablation (T17–T20), the model & cost sweep (T21), and the PR-blocking CI quality
gate (T22) are built and tested.** The deterministic ablation (dense → +BM25/RRF →
+rerank) is measured, and the judged generation path now has its first measured
numbers from the committed CI baseline (dense, 25-question subset). Both the
recall@5 and groundedness gates are live against `evals/baseline.json`. The golden
set is drafted (30 candidate rows) and passes automated verification (structural +
25/25 self-consistency); human sign-off is the last step. Next:
sign off the golden set, run the full hybrid + model-sweep judged picture, add
the embedding-model dimension to the ablation (T20), then the portfolio writeup
(**T23**). See
[`DESIGN.md`](DESIGN.md) §12 (build order) and
[`PRD.md` → Implementation status](PRD.md#implementation-status).

Verified on 3 real 10-Ks (AAPL FY2025, MSFT FY2025, NVDA FY2026): **789
structure-aware chunks**, section-labelled (Item 1 / 1A / 7 / 8), a stamped exact
dense index, and a config-toggleable `Retrieve` seam (dense + BM25 → RRF →
cross-encoder rerank) returning ranked evidence with per-stage provenance.
**62 unit tests** pass offline.

## Run it (Slice 1)

```bash
uv sync --extra dev                        # CPU-only torch via the pinned wheel index (pyproject [tool.uv.sources]); no CUDA
uv run pytest                              # 62 tests, no model download needed

ragauge acquire                            # download AAPL/MSFT/NVDA 10-Ks + manifest (EDGAR)
ragauge ingest                             # parse → segment → chunk → data/chunks.jsonl
ragauge inspect --doc AAPL --section ITEM_1A   # eyeball chunks + metadata
ragauge build-index                        # embed (bge-base-en-v1.5) → indexes/dense/
ragauge query "What supply-chain risks does the company face?"
```

## Run the eval harness

```bash
ragauge eval --no-judge                    # deterministic baseline: recall@5, MRR → metrics.json
ragauge eval                               # full run: + grounded generation & LLM-as-judge
                                           #   (needs ANTHROPIC_API_KEY; judge ≥ generator enforced)

# Hybrid retrieval stack is config-toggleable; pick a rung directly:
python -m ragauge.eval.run --retrieval hybrid+rerank --no-judge

# The thesis artifact — run all three rungs and print the comparison table:
python -m ragauge.eval.ablation --no-judge   # recall@5 / MRR / p95 per config → metrics_ablation.json
python -m ragauge.eval.ablation              # + groundedness / $/run (needs ANTHROPIC_API_KEY)

# Model & cost sweep — pick the generator by data (needs ANTHROPIC_API_KEY):
python -m ragauge.eval.sweep                 # haiku/sonnet/opus: groundedness vs $/query vs p50/p95
                                             #   + $/run with prompt caching on vs off → dashboard.md
python -m ragauge.eval.sweep --limit 3       # cheap smoke run over 3 questions

# Verify the golden set is trustworthy (T9): structural checks + self-consistency
python -m ragauge.eval.verify                 # structural only (free): ids resolve, cardinality, spread
python -m ragauge.eval.verify --judge         # + grade each gold answer vs its cited chunks (LLM)
```

## CI quality gate (blocks regressions)

[`.github/workflows/eval-gate.yml`](.github/workflows/eval-gate.yml) runs on **every PR**:
it builds the corpus + index (cached), runs the **25-question eval subset**, and
**fails the build** if `recall@5` or `groundedness` drops more than **0.05** below
the committed baseline ([`evals/baseline.json`](evals/baseline.json)). The metrics
delta is posted as a sticky **PR comment** (and the run summary).

```bash
# Reproduce the gate locally:
python -m ragauge.eval.run --limit 25 --out metrics.json   # judged; add --no-judge to skip the LLM
python -m ragauge.eval.gate --baseline evals/baseline.json --metrics metrics.json
#   → prints the delta table; exit 1 on a regression
```

- **recall@5 gates every PR for free** — it's deterministic, no API key, so even
  fork PRs are protected. **groundedness** is gated when a judged run is available
  (the `ANTHROPIC_API_KEY` secret on same-repo PRs); when it can't be measured the
  gate **skips** it rather than failing.
- The 25-question subset is the first 25 of the stably-ordered golden set, so the
  baseline and every PR run grade the **same** questions; a corpus/chunking change
  is flagged in the comment as "regenerate the baseline", not a phantom regression.

`--no-judge` runs **offline with no API key** — recall@5 / MRR /
unanswerable-precision are pure functions of the ranking, so the baseline never
depends on an LLM. A full run adds the grounded generator (cites `chunk_id`s or
abstains) and the structured LLM-as-judge (Pydantic `{supported,
unsupported_claims, score}`), writing per-question results + aggregates +
real `$/run` to `metrics.json`.

> Set a contact `User-Agent` for EDGAR via `SEC_USER_AGENT` in `.env` (SEC fair-access policy).
