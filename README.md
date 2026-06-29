# RAGauge — *measure your RAG*

## Headline metrics

*Current as of 2026-06-29 — eval harness + retrieval ablation built. Retrieval
metrics are deterministic (no LLM) on the 30-row golden set (candidates; human
verification pending). Judged generation metrics need an `ANTHROPIC_API_KEY` run
and are unmeasured in this offline environment.*

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

### Generation (needs a judged run)

| Metric | Value | Status |
|---|---|---|
| **groundedness / supported-claim rate** | — | wired (`T14`); needs a judged run |
| **unsupported-claim rate** | — | wired (`T14`); needs a judged run |
| **$ / eval run** | — | wired (`T15`, provider token counts); needs a judged run |

> Retrieval recall@5 / MRR / p95 are pure functions of the ranking — measured
> here with no API key. The judged metrics (groundedness, unsupported-claim rate,
> $/run) are implemented and stub-validated; one `ragauge eval` run with a key
> fills the ablation's groundedness columns in. **Honest numbers or none.** See
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

🟢 **Slice 1 (ingest + dense retrieval), the eval harness, and the hybrid
retrieval ablation (T17–T20) are built and tested.** The deterministic ablation
(dense → +BM25/RRF → +rerank) is measured; the golden set is drafted (30
candidate rows, human verification pending) and the judged generation path is
implemented and stub-validated, awaiting an `ANTHROPIC_API_KEY` run. Next:
human-verify the golden set, add the embedding-model dimension to the ablation
(T20), then the model/cost sweep + CI gate (**T21–T23**). See
[`DESIGN.md`](DESIGN.md) §12 (build order) and
[`PRD.md` → Implementation status](PRD.md#implementation-status).

Verified on 3 real 10-Ks (AAPL FY2025, MSFT FY2025, NVDA FY2026): **789
structure-aware chunks**, section-labelled (Item 1 / 1A / 7 / 8), a stamped exact
dense index, and a config-toggleable `Retrieve` seam (dense + BM25 → RRF →
cross-encoder rerank) returning ranked evidence with per-stage provenance.
**34 unit tests** pass offline.

## Run it (Slice 1)

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e .                       # add torch CPU wheel: --extra-index-url https://download.pytorch.org/whl/cpu
pytest                                     # 19 tests, no model download needed

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
```

`--no-judge` runs **offline with no API key** — recall@5 / MRR /
unanswerable-precision are pure functions of the ranking, so the baseline never
depends on an LLM. A full run adds the grounded generator (cites `chunk_id`s or
abstains) and the structured LLM-as-judge (Pydantic `{supported,
unsupported_claims, score}`), writing per-question results + aggregates +
real `$/run` to `metrics.json`.

> Set a contact `User-Agent` for EDGAR via `SEC_USER_AGENT` in `.env` (SEC fair-access policy).
