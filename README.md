# RAGauge — *measure your RAG*

## Headline metrics

*Current as of 2026-06-29 — eval harness built. Retrieval metrics are the
**honest dense-only baseline** on the 30-row golden set (candidates; human
verification pending). Judged generation metrics need an `ANTHROPIC_API_KEY` run
and are unmeasured in this offline environment.*

| Metric | Value | Status |
|---|---|---|
| **recall@5** (dense-only) | **0.32** | ✅ measured — `T10`, deterministic, no LLM |
| **MRR** (dense-only) | **0.17** | ✅ measured — `T10`, deterministic, no LLM |
| **groundedness / supported-claim rate** | — | wired (`T14`); needs a judged run |
| **unsupported-claim rate** | — | wired (`T14`); needs a judged run |
| **$ / eval run** | — | wired (`T15`, provider token counts); needs a judged run |
| p95 latency (dense retrieval) | **~189 ms** | ✅ measured over the golden set |

> **The numbers are deliberately unflattering.** recall@5 = 0.32 is the *honest*
> dense-only baseline — bge struggles on the numeric/table questions that
> dominate a 10-K golden set. That gap is the point: it's the headroom the
> BM25 + RRF + rerank ablation (**T17–T20**) has to earn back, measured. The
> judged metrics (groundedness, unsupported-claim rate, $/run) are implemented and
> validated on stubbed calls; one `ragauge eval` run with an API key fills them
> in. **Honest numbers or none.** See [DESIGN.md](DESIGN.md) §7 and
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

🟢 **Slice 1 (ingest + dense retrieval) + the eval harness are built and
tested.** The deterministic retrieval baseline (recall@5 / MRR) is measured; the
golden set is drafted (30 candidate rows, human verification pending) and the
judged generation path is implemented and stub-validated, awaiting an
`ANTHROPIC_API_KEY` run. Next: human-verify the golden set, then the
BM25 + RRF + rerank ablation (**T17–T20**). See [`DESIGN.md`](DESIGN.md) §12
(build order) and [`PRD.md` → Implementation status](PRD.md#implementation-status).

Verified on 3 real 10-Ks (AAPL FY2025, MSFT FY2025, NVDA FY2026): **789
structure-aware chunks**, section-labelled (Item 1 / 1A / 7 / 8), a stamped exact
dense index, and a config-toggleable `Retrieve` seam returning ranked evidence
(query p95 < 200 ms). **19 unit tests** pass offline.

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
```

`--no-judge` runs **offline with no API key** — recall@5 / MRR /
unanswerable-precision are pure functions of the ranking, so the baseline never
depends on an LLM. A full run adds the grounded generator (cites `chunk_id`s or
abstains) and the structured LLM-as-judge (Pydantic `{supported,
unsupported_claims, score}`), writing per-question results + aggregates +
real `$/run` to `metrics.json`.

> Set a contact `User-Agent` for EDGAR via `SEC_USER_AGENT` in `.env` (SEC fair-access policy).
