# RAGauge — *measure your RAG*

## Headline metrics

*Current as of 2026-06-29 — Slice 1 (ingest + dense retrieval) shipped; eval
harness not built yet, so the judged numbers are honestly unmeasured.*

| Metric | Value | Status |
|---|---|---|
| **recall@5** | — | lands at **T10** (next task) |
| **groundedness / supported-claim rate** | — | lands at **T14** (LLM judge) |
| **unsupported-claim rate** | — | lands at **T14** (LLM judge) |
| **$ / eval run** | — | lands at **T15** (provider token-counting) |
| p95 latency (dense retrieval) | **~96–186 ms** | ✅ measured (Slice 1); end-to-end at T15 |

> **Why dashes, not numbers?** The headline of this project is *honest*
> measurement. recall@5 is the first eval number and it lands one task away at
> **T10**; the judged generation metrics (groundedness, unsupported-claim rate)
> and real $/run arrive with the harness at **T14–T16**, and the full ablation
> table at **T20**. Values are shown blank rather than guessed — **honest numbers
> or none.** See [DESIGN.md](DESIGN.md) §7 and
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

🟢 **Slice 1 (ingest + dense retrieval, T1–T8) built, tested, and verified
end-to-end — 8 / 23 subtasks done.** Next: **T9** (golden set) → **T10** (first
recall@5 number). The headline metrics above stay blank until that first eval
run — *honest numbers or none.* See [`DESIGN.md`](DESIGN.md) §12 (build order)
and [`PRD.md` → Implementation status](PRD.md#implementation-status).

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

> Set a contact `User-Agent` for EDGAR via `SEC_USER_AGENT` in `.env` (SEC fair-access policy).
