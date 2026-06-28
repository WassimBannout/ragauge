# RAGauge — *measure your RAG*

An **eval-first** retrieval-augmented QA system over SEC 10-K filings. The
headline isn't the chatbot — it's the **evaluation harness**: a hand-verified
golden set, retrieval + groundedness metrics, an ablation that proves every
retrieval stage earns its complexity, a cost/latency/quality dashboard, and a
CI gate that blocks quality regressions.

> **Thesis:** *I can make an LLM system measurably reliable — and prove it with numbers.*

## Headline metrics

> _Populated once the eval harness runs (see [DESIGN.md](DESIGN.md) §7)._

| Metric | Value |
|---|---|
| recall@5 | _tbd_ |
| groundedness | _tbd_ |
| unsupported-claim rate | _tbd_ |
| $ / eval run | _tbd_ |
| p95 latency | _tbd_ |

## What's here

- **[DESIGN.md](DESIGN.md)** — full architecture: ingest → retrieve → generate →
  eval harness, with diagrams and the signal-vs-table-stakes breakdown.
- **PRD.md** — epic, scoped subtasks, and a living progress tracker _(in progress)_.

## Status

🚧 In active development. Built plan-first: design and PRD precede implementation.
See `DESIGN.md` §12 for the build order.
