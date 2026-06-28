# RAGauge — *measure your RAG*

## Headline metrics

| Metric | Value |
|---|---|
| recall@5 | _not yet measured_ |
| groundedness / supported-claim rate | _not yet measured_ |
| unsupported-claim rate | _not yet measured_ |
| $ / eval run | _not yet measured_ |
| p95 latency | _not yet measured_ |

> **Current as of 2026-06-29 — pre-baseline (no eval run yet).** These populate
> from the first eval run: the dense-only retrieval baseline (recall@5) lands at
> subtask **T10**, the judged generation + cost/latency metrics at **T14–T16**,
> and the full ablation table at **T20**. They are shown blank rather than
> guessed — **honest numbers or none.** See [DESIGN.md](DESIGN.md) §7 and the
> progress tracker in [PRD.md](PRD.md).

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

🚧 **Planning complete; Slice 1 (ingest + dense retrieval, T2–T8) fully specced
and build-ready — 0 / 23 subtasks coded (next: T1, scaffolding).** Built
plan-first: design and PRD precede implementation. The embedding model is a
decided baseline (`bge-base-en-v1.5`) *and* a measured ablation dimension. See
[`DESIGN.md`](DESIGN.md) §12 for the build order and
[`PRD.md` → Implementation status](PRD.md#implementation-status) for the
detailed snapshot.
