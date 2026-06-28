# CLAUDE.md — RAGauge

Project memory for Claude Code. Read this first; it points to the detailed docs.

## What this is
**RAGauge** — an *eval-first* retrieval-augmented QA system over SEC 10-K filings.
The headline is **the evaluation harness, not the chatbot**: a hand-verified
golden set, retrieval + groundedness metrics, a retrieval ablation, a
cost/latency/quality dashboard, and a CI gate that blocks quality regressions.

It is a **job-search portfolio piece**. The thesis it must prove: *I can make an
LLM system measurably reliable, and prove it with numbers.* Optimize every
decision for that signal.

## Source of truth (read these before acting)
- **`DESIGN.md`** — architecture (ingest / retrieve / generate / eval harness),
  data contracts, chunking, hybrid retrieval, abstention, diagrams. §12 has the
  build order; §13 has the signal-vs-table-stakes and interview probes.
- **`PRD.md`** — the epic, success metrics, and the scoped subtask checklist
  (the living progress tracker — update it as work lands).
- Work is also tracked as GitHub **Epic issue #1** (labels `epic`, `rag`, `eval`).

## How we work here
- **Plan before code.** This project is built by following a 10-step plan-first
  playbook. Design and PRD precede implementation — **do not write
  implementation code until the design/PRD for a slice are settled** (build
  order in `DESIGN.md` §12).
- **Built by a separate AI chat too.** The owner runs playbook prompts in
  another chat that edits these same files. **Verify the filesystem state — do
  not trust stale summaries.**
- **MVP / incremental.** Dense-only retrieval first for an honest baseline, then
  add BM25 + RRF + rerank behind config toggles so each stage can be ablated.
- **Everything measurable.** Don't claim "better" — show the before/after table.

## Conventions
- **Commits:** split-by-concern, one per playbook step / logical change. Clear
  messages (`docs:`, `feat:`, `chore:`). **No AI attribution** — never add
  "Co-authored-by" or similar.
- **Secrets:** never commit API keys. Use `.env` (already gitignored).
- **Data:** raw filings and built indexes are gitignored; keep the repo lean.
- **Models:** default to the latest Claude models for generation + judge; the
  judge needs a structured (Pydantic) output schema. **Verify exact current
  model IDs/pricing against the live Claude API reference before wiring them** —
  do not hardcode guessed IDs. Model choice is a measured decision via the sweep.
- **Cost:** use the provider token-counting API for $ figures, not a generic
  tokenizer.

## Headline metrics (keep current at the top of README.md)
recall@5 · groundedness / supported-claim rate · unsupported-claim rate ·
$ per eval run · p95 latency.

## At the end of every working session
Update `PRD.md` (progress: done / partial / pending / next steps / deviations)
and the README headline metrics. Living docs are how context survives across
sessions.
