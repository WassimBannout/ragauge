# RAGauge — model & cost sweep

Generator picked by data, not guess (PRD T21). Golden set `data/golden/candidates.jsonl` (3 questions), retrieval held at `dense`, judge fixed at `claude-opus-4-8` (≥ every generator under test).

- corpus `60707081f218` · config `7e1319e7b4e4` · caching `on` · 2026-06-29T14:02:31.568277+00:00
- cost from the provider's billed token `usage` (never a generic tokenizer); latency is wall-clock per generation call.

## Quality vs. cost vs. latency

| tier | model | recall@5 | MRR | grounded-rate | unsupp-rate | mean score | gen p50/p95 ms | $/query | $/run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cheapest | claude-haiku-4-5 | 0.3333 | 0.0833 | 1.0000 | 0.0000 | 1.0000 | 1885 / 4613 | $0.0123 | $0.0369 |
| balanced | claude-sonnet-4-6 | 0.3333 | 0.0833 | 0.6667 | 0.3333 | 0.6667 | 3163 / 7475 | $0.0250 | $0.0749 |
| most capable | claude-opus-4-8 | 0.3333 | 0.0833 | 1.0000 | 0.0000 | 1.0000 | 2851 / 4090 | $0.0271 | $0.0814 |

**Recommendation.** `claude-haiku-4-5` (cheapest) — top groundedness *and* lowest cost (1.0000 at $0.0123/query). Clear pick.

## Prompt caching — $/run on vs. off

| model | $/run cache-off | $/run cache-on | saving | cache-read tok | cache-write tok |
| --- | --- | --- | --- | --- | --- |
| claude-haiku-4-5 | $0.0369 | $0.0369 | $0.0000 (0.0%) | 0 | 0 |
| claude-sonnet-4-6 | $0.0749 | $0.0749 | $0.0000 (0.0%) | 0 | 0 |
| claude-opus-4-8 | $0.0814 | $0.0814 | $0.0000 (0.0%) | 0 | 0 |

_The provider reported **zero** cached tokens: the stable instruction prefix is below the model's minimum cacheable size (1024–4096 tokens), so it silently does not cache and the realized saving is $0. The breakpoint is placed correctly (instructions cached, per-question evidence after it) — caching would pay off here only with a larger shared prefix (e.g. a big few-shot block or a shared document)._

## Note on recall

`recall@5` is **identical across models** — it is a property of the retrieval stack, not the generator, so the model sweep cannot move it. Recall *lift* is the headline of the retrieval ablation (`metrics_ablation.json` / `python -m ragauge.eval.ablation`); it is surfaced here only to make the separation of concerns explicit.
