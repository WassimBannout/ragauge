"""Eval harness — golden set, metrics, judge, RunReport (DESIGN.md §7).

Implemented: ``golden`` (load the hand-verified set), ``metrics`` (deterministic
recall@5 / MRR / unanswerable-precision — no LLM), ``judge`` (structured
LLM-as-judge), ``cost`` (real $/run from provider token counts), and ``run`` (the
T10 + T14–T16 orchestrator: golden set -> pipeline -> ``metrics.json``).

Still to come: BM25/RRF/rerank ablation table (T17–T20) and the CI gate (T22).
"""
