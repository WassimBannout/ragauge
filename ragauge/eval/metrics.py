"""Deterministic retrieval metrics — no LLM (PRD §7.2, DESIGN.md §7.2).

recall@k, MRR, and unanswerable-precision are pure functions of
(gold_chunk_ids, ranked retrieved ids, abstention decisions). Keeping them
LLM-free is deliberate: these numbers are cheap, reproducible, and not hostage to
judge variance — they are computed every run regardless of whether the judge ran.
"""

from __future__ import annotations

from statistics import median


def recall_at_k(gold_chunk_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    """Fraction of gold chunks present in the top-k retrieved ids.

    For a single-gold question this is 1.0 (hit) or 0.0 (miss); for multi-hop it
    is the share of the required chunks that surfaced. Undefined (returns 0.0)
    for rows with no gold chunks — callers exclude unanswerable rows upstream.
    """
    if not gold_chunk_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    hits = sum(1 for g in gold_chunk_ids if g in top)
    return hits / len(gold_chunk_ids)


def reciprocal_rank(gold_chunk_ids: list[str], retrieved_ids: list[str]) -> float:
    """1 / rank of the first gold chunk in the full ranking, else 0.0."""
    gold = set(gold_chunk_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def percentiles(values: list[float]) -> dict[str, float]:
    """p50 / p95 / max for a latency series (empty -> zeros)."""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    s = sorted(values)
    idx95 = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return {"p50": median(s), "p95": s[idx95], "max": s[-1]}


def unanswerable_precision(
    abstained_flags: list[bool], is_unanswerable: list[bool]
) -> tuple[float | None, int, int]:
    """Of the questions the system abstained on, the share that were truly
    unanswerable. ``None`` when the system never abstained (precision undefined).

    Returns (precision, n_abstained, n_correct_abstentions). No LLM needed: the
    abstention signal is the dual-trigger gate / generator output, and the label
    is the golden ``type`` (DESIGN.md §6.2).
    """
    n_abstained = sum(abstained_flags)
    correct = sum(1 for a, u in zip(abstained_flags, is_unanswerable) if a and u)
    if n_abstained == 0:
        return None, 0, 0
    return correct / n_abstained, n_abstained, correct


def unanswerable_recall(
    abstained_flags: list[bool], is_unanswerable: list[bool]
) -> float | None:
    """Of the truly-unanswerable rows, the share the system abstained on.
    ``None`` if the set has no unanswerable rows."""
    n_unans = sum(is_unanswerable)
    if n_unans == 0:
        return None
    caught = sum(1 for a, u in zip(abstained_flags, is_unanswerable) if a and u)
    return caught / n_unans
