"""Reciprocal Rank Fusion (PRD T18, DESIGN.md §5.1).

Combine ranked lists from incomparable scoring spaces (dense cosine vs. BM25)
**by rank, not by raw score** — sidestepping the score-normalization problem
entirely. Each list contributes ``1 / (rrf_k + rank)`` to every id it ranks; the
fused order sums those contributions. Parameter-light and robust, which is why
it's the default fusion (DESIGN.md §5.1).
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one ``(chunk_id, rrf_score)`` ranking.

    ``k`` is the RRF constant (60 is the canonical default): larger ``k`` flattens
    the contribution curve, reducing how much top ranks dominate. Ties break by
    ``chunk_id`` for determinism (NFR1)."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
