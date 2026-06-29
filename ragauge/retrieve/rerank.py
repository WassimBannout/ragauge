"""Cross-encoder rerank (PRD T19, DESIGN.md §5.1).

A cross-encoder jointly encodes ``(query, passage)`` and scores their relevance
directly — strictly more accurate than the bi-encoder dense retriever, but
quadratically more expensive, so it runs **last, on a short candidate list**.
That latency/quality trade is the whole point of the ablation: the dashboard
shows what the +rerank stage buys and what it costs (DESIGN.md §5.3).

The model is loaded lazily (it downloads weights on first use) so the
deterministic retrieval baseline and the non-rerank configs never pay for it.
"""

from __future__ import annotations

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Wraps a sentence-transformers ``CrossEncoder`` behind a small ``score``
    surface so the retriever — and tests — depend only on the scoring contract,
    not the model."""

    def __init__(self, model_id: str = DEFAULT_RERANK_MODEL):
        from sentence_transformers import CrossEncoder  # lazy: heavy import

        self.model_id = model_id
        self._model = CrossEncoder(model_id)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance score per passage for the query (higher = more relevant)."""
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
