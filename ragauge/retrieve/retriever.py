"""The Retrieve seam (PRD §S1.6 FR8 / T8, DESIGN.md §2.2).

``Retriever.retrieve(query, config) -> RetrievedChunk[]`` is the *only* surface
the eval harness and CLI talk to. Stages compose behind config toggles so each
is a config diff the ablation can isolate (DESIGN.md §5.2):

    query → {dense top-k, BM25 top-k} → RRF → fused pool → cross-encoder rerank
          → top-n

Every arrow is a toggle. ``stage_provenance`` records which stage(s) surfaced
each chunk and at what rank/score, so an ablation delta is attributable to the
stage that caused it.
"""

from __future__ import annotations

from pathlib import Path

from ragauge.config import PipelineConfig, RetrievalConfig
from ragauge.contracts import ProvenanceEntry, RetrievedChunk, Stage
from ragauge.ingest.store import load_chunk_map
from ragauge.retrieve.bm25 import Bm25Index
from ragauge.retrieve.embedder import Embedder
from ragauge.retrieve.fusion import reciprocal_rank_fusion
from ragauge.retrieve.index import ExactFlatIndex


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        index: ExactFlatIndex,
        chunk_map: dict,
        *,
        bm25_index: Bm25Index | None = None,
        reranker=None,
    ):
        self.embedder = embedder
        self.index = index
        self.chunk_map = chunk_map
        # BM25 and the cross-encoder are built lazily the first time a config
        # asks for them (and only then), but may be injected for tests.
        self._bm25 = bm25_index
        self._reranker = reranker

    # --- stage backends ----------------------------------------------------- #
    def _bm25_index(self) -> Bm25Index:
        if self._bm25 is None:
            self._bm25 = Bm25Index.from_chunks(list(self.chunk_map.values()))
        return self._bm25

    def _get_reranker(self, model_id: str):
        if self._reranker is None:
            from ragauge.retrieve.rerank import CrossEncoderReranker

            self._reranker = CrossEncoderReranker(model_id)
        return self._reranker

    # --- public seam -------------------------------------------------------- #
    def retrieve(
        self, query: str, config: RetrievalConfig | None = None
    ) -> list[RetrievedChunk]:
        config = config or RetrievalConfig()
        if not (config.dense or config.bm25):
            raise ValueError("no retrieval stage enabled (dense and bm25 both off)")

        # Base stages cast a wider net when a narrowing stage (fusion/rerank)
        # follows; otherwise they return the final top_k directly.
        depth = config.candidate_k if (config.fusion or config.rerank) else config.top_k

        # Per-stage ranked id lists + (rank, score) lookups for provenance.
        stage_hits: dict[Stage, list[tuple[str, float]]] = {}
        if config.dense:
            stage_hits[Stage.DENSE] = self._dense_hits(query, depth)
        if config.bm25:
            stage_hits[Stage.BM25] = self._bm25_index().search(query, depth)

        candidates = self._combine(stage_hits, config)

        if config.rerank:
            candidates = self._rerank(query, candidates[: config.candidate_k], config)
        else:
            candidates = candidates[: config.top_k]
        return candidates

    # --- combination -------------------------------------------------------- #
    def _combine(
        self, stage_hits: dict[Stage, list[tuple[str, float]]], config: RetrievalConfig
    ) -> list[RetrievedChunk]:
        """Fuse the active stages (RRF) or, with a single stage / fusion off,
        order by that stage. Both paths emit one ``ProvenanceEntry`` per stage
        that surfaced the chunk."""
        # rank/score per stage for provenance assembly.
        rank_score: dict[Stage, dict[str, tuple[int, float]]] = {
            stage: {cid: (rank, score) for rank, (cid, score) in enumerate(hits, 1)}
            for stage, hits in stage_hits.items()
        }

        if config.fusion and len(stage_hits) >= 2:
            ranked_lists = [[cid for cid, _ in hits] for hits in stage_hits.values()]
            fused = reciprocal_rank_fusion(ranked_lists, k=config.rrf_k)
            ordered = [(cid, score) for cid, score in fused]
        else:
            # Single stage, or multiple stages with fusion off: union, ordered by
            # best per-stage rank (then stable by id) — deterministic without
            # comparing incomparable raw scores.
            best_rank: dict[str, int] = {}
            for stage, hits in stage_hits.items():
                for rank, (cid, _) in enumerate(hits, 1):
                    best_rank[cid] = min(best_rank.get(cid, rank), rank)
            order = sorted(best_rank.items(), key=lambda kv: (kv[1], kv[0]))
            # Use the surfacing stage's own score as the ordering score.
            ordered = [
                (cid, self._any_score(cid, rank_score)) for cid, _ in order
            ]

        results: list[RetrievedChunk] = []
        for cid, score in ordered:
            chunk = self.chunk_map.get(cid)
            if chunk is None:  # index/store drift — skip rather than crash
                continue
            provenance = [
                ProvenanceEntry(stage=stage, rank=rs[cid][0], score=rs[cid][1])
                for stage, rs in rank_score.items()
                if cid in rs
            ]
            results.append(
                RetrievedChunk(chunk=chunk, score=score, stage_provenance=provenance)
            )
        return results

    @staticmethod
    def _any_score(
        cid: str, rank_score: dict[Stage, dict[str, tuple[int, float]]]
    ) -> float:
        for rs in rank_score.values():
            if cid in rs:
                return rs[cid][1]
        return 0.0

    # --- rerank ------------------------------------------------------------- #
    def _rerank(
        self, query: str, candidates: list[RetrievedChunk], config: RetrievalConfig
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        reranker = self._get_reranker(config.rerank_model)
        scores = reranker.score(query, [c.chunk.text for c in candidates])
        order = sorted(
            zip(candidates, scores),
            key=lambda cs: (-cs[1], cs[0].chunk.chunk_id),
        )[: config.top_n]
        reranked: list[RetrievedChunk] = []
        for rank, (rc, score) in enumerate(order, start=1):
            reranked.append(
                RetrievedChunk(
                    chunk=rc.chunk,
                    score=float(score),  # cross-encoder score is the final order
                    stage_provenance=rc.stage_provenance
                    + [ProvenanceEntry(stage=Stage.RERANK, rank=rank, score=float(score))],
                )
            )
        return reranked

    # --- dense -------------------------------------------------------------- #
    def _dense_hits(self, query: str, depth: int) -> list[tuple[str, float]]:
        qvec = self.embedder.encode_queries([query])[0]
        # DEMO REGRESSION — DO NOT MERGE. Inverting the query vector flips cosine
        # similarity so the index returns the *least*-relevant chunks, tanking
        # recall@5. This exists only to prove the CI quality gate blocks a
        # regression; revert before merging.
        qvec = -qvec
        return self.index.search(qvec, depth)


def build_retriever(
    embedder: Embedder,
    *,
    index_dir: str | Path,
    store_path: str | Path,
    config: PipelineConfig | None = None,
) -> Retriever:
    """Load stamped index + chunk store and return a ready Retriever.

    The index must have been built with the **same embedder now encoding
    queries**, so we validate the stamp against ``embedder.model_id`` (the binding
    invariant) plus the chunking config — refusing to serve a stale index. BM25
    is built lazily from the same chunk store on first use (no persisted
    artifact; see ``bm25.py``)."""
    expect_chunk = config.chunking.hash() if config is not None else None
    index = ExactFlatIndex.load(
        index_dir,
        expect_model_id=embedder.model_id,
        expect_chunking_hash=expect_chunk,
    )
    return Retriever(embedder, index, load_chunk_map(store_path))
