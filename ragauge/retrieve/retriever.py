"""The Retrieve seam (PRD §S1.6 FR8 / T8, DESIGN.md §2.2).

``Retriever.retrieve(query, config) -> RetrievedChunk[]`` is the *only* surface
the eval harness and CLI talk to. Slice 1 implements the **dense** stage; the
other toggles raise until their slice lands, but the seam's shape is final so
BM25 / fusion / rerank attach without changing callers.
"""

from __future__ import annotations

from pathlib import Path

from ragauge.config import PipelineConfig, RetrievalConfig
from ragauge.contracts import ProvenanceEntry, RetrievedChunk, Stage
from ragauge.ingest.store import load_chunk_map
from ragauge.retrieve.embedder import Embedder
from ragauge.retrieve.index import ExactFlatIndex


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        index: ExactFlatIndex,
        chunk_map: dict,
    ):
        self.embedder = embedder
        self.index = index
        self.chunk_map = chunk_map

    def retrieve(
        self, query: str, config: RetrievalConfig | None = None
    ) -> list[RetrievedChunk]:
        config = config or RetrievalConfig()

        if config.bm25 or config.fusion or config.rerank:
            raise NotImplementedError(
                "BM25 / fusion / rerank are wired in later slices (T17–T19); "
                "Slice 1 serves dense-only."
            )
        if not config.dense:
            raise ValueError("no retrieval stage enabled (dense is off)")

        return self._dense(query, config.top_k)

    def _dense(self, query: str, top_k: int) -> list[RetrievedChunk]:
        qvec = self.embedder.encode_queries([query])[0]
        hits = self.index.search(qvec, top_k)
        results: list[RetrievedChunk] = []
        for rank, (chunk_id, score) in enumerate(hits, start=1):
            chunk = self.chunk_map.get(chunk_id)
            if chunk is None:  # index/store drift — skip rather than crash
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    stage_provenance=[
                        ProvenanceEntry(stage=Stage.DENSE, rank=rank, score=score)
                    ],
                )
            )
        return results


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
    invariant) plus the chunking config — refusing to serve a stale index."""
    expect_chunk = config.chunking.hash() if config is not None else None
    index = ExactFlatIndex.load(
        index_dir,
        expect_model_id=embedder.model_id,
        expect_chunking_hash=expect_chunk,
    )
    return Retriever(embedder, index, load_chunk_map(store_path))
