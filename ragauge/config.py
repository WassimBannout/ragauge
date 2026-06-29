"""Declarative configuration that drives ingest + serving, plus stable hashing.

A run is identified by ``(corpus_hash, embedding_model_id, chunking_config_hash)``
(DESIGN.md §9). Every retrieval stage is a toggle here so later slices ablate by
config diff (DESIGN.md §10) — only ``dense`` is implemented in Slice 1, but the
shape is already correct.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel


def stable_hash(obj: object) -> str:
    """Deterministic short hash of any JSON-able object (sorted keys)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class ChunkingConfig(BaseModel):
    """Structure-aware chunking knobs (PRD §S1.2). Defaults are a documented
    starting point, validated/retuned against recall@5 at T10 — not tuned truth."""

    target_tokens: int = 400  # content tokens per prose chunk
    max_tokens: int = 512  # hard ceiling = embedding model max sequence length
    overlap_tokens: int = 56  # ~10–15% cross-boundary overlap
    min_chunk_tokens: int = 24  # drop slivers below this

    def hash(self) -> str:
        return stable_hash(self.model_dump())


class EmbeddingConfig(BaseModel):
    """Dense embedding backbone (PRD §S1.4). Local bge by default: deterministic,
    zero marginal cost, reproducible — the harness's deterministic backbone."""

    model_id: str = "BAAI/bge-base-en-v1.5"
    dim: int = 768
    max_seq_length: int = 512
    normalize: bool = True
    # bge/e5 are asymmetric: the query gets an instruction prefix, passages don't.
    query_instruction: str = "Represent this sentence for searching relevant passages: "


class RetrievalConfig(BaseModel):
    """Per-stage toggles + parameters. Slice 1 implements ``dense`` only; the
    other flags are the modularity seam for T17–T19 (DESIGN.md §5.1)."""

    dense: bool = True
    bm25: bool = False
    fusion: bool = False
    rerank: bool = False

    top_k: int = 5  # candidates per stage / final dense top-k
    top_n: int = 5  # final shortlist after rerank (unused until T19)
    rrf_k: int = 60  # RRF constant (unused until T18)
    min_score: float | None = None  # pre-generation evidence gate (unused until T13)


class PipelineConfig(BaseModel):
    """The single config object the harness/CLI hash into a RunReport."""

    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()

    def hash(self) -> str:
        return stable_hash(self.model_dump())
