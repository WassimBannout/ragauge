"""T17–T19 — BM25, RRF fusion, cross-encoder rerank, and stage composition.

All deterministic and model-free: BM25 and RRF are pure functions, and the
reranker is exercised through an injected stub so the composition logic is tested
without downloading a cross-encoder.
"""

import numpy as np

from ragauge.config import PipelineConfig, RetrievalConfig
from ragauge.contracts import Chunk, Section, Stage
from ragauge.ingest.store import write_chunks
from ragauge.retrieve.bm25 import Bm25Index, tokenize
from ragauge.retrieve.embedder import HashingEmbedder
from ragauge.retrieve.fusion import reciprocal_rank_fusion
from ragauge.retrieve.index import build_dense_index
from ragauge.retrieve.retriever import Retriever, build_retriever


def _chunks() -> list[Chunk]:
    texts = {
        Section.ITEM_1A: "The company faces significant supply chain and component shortage risk.",
        Section.ITEM_7: "Total net sales were 383.3 billion dollars in fiscal 2023.",
        Section.ITEM_1: "The company designs phones tablets and personal computers.",
    }
    return [
        Chunk.create(
            doc_id="AAPL-10K-FY2023",
            company="AAPL",
            fiscal_year=2023,
            section=sec,
            anchor=f"{sec.value} · h · ¶1",
            text=txt,
        )
        for sec, txt in texts.items()
    ]


# --- BM25 ------------------------------------------------------------------- #
def test_bm25_matches_exact_figure_token():
    chunks = _chunks()
    idx = Bm25Index.from_chunks(chunks)
    hits = idx.search("383.3 billion net sales", k=3)
    assert hits, "expected a lexical match"
    assert hits[0][0] == chunks[1].chunk_id  # the ITEM_7 net-sales chunk
    assert hits[0][1] > 0


def test_bm25_keeps_decimals_as_one_token():
    assert "383.3" in tokenize("Total net sales were 383.3 billion")


def test_bm25_no_overlap_returns_empty():
    idx = Bm25Index.from_chunks(_chunks())
    assert idx.search("zzz nonexistent terminology", k=3) == []


# --- RRF -------------------------------------------------------------------- #
def test_rrf_rewards_consensus_across_lists():
    # 'b' is ranked highly by both lists; 'a' and 'c' each top only one.
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
    ids = [cid for cid, _ in fused]
    assert ids[0] == "b"
    # scores strictly descending
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_deterministic_tiebreak_by_id():
    fused = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k=60)
    # equal scores -> id-ascending order
    assert [cid for cid, _ in fused] == ["x", "y"]


# --- composition through the seam ------------------------------------------- #
def _build(tmp_path):
    emb = HashingEmbedder()
    chunks = _chunks()
    store = tmp_path / "chunks.jsonl"
    write_chunks(store, chunks)
    cfg = PipelineConfig()
    build_dense_index(
        chunks, emb, corpus_hash="c",
        chunking_config_hash=cfg.chunking.hash(), index_dir=tmp_path / "idx",
    )
    return build_retriever(emb, index_dir=tmp_path / "idx", store_path=store, config=cfg), chunks


def test_hybrid_fusion_carries_both_stage_provenance(tmp_path):
    retriever, chunks = _build(tmp_path)
    cfg = RetrievalConfig(dense=True, bm25=True, fusion=True, top_k=3, candidate_k=10)
    results = retriever.retrieve("supply chain shortage risk", cfg)
    assert results
    stages = {p.stage for r in results for p in r.stage_provenance}
    assert Stage.DENSE in stages and Stage.BM25 in stages
    # fused score ordering is descending
    assert all(
        results[i].score >= results[i + 1].score for i in range(len(results) - 1)
    )


class _StubReranker:
    """Scores by exact substring presence of the query — deterministic, no model."""

    model_id = "stub"

    def score(self, query, passages):
        return [float(query.lower() in p.lower()) for p in passages]


def test_rerank_reorders_and_appends_provenance(tmp_path):
    retriever, chunks = _build(tmp_path)
    retriever._reranker = _StubReranker()
    cfg = RetrievalConfig(
        dense=True, bm25=True, fusion=True, rerank=True,
        top_k=3, top_n=2, candidate_k=10,
    )
    results = retriever.retrieve("personal computers", cfg)
    assert len(results) == 2  # cut to top_n
    # the stub ranks the chunk containing the query string first
    assert results[0].chunk.chunk_id == chunks[2].chunk_id
    assert results[0].stage_provenance[-1].stage is Stage.RERANK
    assert results[0].stage_provenance[-1].rank == 1


def test_dense_only_unchanged_baseline(tmp_path):
    retriever, _ = _build(tmp_path)
    cfg = RetrievalConfig(top_k=2)  # dense-only defaults
    results = retriever.retrieve("supply chain risk", cfg)
    assert len(results) == 2
    assert all(
        p.stage is Stage.DENSE for r in results for p in r.stage_provenance
    )
