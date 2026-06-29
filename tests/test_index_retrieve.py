"""T7/T8 — exact index build/search, stamp enforcement, and the dense seam."""

import numpy as np
import pytest

from ragauge.config import PipelineConfig, RetrievalConfig
from ragauge.contracts import Chunk, Section, Stage
from ragauge.ingest.store import write_chunks
from ragauge.retrieve.embedder import HashingEmbedder
from ragauge.retrieve.index import (
    ExactFlatIndex,
    IndexStampMismatch,
    build_dense_index,
)
from ragauge.retrieve.retriever import Retriever, build_retriever


def _chunks() -> list[Chunk]:
    texts = {
        Section.ITEM_1A: "The company faces significant supply chain and component shortage risk.",
        Section.ITEM_7: "Total net sales were 383 billion dollars in fiscal 2023.",
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


def test_exact_search_ranks_nearest_first():
    emb = HashingEmbedder()
    chunks = _chunks()
    index = build_dense_index(
        chunks, emb, corpus_hash="c", chunking_config_hash="k", index_dir=_tmpdir()
    )
    qvec = emb.encode_queries(["supply chain shortage risk"])[0]
    hits = index.search(qvec, k=3)
    assert hits[0][0] == chunks[0].chunk_id  # the ITEM_1A risk chunk
    assert hits[0][1] >= hits[-1][1]  # scores descending


def test_stamp_mismatch_refuses_to_load(tmp_path):
    emb = HashingEmbedder()
    build_dense_index(
        _chunks(), emb, corpus_hash="c1", chunking_config_hash="k1", index_dir=tmp_path
    )
    # Right model, wrong corpus hash -> refuse rather than silently serve stale.
    with pytest.raises(IndexStampMismatch):
        ExactFlatIndex.load(
            tmp_path, expect_model_id=emb.model_id, expect_corpus_hash="OTHER"
        )


def test_retriever_seam_returns_dense_provenance(tmp_path):
    emb = HashingEmbedder()
    chunks = _chunks()
    store = tmp_path / "chunks.jsonl"
    write_chunks(store, chunks)
    cfg = PipelineConfig()
    build_dense_index(
        chunks,
        emb,
        corpus_hash="c",
        chunking_config_hash=cfg.chunking.hash(),
        index_dir=tmp_path / "idx",
    )
    retriever = build_retriever(
        emb, index_dir=tmp_path / "idx", store_path=store, config=cfg
    )
    cfg.retrieval.top_k = 2
    results = retriever.retrieve("supply chain risk", cfg.retrieval)

    assert len(results) == 2
    assert results[0].stage_provenance[0].stage is Stage.DENSE
    assert results[0].stage_provenance[0].rank == 1
    assert results[0].score >= results[1].score


def test_no_stage_enabled_raises():
    # dense and bm25 both off -> nothing to retrieve.
    retriever = Retriever(HashingEmbedder(), _dummy_index(), {})
    with pytest.raises(ValueError):
        retriever.retrieve("q", RetrievalConfig(dense=False, bm25=False))


# --- helpers ---------------------------------------------------------------- #
def _tmpdir():
    import tempfile

    return tempfile.mkdtemp()


def _dummy_index():
    return ExactFlatIndex(np.zeros((1, 4), dtype=np.float32), ["x"], {})
