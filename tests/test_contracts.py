"""T2 — data contracts: round-trip + content-addressed chunk_id stability."""

from ragauge.contracts import (
    Answer,
    Chunk,
    ContentType,
    GoldRow,
    GoldType,
    ProvenanceEntry,
    RetrievedChunk,
    RunReport,
    Section,
    Stage,
    compute_chunk_id,
)


def _chunk(text="Total revenue was $383 billion in fiscal 2023.") -> Chunk:
    return Chunk.create(
        doc_id="AAPL-10K-FY2023",
        company="AAPL",
        fiscal_year=2023,
        section=Section.ITEM_7,
        anchor='ITEM_7 · "MD&A" · ¶1',
        text=text,
    )


def test_chunk_id_is_content_addressed_and_deterministic():
    c1 = _chunk()
    c2 = _chunk()
    # Same input -> same id on every (re)build. No UUIDs / timestamps.
    assert c1.chunk_id == c2.chunk_id
    assert c1.chunk_id == compute_chunk_id(c1.doc_id, c1.section, c1.text)
    assert c1.chunk_id.startswith("AAPL-10K-FY2023:ITEM_7:")
    assert len(c1.chunk_id.rsplit(":", 1)[-1]) == 12


def test_chunk_id_changes_with_content_or_scope():
    base = _chunk()
    assert base.chunk_id != _chunk("Different text entirely.").chunk_id
    # Same text, different section -> different id (scope prefix prevents collisions).
    other = Chunk.create(
        doc_id="AAPL-10K-FY2023",
        company="AAPL",
        fiscal_year=2023,
        section=Section.ITEM_1,
        anchor="x",
        text=base.text,
    )
    assert other.chunk_id != base.chunk_id


def test_chunk_roundtrip():
    c = _chunk()
    assert Chunk.model_validate_json(c.model_dump_json()) == c


def test_retrieved_chunk_carries_stage_provenance():
    rc = RetrievedChunk(
        chunk=_chunk(),
        score=0.91,
        stage_provenance=[ProvenanceEntry(stage=Stage.DENSE, rank=1, score=0.91)],
    )
    rt = RetrievedChunk.model_validate_json(rc.model_dump_json())
    assert rt.stage_provenance[0].stage is Stage.DENSE
    assert rt == rc


def test_remaining_contracts_roundtrip():
    g = GoldRow(
        id="q1",
        question="What was revenue?",
        gold_answer="$383B",
        gold_chunk_ids=["AAPL-10K-FY2023:ITEM_7:abc123def456"],
        type=GoldType.SINGLE_DOC,
        difficulty="easy",
    )
    assert GoldRow.model_validate_json(g.model_dump_json()) == g

    a = Answer(text="x", citations=["c1"], abstained=False)
    assert Answer.model_validate_json(a.model_dump_json()) == a

    r = RunReport(config_hash="h1", corpus_hash="h2")
    assert RunReport.model_validate_json(r.model_dump_json()) == r
