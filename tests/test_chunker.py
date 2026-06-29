"""T5 — structure-aware chunking: tables atomic, no mid-row/mid-sentence splits,
full metadata, token budget respected (PRD §S1.2 / S1.8.3)."""

from ragauge.config import ChunkingConfig
from ragauge.contracts import ContentType, Section
from ragauge.ingest.chunker import Chunker
from ragauge.ingest.parse import Block
from ragauge.ingest.sections import SectionSpan


def _span(blocks, section=Section.ITEM_7, heading="MD&A"):
    return SectionSpan(section=section, item_label="7", heading=heading, blocks=blocks)


META = dict(
    doc_id="AAPL-10K-FY2023",
    company="AAPL",
    fiscal_year=2023,
    source_path="data/raw/AAPL-10K-FY2023.htm",
)


def test_data_table_kept_atomic_with_label_value_binding():
    # The §4.3 / probe-#1 exhibit: a financial table a naive character splitter
    # would shatter, kept whole so "$383,285" stays bound to "Total net sales".
    table = Block(
        text="($M) | FY2023 | FY2022\n--- | --- | ---\nTotal net sales | 383,285 | 394,328",
        kind="table",
        table_rows=[
            ["($M)", "FY2023", "FY2022"],
            ["Total net sales", "383,285", "394,328"],
        ],
    )
    chunker = Chunker(ChunkingConfig())
    chunks = chunker.chunk_span(_span([table]), **META)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.content_type is ContentType.TABLE
    # Number and its row label survive together in one chunk.
    assert "Total net sales" in c.text and "383,285" in c.text
    assert "FY2023" in c.text  # column header retained


def test_full_metadata_and_caption_header():
    span = _span([Block(text="Revenue grew ten percent this year.", kind="prose")])
    c = Chunker(ChunkingConfig()).chunk_span(span, **META)[0]
    assert c.doc_id == "AAPL-10K-FY2023"
    assert c.company == "AAPL" and c.fiscal_year == 2023
    assert c.section is Section.ITEM_7
    assert c.anchor.startswith('ITEM_7 · "MD&A" · ¶')
    assert c.chunk_id.startswith("AAPL-10K-FY2023:ITEM_7:")
    assert c.token_count > 0
    assert c.text.startswith("[AAPL · FY2023 · ITEM_7 · ")  # caption header


def test_no_midsentence_split_and_budget_respected():
    sentences = [f"Sentence number {i} ends here." for i in range(40)]
    span = _span([Block(text=" ".join(sentences), kind="prose")])
    cfg = ChunkingConfig(target_tokens=20, overlap_tokens=4, max_tokens=64, min_chunk_tokens=4)
    chunks = Chunker(cfg).chunk_span(span, **META)

    assert len(chunks) > 1  # actually split
    for c in chunks:
        body = c.text.split("\n", 1)[1]  # drop caption line
        # Every chunk ends on a sentence boundary -> no mid-sentence cut.
        assert body.rstrip().endswith(".")
        # Content stays within a sane multiple of the budget (sentences are atomic).
        assert c.token_count <= cfg.max_tokens + 20


def test_consecutive_chunks_overlap():
    sentences = [f"Alpha beta gamma delta number {i} here." for i in range(30)]
    span = _span([Block(text=" ".join(sentences), kind="prose")])
    cfg = ChunkingConfig(target_tokens=24, overlap_tokens=8, max_tokens=64, min_chunk_tokens=4)
    chunks = Chunker(cfg).chunk_span(span, **META)
    # Some sentence from the tail of chunk i reappears at the head of chunk i+1.
    first_tail = chunks[0].text.split(".")[-2]
    assert first_tail.strip()[:10] in chunks[1].text
