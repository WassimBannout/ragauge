"""Ingest orchestrator: filing HTML -> Chunks (PRD §S1, FR2–FR6).

Composes parse -> segment -> chunk for one filing, and runs the whole corpus to
a JSONL chunk store. The embedding model's own tokenizer is the right token
counter (it is the binding 512-token constraint, PRD §S1.2.2); callers inject
it, falling back to a whitespace counter for tests.
"""

from __future__ import annotations

from pathlib import Path

from ragauge.config import ChunkingConfig
from ragauge.contracts import Chunk
from ragauge.ingest.chunker import Chunker, TokenCounter
from ragauge.ingest.parse import parse_html
from ragauge.ingest.sections import segment_sections
from ragauge.ingest.store import write_chunks


def ingest_filing(
    html: str,
    *,
    doc_id: str,
    company: str,
    fiscal_year: int,
    source_path: str,
    config: ChunkingConfig | None = None,
    count_tokens: TokenCounter | None = None,
) -> list[Chunk]:
    config = config or ChunkingConfig()
    chunker = Chunker(config, count_tokens)
    blocks = parse_html(html)
    spans = segment_sections(blocks)
    chunks: list[Chunk] = []
    for span in spans:
        chunks.extend(
            chunker.chunk_span(
                span,
                doc_id=doc_id,
                company=company,
                fiscal_year=fiscal_year,
                source_path=source_path,
            )
        )
    return chunks


def ingest_corpus(
    manifest: dict,
    *,
    store_path: str | Path = "data/chunks.jsonl",
    config: ChunkingConfig | None = None,
    count_tokens: TokenCounter | None = None,
) -> list[Chunk]:
    """Ingest every filing in a manifest and persist the chunk store."""
    all_chunks: list[Chunk] = []
    for entry in manifest["filings"]:
        html = Path(entry["source_path"]).read_text(encoding="utf-8", errors="ignore")
        all_chunks.extend(
            ingest_filing(
                html,
                doc_id=entry["doc_id"],
                company=entry["company"],
                fiscal_year=entry["fiscal_year"],
                source_path=entry["source_path"],
                config=config,
                count_tokens=count_tokens,
            )
        )
    write_chunks(store_path, all_chunks)
    return all_chunks
