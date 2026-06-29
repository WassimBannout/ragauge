"""Typed records that cross component boundaries (DESIGN.md §2.3).

The hard rule: data moves between ingest / retrieve / generate / eval as these
validated records, never as hidden global state. The load-bearing detail is the
**stable, content-addressed** ``chunk_id`` — citations and golden labels must
survive re-ingestion under the same chunking config (PRD §S1.3).
"""

from __future__ import annotations

import hashlib
from enum import Enum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Section(str, Enum):
    """10-K Item sections we segment as first-class retrieval filters.

    Items we do not target individually collapse to ``OTHER`` (PRD §S1.3).
    """

    ITEM_1 = "ITEM_1"  # Business
    ITEM_1A = "ITEM_1A"  # Risk Factors
    ITEM_7 = "ITEM_7"  # MD&A
    ITEM_8 = "ITEM_8"  # Financial Statements
    OTHER = "OTHER"


class ContentType(str, Enum):
    PROSE = "prose"
    TABLE = "table"
    FOOTNOTE = "footnote"


class Stage(str, Enum):
    """Retrieval stages that can surface a chunk. Only ``DENSE`` is wired in
    Slice 1; the rest exist so ``stage_provenance`` and config toggles are
    already shaped for the ablation (DESIGN.md §5)."""

    DENSE = "dense"
    BM25 = "bm25"
    FUSION = "fusion"
    RERANK = "rerank"


class GoldType(str, Enum):
    SINGLE_DOC = "single_doc"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"


# --------------------------------------------------------------------------- #
# Chunk — the atomic unit of evidence
# --------------------------------------------------------------------------- #
def compute_chunk_id(doc_id: str, section: Section, normalized_text: str) -> str:
    """Deterministic content-addressed id: ``{doc_id}:{section}:{sha256[:12]}``.

    No UUIDs, no timestamps — identical input + identical chunking config yields
    an identical id on every rebuild (PRD §S1.3). The ``doc_id``/``section``
    prefix prevents collisions when boilerplate recurs across filings.
    """
    digest = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}:{section.value}:{digest}"


class Chunk(BaseModel):
    """A structure-aware, size-bounded unit of a filing with full metadata."""

    doc_id: str
    company: str
    fiscal_year: int
    section: Section
    anchor: str
    chunk_id: str
    text: str
    content_type: ContentType = ContentType.PROSE
    token_count: int = 0
    char_span: tuple[int, int] | None = None
    source_path: str = ""

    @classmethod
    def create(
        cls,
        *,
        doc_id: str,
        company: str,
        fiscal_year: int,
        section: Section,
        anchor: str,
        text: str,
        content_type: ContentType = ContentType.PROSE,
        token_count: int = 0,
        char_span: tuple[int, int] | None = None,
        source_path: str = "",
    ) -> "Chunk":
        """Build a Chunk, deriving ``chunk_id`` from the (already normalized)
        text so the id is content-addressed by construction."""
        return cls(
            doc_id=doc_id,
            company=company,
            fiscal_year=fiscal_year,
            section=section,
            anchor=anchor,
            chunk_id=compute_chunk_id(doc_id, section, text),
            text=text,
            content_type=content_type,
            token_count=token_count,
            char_span=char_span,
            source_path=source_path,
        )


# --------------------------------------------------------------------------- #
# RetrievedChunk — a Chunk plus how retrieval surfaced it
# --------------------------------------------------------------------------- #
class ProvenanceEntry(BaseModel):
    """Which stage surfaced a chunk, at what rank and raw score.

    A chunk fused from multiple stages carries one entry per contributing
    stage — this is what makes ablation deltas attributable (DESIGN.md §2.3)."""

    stage: Stage
    rank: int
    score: float


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float  # final/fused score used for ordering
    stage_provenance: list[ProvenanceEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Answer — generation output (stub fields wired in a later slice)
# --------------------------------------------------------------------------- #
class Answer(BaseModel):
    text: str = ""
    citations: list[str] = Field(default_factory=list)  # chunk_ids
    abstained: bool = False
    evidence_used: list[str] = Field(default_factory=list)  # chunk_ids
    input_tokens: int = 0  # uncached input tokens (billed at full rate)
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0  # prefix written to cache this call
    cache_read_input_tokens: int = 0  # prefix served from cache this call
    cost_usd: float = 0.0
    latency_ms: float = 0.0


# --------------------------------------------------------------------------- #
# GoldRow — one hand-verified golden-set row (built at T9)
# --------------------------------------------------------------------------- #
class GoldRow(BaseModel):
    id: str
    question: str
    gold_answer: str
    gold_chunk_ids: list[str] = Field(default_factory=list)
    type: GoldType
    difficulty: str  # e.g. easy | medium | hard


# --------------------------------------------------------------------------- #
# RunReport — one eval run's persisted result (assembled at T16)
# --------------------------------------------------------------------------- #
class RunReport(BaseModel):
    config_hash: str
    corpus_hash: str
    embedding_model_id: str = ""
    generator_model_id: str = ""
    judge_model_id: str = ""
    timestamp: str = ""
    per_question: list[dict] = Field(default_factory=list)
    aggregates: dict = Field(default_factory=dict)
    cost_usd: float = 0.0
