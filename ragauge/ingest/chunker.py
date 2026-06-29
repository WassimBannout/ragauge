"""Structure-aware, size-bounded chunking (PRD §S1.2 / FR4, DESIGN.md §4).

Order of operations is the whole point: **structure first, size second.**
Sections are already segmented; here we pack semantic units (sentences /
sub-headings) into chunks up to a token budget, **never splitting mid-sentence
or mid-row**, keep data tables atomic, and stamp every chunk with full metadata
and a content-addressed ``chunk_id``. A short caption header keeps "$X" attached
to "Total revenue, FY2023".
"""

from __future__ import annotations

import re
from typing import Callable

from ragauge.config import ChunkingConfig
from ragauge.contracts import Chunk, ContentType, Section
from ragauge.ingest.parse import Block
from ragauge.ingest.sections import SectionSpan

TokenCounter = Callable[[str], int]

# Sentence boundary: end punctuation + space + capital/quote/number start.
# A heuristic, documented as such — good enough for 10-K prose, never splits an
# abbreviation perfectly but errs toward *not* splitting (keeps sentences whole).
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=["\'(]?[A-Z0-9])')


def _word_count(text: str) -> int:
    """Fallback token counter (whitespace words) when no model tokenizer is
    injected — used by unit tests so they need no model download."""
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _caption(company: str, fiscal_year: int, section: Section, heading: str) -> str:
    head = heading.strip().strip('"')
    return f'[{company} · FY{fiscal_year} · {section.value} · "{head}"]'


def _pack(units: list[tuple[str, int]], target: int, overlap: int) -> list[list[int]]:
    """Pack unit indices into chunks up to ``target`` tokens with ``overlap``
    carried between consecutive chunks. Units are never split."""
    chunks: list[list[int]] = []
    n = len(units)
    i = 0
    while i < n:
        cur: list[int] = []
        tok = 0
        j = i
        while j < n:
            ut = units[j][1]
            if cur and tok + ut > target:
                break
            cur.append(j)
            tok += ut
            j += 1
        if not cur:  # a single unit larger than target — emit it alone, intact
            cur = [i]
            j = i + 1
        chunks.append(cur)
        if j >= n:
            break
        # Walk back to build the overlap window for the next chunk.
        ov = 0
        start = j
        k = j - 1
        while k >= i and ov + units[k][1] <= overlap:
            ov += units[k][1]
            start = k
            k -= 1
        i = max(start, i + 1)  # guarantee forward progress
    return chunks


class Chunker:
    def __init__(self, config: ChunkingConfig, count_tokens: TokenCounter | None = None):
        self.config = config
        self.count_tokens = count_tokens or _word_count

    # ----------------------------------------------------------------- #
    def chunk_span(
        self,
        span: SectionSpan,
        *,
        doc_id: str,
        company: str,
        fiscal_year: int,
        source_path: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        ordinal = 0
        heading = span.heading or span.section.value
        full_text = "\n\n".join(b.text for b in span.blocks)
        cursor = 0

        # Build prose runs interrupted by tables (which are atomic).
        prose_units: list[tuple[str, int]] = []

        def flush_prose() -> None:
            nonlocal ordinal, cursor
            if not prose_units:
                return
            packs = _pack(
                prose_units, self.config.target_tokens, self.config.overlap_tokens
            )
            for idxs in packs:
                content = " ".join(prose_units[k][0] for k in idxs)
                if self.count_tokens(content) < self.config.min_chunk_tokens and chunks:
                    # Merge a sliver into the previous chunk rather than emit it.
                    prev = chunks[-1]
                    merged = prev.text + " " + content
                    chunks[-1] = self._make_chunk(
                        merged_text=merged,
                        doc_id=doc_id,
                        company=company,
                        fiscal_year=fiscal_year,
                        section=span.section,
                        heading=heading,
                        ordinal=prev_ordinal[0],
                        content_type=ContentType.PROSE,
                        source_path=source_path,
                        char_span=prev.char_span,
                    )
                    continue
                ordinal += 1
                prev_ordinal[0] = ordinal
                cspan, cursor = _locate(full_text, content, cursor)
                chunks.append(
                    self._make_chunk(
                        content=content,
                        doc_id=doc_id,
                        company=company,
                        fiscal_year=fiscal_year,
                        section=span.section,
                        heading=heading,
                        ordinal=ordinal,
                        content_type=ContentType.PROSE,
                        source_path=source_path,
                        char_span=cspan,
                    )
                )
            prose_units.clear()

        prev_ordinal = [0]

        for block in span.blocks:
            if block.kind == "table":
                flush_prose()
                for piece in self._table_pieces(block):
                    ordinal += 1
                    prev_ordinal[0] = ordinal
                    cspan, cursor = _locate(full_text, block.text, cursor)
                    chunks.append(
                        self._make_chunk(
                            content=piece,
                            doc_id=doc_id,
                            company=company,
                            fiscal_year=fiscal_year,
                            section=span.section,
                            heading=heading,
                            ordinal=ordinal,
                            content_type=ContentType.TABLE,
                            source_path=source_path,
                            char_span=cspan,
                        )
                    )
                continue

            if block.looks_like_heading and block is not span.blocks[0]:
                # A sub-heading updates the locator for following prose, and is
                # itself kept as a prose unit so its text isn't lost.
                heading = block.text.strip()
            for sent in _split_sentences(block.text):
                prose_units.append((sent, self.count_tokens(sent)))

        flush_prose()
        return chunks

    # ----------------------------------------------------------------- #
    def _table_pieces(self, block: Block) -> list[str]:
        """A data table is one chunk; if it would exceed the model max, split on
        row groups, repeating the header row in each piece (PRD §S1.2.3)."""
        rows = block.table_rows
        budget = self.config.max_tokens - 32  # leave headroom for the caption
        if not rows or self.count_tokens(block.text) <= budget:
            return [block.text]

        header = rows[0]
        header_line = " | ".join(header)
        sep = " | ".join("---" for _ in header)
        pieces: list[str] = []
        cur: list[str] = []
        cur_tok = self.count_tokens(header_line)
        for row in rows[1:]:
            line = " | ".join(row)
            rt = self.count_tokens(line)
            if cur and cur_tok + rt > budget:
                pieces.append("\n".join([header_line, sep, *cur]))
                cur = []
                cur_tok = self.count_tokens(header_line)
            cur.append(line)
            cur_tok += rt
        if cur:
            pieces.append("\n".join([header_line, sep, *cur]))
        return pieces

    def _make_chunk(
        self,
        *,
        section: Section,
        doc_id: str,
        company: str,
        fiscal_year: int,
        heading: str,
        ordinal: int,
        content_type: ContentType,
        source_path: str,
        char_span: tuple[int, int] | None,
        content: str | None = None,
        merged_text: str | None = None,
    ) -> Chunk:
        if merged_text is not None:
            text = merged_text
        else:
            caption = _caption(company, fiscal_year, section, heading)
            text = f"{caption}\n{content}"
        anchor = f'{section.value} · "{heading.strip().strip(chr(34))}" · ¶{ordinal}'
        return Chunk.create(
            doc_id=doc_id,
            company=company,
            fiscal_year=fiscal_year,
            section=section,
            anchor=anchor,
            text=text,
            content_type=content_type,
            token_count=self.count_tokens(text),
            char_span=char_span,
            source_path=source_path,
        )


def _locate(haystack: str, needle: str, start: int) -> tuple[tuple[int, int] | None, int]:
    """Best-effort char span of ``needle`` in ``haystack`` from ``start`` (for
    inspection only). Returns (span_or_None, advanced_cursor)."""
    probe = needle[:80]
    idx = haystack.find(probe, start)
    if idx == -1:
        idx = haystack.find(probe)
    if idx == -1:
        return None, start
    return (idx, idx + len(needle)), idx + 1
