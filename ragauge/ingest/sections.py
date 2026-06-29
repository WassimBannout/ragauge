"""Item segmentation (PRD §S1.2.1 / FR3).

Structure-first: split the block stream on **Item boundaries** before any
size-based chunking. The hard part is disambiguating real body headings from
their Table-of-Contents echoes — ToC entries are hyperlinks (``<a href>``) to
anchors, the body heading is the anchor target. So a heading that came from a
link is treated as a ToC echo and ignored as a section start.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragauge.contracts import Section
from ragauge.ingest.parse import Block

# Map a parsed "Item N" marker to the sections we target individually.
# Everything else (Items 2–6, 7A, 9–15, cover page) collapses to OTHER.
_ITEM_TO_SECTION = {
    "1": Section.ITEM_1,
    "1A": Section.ITEM_1A,
    "7": Section.ITEM_7,
    "8": Section.ITEM_8,
}

_ITEM_HEAD_RE = re.compile(r"^\s*item\s+(\d+[a-c]?)\b[.:)\s-]*(.*)$", re.IGNORECASE)


@dataclass
class SectionSpan:
    section: Section
    item_label: str  # raw item marker, e.g. "1A" (for OTHER too, when known)
    heading: str  # nearest body heading text
    blocks: list[Block]


def _item_marker(block: Block) -> str | None:
    """Return the normalized item marker (e.g. ``1A``) if this block is a real
    body Item *section heading*, else None.

    Two false positives are rejected: (a) ToC echoes, which are hyperlinks
    (``is_link``); (b) **running page headers** — many filers repeat a bare
    "Item 1A" at the top of every page. A genuine section heading carries a
    **title after the number** ("Item 1A. Risk Factors"); a bare marker does
    not. Requiring a title kills the running-header false boundaries."""
    if not block.looks_like_heading or block.is_link:
        return None
    m = _ITEM_HEAD_RE.match(block.text)
    if not m:
        return None
    title = m.group(2) or ""
    if not any(ch.isalpha() for ch in title):
        return None  # bare "Item 1A" with no title -> running header, not a start
    return m.group(1).upper()


def segment_sections(blocks: list[Block]) -> list[SectionSpan]:
    """Split blocks into contiguous section spans.

    Text before the first recognized Item is ``OTHER`` (cover page); text under
    an unrecognized Item inherits ``OTHER`` but keeps its raw marker for anchors.
    """
    spans: list[SectionSpan] = []
    current = SectionSpan(Section.OTHER, item_label="", heading="Cover", blocks=[])

    for block in blocks:
        marker = _item_marker(block)
        if marker is not None:
            # Close the running span (if it has content) and open a new one.
            if current.blocks:
                spans.append(current)
            section = _ITEM_TO_SECTION.get(marker, Section.OTHER)
            heading = block.text.strip()
            current = SectionSpan(
                section=section, item_label=marker, heading=heading, blocks=[block]
            )
        else:
            current.blocks.append(block)

    if current.blocks:
        spans.append(current)

    spans = _merge_repeat_sections(spans)
    return _promote_incorporated_financials(spans)


# Headings that mark the *actual* consolidated financial statements (Item 8 content).
_FIN_ANCHOR = re.compile(
    r"report of independent registered public accounting firm"
    r"|consolidated balance sheet"
    r"|consolidated statements of (income|operations)",
    re.IGNORECASE,
)
# Below this character count, an Item 8 span is a by-reference stub, not the
# statements themselves (e.g. NVDA: "...set forth in our Consolidated Financial
# Statements...included in this Annual Report"). AAPL/MSFT inline Item 8 is far larger.
_STUB_ITEM8_CHARS = 1200


def _promote_incorporated_financials(spans: list[SectionSpan]) -> list[SectionSpan]:
    """Relabel financials incorporated into Item 8 *by reference*.

    Some filers (e.g. NVDA) put a one-line Item 8 stub and place the consolidated
    statements physically in the Item 15 schedules region. Those F-pages *are*
    Item 8 content, so for a section-aware finance system they must carry the
    ``ITEM_8`` label. Tightly gated: fires only when the detected Item 8 span is a
    stub, so inline-Item-8 filings are untouched. (A documented §DESIGN 14
    robustness measure.)"""
    item8 = [s for s in spans if s.section is Section.ITEM_8]
    if not item8 or sum(len(b.text) for b in item8[0].blocks) >= _STUB_ITEM8_CHARS:
        return spans  # no Item 8, or it is already inline — nothing to do

    # The real F-pages are by far the largest OTHER span containing the anchor;
    # incidental mentions (a ToC line, an MD&A reference) sit in tiny spans. Pick
    # the largest qualifying span so we promote the statements, not a mention.
    def _anchor_idx(span: SectionSpan) -> int | None:
        return next(
            (i for i, b in enumerate(span.blocks) if _FIN_ANCHOR.search(b.text)), None
        )

    candidates = [
        (i, _anchor_idx(s))
        for i, s in enumerate(spans)
        if s.section is Section.OTHER and _anchor_idx(s) is not None
    ]
    if not candidates:
        return spans
    target_i, idx = max(
        candidates, key=lambda c: sum(len(b.text) for b in spans[c[0]].blocks)
    )

    out: list[SectionSpan] = []
    for i, span in enumerate(spans):
        if i != target_i:
            out.append(span)
            continue
        if span.blocks[:idx]:
            out.append(
                SectionSpan(Section.OTHER, span.item_label, span.heading, span.blocks[:idx])
            )
        out.append(
            SectionSpan(
                Section.ITEM_8,
                "8",
                "Financial Statements (incorporated by reference)",
                span.blocks[idx:],
            )
        )
    return out


def _merge_repeat_sections(spans: list[SectionSpan]) -> list[SectionSpan]:
    """If a target Item appears more than once (e.g. a stray ToC marker slipped
    through), keep the **largest** span for each target section and demote the
    smaller duplicates to OTHER — the body section is far larger than an echo."""
    # Index target spans by section.
    best: dict[Section, int] = {}
    for i, span in enumerate(spans):
        if span.section is Section.OTHER:
            continue
        size = sum(len(b.text) for b in span.blocks)
        if span.section not in best or size > sum(
            len(b.text) for b in spans[best[span.section]].blocks
        ):
            best[span.section] = i

    out: list[SectionSpan] = []
    for i, span in enumerate(spans):
        if span.section is not Section.OTHER and best.get(span.section) != i:
            span = SectionSpan(Section.OTHER, span.item_label, span.heading, span.blocks)
        out.append(span)
    return out
