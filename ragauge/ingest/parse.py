"""HTML 10-K -> ordered, normalized structural blocks (PRD §S1.1).

We parse the **primary rendered HTML** (iXBRL), not the XBRL facts — the MVP
reads the rendered text. The output is a flat, in-document-order list of
:class:`Block`s (headings / prose / tables); section segmentation and chunking
consume that stream. Tables are classified layout-vs-data and linearized so a
number never strands from its row label and column header.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from ragauge.ingest.normalize import normalize_text

# Tags that contribute no readable content.
_DROP_TAGS = {"script", "style", "head", "meta", "link", "noscript"}
# Block-level tags whose text we treat as a paragraph unit.
_BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

_ITEM_RE = re.compile(r"^\s*item\s+\d+[a-c]?\b", re.IGNORECASE)


@dataclass
class Block:
    """One linear unit of a filing in document order."""

    text: str
    kind: str  # "heading" | "prose" | "table"
    is_link: bool = False  # text came from/inside an <a href> — a ToC echo signal
    looks_like_heading: bool = False  # short, bold/caps, or an Item marker
    table_rows: list[list[str]] = field(default_factory=list)  # data tables only


# --------------------------------------------------------------------------- #
# Table handling
# --------------------------------------------------------------------------- #
def _table_cells(table: Tag) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [
            normalize_text(td.get_text(" ", strip=True))
            for td in tr.find_all(["td", "th"])
        ]
        # Drop fully empty rows (spacer rows are rampant in 10-K markup).
        if any(c for c in cells):
            rows.append(cells)
    return rows


def _is_data_table(rows: list[list[str]]) -> bool:
    """Numeric-density heuristic: real financial tables are mostly numbers laid
    out in a grid; layout tables (used for indentation/columns of prose) are not.
    """
    if len(rows) < 2:
        return False
    n_cols = max((len(r) for r in rows), default=0)
    if n_cols < 2:
        return False

    numeric = total = 0
    for row in rows:
        for cell in row:
            if not cell:
                continue
            total += 1
            # A cell that is mostly digits/currency/parens counts as numeric.
            if re.search(r"\d", cell) and re.fullmatch(r"[\d.,()$%\s+\-–—]*", cell):
                numeric += 1
    if total == 0:
        return False
    return (numeric / total) >= 0.30


def _linearize_data_table(rows: list[list[str]]) -> str:
    """Serialize a data table so each value stays adjacent to its row label and
    column header (PRD §S1.1.4). First non-empty row is treated as the header."""
    rows = [[c for c in r] for r in rows if any(c for c in r)]
    if not rows:
        return ""
    header = rows[0]
    lines = [" | ".join(header)]
    lines.append(" | ".join("---" for _ in header))
    for row in rows[1:]:
        lines.append(" | ".join(row))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Heading detection
# --------------------------------------------------------------------------- #
def _looks_like_heading(text: str, tag_name: str) -> bool:
    if tag_name in _HEADING_TAGS:
        return True
    if _ITEM_RE.match(text):
        return True
    words = text.split()
    if 0 < len(words) <= 12:
        letters = [c for c in text if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
            return True  # ALL-CAPS short line
    return False


def _contains_link(tag: Tag) -> bool:
    if tag.name == "a" and tag.get("href"):
        return True
    return tag.find("a", href=True) is not None


# --------------------------------------------------------------------------- #
# DOM walk
# --------------------------------------------------------------------------- #
def _drop_hidden(soup: BeautifulSoup) -> None:
    # Work in materialized passes — decomposing/unwrapping mutates the tree, so a
    # live find_all(True) iterator would hand back stale (attrs=None) tags.
    for tag in list(soup.find_all(_DROP_TAGS)):
        tag.decompose()

    for tag in list(soup.find_all(lambda t: t.name and t.name.startswith("ix:"))):
        if tag.decomposed:
            continue
        if tag.name == "ix:hidden":
            tag.decompose()  # hidden XBRL facts: drop entirely
        else:
            tag.unwrap()  # ix:nonNumeric/nonFraction wrap *visible* text: keep it

    for tag in list(soup.find_all(True)):
        if tag.decomposed or not tag.attrs:
            continue
        style = (tag.get("style") or "").replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            tag.decompose()


def parse_html(html: str) -> list[Block]:
    """Parse a filing's primary HTML into ordered :class:`Block`s."""
    # iXBRL filings are often served with a leading XML declaration, which makes
    # bs4 want an XML parser. We deliberately parse the *rendered* tree as HTML
    # (PRD §S1.1.1); strip the declaration so the HTML parser is happy.
    html = re.sub(r"^\s*<\?xml[^>]*\?>", "", html, count=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    _drop_hidden(soup)
    body = soup.body or soup

    blocks: list[Block] = []
    seen_tables: set[int] = set()

    for element in body.descendants:
        if not isinstance(element, Tag):
            continue

        if element.name == "table":
            if id(element) in seen_tables:
                continue
            seen_tables.add(id(element))
            # Skip nested tables already covered by an ancestor table.
            if element.find_parent("table") is not None:
                continue
            rows = _table_cells(element)
            if _is_data_table(rows):
                text = _linearize_data_table(rows)
                if text:
                    blocks.append(Block(text=text, kind="table", table_rows=rows))
            else:
                # Layout table -> unwrap to prose (one block of joined cell text).
                flat = normalize_text(element.get_text(" ", strip=True))
                if flat:
                    blocks.append(
                        Block(
                            text=flat,
                            kind="prose",
                            is_link=_contains_link(element),
                            looks_like_heading=_looks_like_heading(flat, "div"),
                        )
                    )
            continue

        if element.name in _BLOCK_TAGS and element.find_parent("table") is None:
            # Only take leaf-ish blocks: skip a container whose text is just its
            # block children concatenated (those children emit their own blocks).
            if element.find(_BLOCK_TAGS - {"tr"}) is not None:
                continue
            text = normalize_text(element.get_text(" ", strip=True))
            if not text:
                continue
            is_heading = _looks_like_heading(text, element.name)
            blocks.append(
                Block(
                    text=text,
                    kind="heading" if is_heading else "prose",
                    is_link=_contains_link(element),
                    looks_like_heading=is_heading,
                )
            )

    return _dedupe_adjacent(blocks)


def _dedupe_adjacent(blocks: list[Block]) -> list[Block]:
    """Collapse immediately-repeated identical blocks (markup duplication)."""
    out: list[Block] = []
    for b in blocks:
        if out and out[-1].text == b.text and out[-1].kind == b.kind:
            continue
        out.append(b)
    return out
