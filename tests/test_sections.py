"""T4 — Item segmentation, with ToC echoes disambiguated (PRD §S1.2.1)."""

from ragauge.contracts import Section
from ragauge.ingest.parse import Block
from ragauge.ingest.sections import segment_sections


def H(text, is_link=False):
    return Block(text=text, kind="heading", is_link=is_link, looks_like_heading=True)


def P(text):
    return Block(text=text, kind="prose")


def test_toc_echo_is_not_a_section_start():
    blocks = [
        # Table-of-contents block: linked Item markers near the top.
        H("Item 1. Business", is_link=True),
        H("Item 1A. Risk Factors", is_link=True),
        # Cover prose.
        P("This annual report covers fiscal 2023."),
        # Real body headings (not links).
        H("Item 1. Business"),
        P("The Company designs and sells smartphones."),
        H("Item 1A. Risk Factors"),
        P("The Company is exposed to supply-chain risk."),
    ]
    spans = segment_sections(blocks)
    by_section = {s.section: s for s in spans if s.section is not Section.OTHER}

    assert Section.ITEM_1 in by_section
    assert Section.ITEM_1A in by_section
    assert "smartphones" in " ".join(b.text for b in by_section[Section.ITEM_1].blocks)
    assert "supply-chain" in " ".join(
        b.text for b in by_section[Section.ITEM_1A].blocks
    )


def test_unrecognized_items_collapse_to_other():
    blocks = [
        H("Item 1. Business"),
        P("Business prose."),
        H("Item 2. Properties"),
        P("We lease offices."),
        H("Item 7. Management's Discussion and Analysis"),
        P("Revenue grew."),
    ]
    spans = segment_sections(blocks)
    sections = [s.section for s in spans]
    assert Section.ITEM_1 in sections
    assert Section.ITEM_7 in sections
    # Item 2 is not a target -> OTHER, and its prose stays with it.
    other = [s for s in spans if s.section is Section.OTHER]
    assert any("lease offices" in " ".join(b.text for b in s.blocks) for s in other)
