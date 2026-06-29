"""Golden-set verifier — structural checks (T9).

Pure functions over a row + chunk map, no network, so they run in the
deterministic suite. These pin the rules a malformed golden row must trip.
"""

from __future__ import annotations

from ragauge.contracts import Chunk, GoldRow, GoldType, Section
from ragauge.eval.verify import _distribution, structural_issues


def _chunk(chunk_id: str) -> Chunk:
    doc_id = chunk_id.split(":", 1)[0]
    return Chunk(
        doc_id=doc_id, company=doc_id.split("-")[0], fiscal_year=2025,
        section=Section.ITEM_8, anchor="", chunk_id=chunk_id, text="evidence",
    )


def _map(*chunk_ids: str) -> dict:
    return {cid: _chunk(cid) for cid in chunk_ids}


def _row(**kw) -> GoldRow:
    base = dict(
        id="q001", question="q?", gold_answer="a", gold_chunk_ids=[],
        type=GoldType.SINGLE_DOC, difficulty="easy",
    )
    base.update(kw)
    return GoldRow(**base)


AAPL = "AAPL-10K-FY2025:ITEM_8:aaa"
MSFT = "MSFT-10K-FY2025:ITEM_8:bbb"


def test_clean_answerable_row_has_no_issues():
    row = _row(gold_chunk_ids=[AAPL])
    assert structural_issues(row, _map(AAPL)) == []


def test_missing_chunk_id_flagged():
    row = _row(gold_chunk_ids=[AAPL])
    issues = structural_issues(row, _map())  # empty corpus
    assert any("not in corpus" in i for i in issues)


def test_answerable_without_chunks_flagged():
    row = _row(gold_chunk_ids=[])
    assert any("cites no gold chunks" in i for i in structural_issues(row, _map()))


def test_answerable_empty_answer_flagged():
    row = _row(gold_chunk_ids=[AAPL], gold_answer="   ")
    assert any("empty gold_answer" in i for i in structural_issues(row, _map(AAPL)))


def test_unanswerable_clean_when_no_chunks():
    row = _row(type=GoldType.UNANSWERABLE, gold_chunk_ids=[],
               gold_answer="Not disclosed in the filing.")
    assert structural_issues(row, _map()) == []


def test_unanswerable_with_chunks_flagged():
    row = _row(type=GoldType.UNANSWERABLE, gold_chunk_ids=[AAPL])
    assert any("should cite no chunks" in i for i in structural_issues(row, _map(AAPL)))


def test_multi_hop_single_filing_flagged():
    row = _row(type=GoldType.MULTI_HOP, gold_chunk_ids=[AAPL])
    assert any("spans only 1 filing" in i for i in structural_issues(row, _map(AAPL)))


def test_multi_hop_two_filings_clean():
    row = _row(type=GoldType.MULTI_HOP, gold_chunk_ids=[AAPL, MSFT])
    assert structural_issues(row, _map(AAPL, MSFT)) == []


def test_unknown_difficulty_flagged():
    row = _row(gold_chunk_ids=[AAPL], difficulty="trivial")
    assert any("unknown difficulty" in i for i in structural_issues(row, _map(AAPL)))


def test_distribution_counts_by_type_and_company():
    rows = [
        _row(id="q1", gold_chunk_ids=[AAPL]),
        _row(id="q2", type=GoldType.MULTI_HOP, gold_chunk_ids=[AAPL, MSFT]),
        _row(id="q3", type=GoldType.UNANSWERABLE, gold_chunk_ids=[]),
    ]
    dist = _distribution(rows)
    assert dist["type"] == {"single_doc": 1, "multi_hop": 1, "unanswerable": 1}
    assert dist["company_gold_chunks"] == {"AAPL": 2, "MSFT": 1}
