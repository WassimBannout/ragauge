# Golden set — candidate rows (T9)

`candidates.jsonl` holds **30 AI-drafted** question/answer rows for the eval set.
Schema matches `GoldRow` (`ragauge/contracts.py`): `id, question, gold_answer,
gold_chunk_ids, type, difficulty`.

> **Status: AUTOMATED VERIFICATION PASSED; human sign-off pending.** Drafted
> against `corpus_hash 60707081f218` (AAPL FY2025, MSFT FY2025, NVDA FY2026 — 789
> chunks); `gold_chunk_ids` are pinned to that corpus + chunking config — re-verify
> if either changes.
>
> `python -m ragauge.eval.verify --judge` (report: `evals/golden_verification.json`)
> reports the set **structurally clean** (all 30 rows: ids resolve,
> answerable/unanswerable cardinality, multi-hop spread ≥2 filings, valid
> difficulty) and **self-consistent** — all **25/25** answerable gold answers were
> judged *supported by the chunks they cite* (`claude-opus-4-8`, $0.23). For a
> 10-K fact the cited chunk **is** the source of truth, so grounded-in-evidence ==
> correct-per-filing. What automation can't do (DESIGN.md §8: "a golden set the
> author didn't check is worthless") still needs a human: question phrasing, and
> the judgment-call rows flagged below — confirm absence for the unanswerables
> (the verifier checks grounding, not absence) and the q024 framing.

## How these were grounded (not invented)
Each answer was traced to specific chunk text in `data/chunks.jsonl`. Every
`gold_chunk_id` was checked to (a) exist in the corpus and (b) actually contain
the stated fact (numbers grepped back to the source line). Unanswerable rows
were checked to confirm the fact is genuinely **absent** from all 789 chunks.

## Coverage

| Dimension | Spread |
|---|---|
| **type** | single_doc 20 · multi_hop 5 · unanswerable 5 (17%) |
| **difficulty** | easy 12 · medium 13 · hard 5 |
| **company (gold chunks)** | AAPL 13 · NVDA 11 · MSFT 10 |
| **section (gold chunks)** | ITEM_8 12 · ITEM_7 10 · ITEM_1 6 · OTHER 6 · ITEM_1A 1 |

Multi-hop rows (q021–q025) each fuse evidence across **2–3 filings**
(cross-company revenue / headcount / R&D / gross-margin / fiscal-year compares).

## Rows I'm unsure about — verify these first
- **q024** (gross-margin compare): NVIDIA prints "71.1%" directly; Apple's ~46.9%
  is **computed** ($195,201M gross margin ÷ $416,161M net sales) — Apple does not
  print a gross-margin percentage. Confirm the comparison framing is acceptable.
- **q027** (Tim Cook comp = unanswerable): rests on exec comp living in the proxy
  (incorporated by reference into Part III), not the 10-K body. "Tim Cook" does
  not appear in any chunk. Confirm this is the intended kind of unanswerable.
- **q029** (Azure $ = unanswerable): rests on Microsoft disclosing Azure *growth
  rates* but no absolute Azure dollar figure (verified: no "Azure … $" co-occurs).
  Adversarial — retrieval will surface Azure-heavy chunks; the answer isn't there.
- **q018** (NVIDIA customer concentration): the 22%/14% customers are **unnamed**
  in the filing (Customer A/B); gold answer must not name them.
- **q028** (Amazon revenue = unanswerable): "Amazon" appears only as a named
  competitor — adversarial retrieval bait, no figures present.

## Notes for the labeler
- Fiscal years differ across the three filings (AAPL Sep-2025, MSFT Jun-2025,
  NVDA Jan-2026); cross-company compares (q021–q025) flag this in the answer.
- All dollar figures are in **millions USD** as reported in the filings.
- Stratification gap: ITEM_1A (Risk Factors) is large but qualitative; only q003
  draws from it. Consider adding 1–2 more risk-factor rows when expanding.
