"""Golden-set verifier — make the ground truth trustworthy (T9).

DESIGN.md §8: *"a golden set the author didn't check is worthless."* Final sign-off
is human, but this tool does the mechanical heavy lifting so the human reviews only
the rows that fail a check — not 30 rows cold.

Two layers:

* **Structural checks (deterministic, no LLM).** Every `gold_chunk_id` resolves in
  the corpus; answerable rows cite ≥1 chunk and have a non-empty answer;
  unanswerable rows cite **none**; multi-hop rows span ≥2 filings; difficulty is a
  known value. These are bugs — a failure exits non-zero.

* **Self-consistency check (opt-in `--judge`).** Reuses the eval harness's own
  `Judge` the other way around: it grades each gold **answer** against the chunks
  that row **cites**. If the judge says the gold answer isn't supported by its own
  cited evidence, the row is suspect — the answer is wrong, or it points at the
  wrong chunks. These are flagged for human review, not auto-fixed: rewriting the
  ground truth would defeat the verification.

    python -m ragauge.eval.verify              # structural checks (free)
    python -m ragauge.eval.verify --judge      # + LLM self-consistency (needs a key)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from ragauge.contracts import Answer, GoldType, RetrievedChunk
from ragauge.eval.golden import DEFAULT_GOLDEN, load_golden
from ragauge.ingest.store import load_chunk_map

STORE = Path("data/chunks.jsonl")
VALID_DIFFICULTY = {"easy", "medium", "hard"}
# Below this judge score, a gold answer is treated as not grounded in its chunks.
GROUNDING_THRESHOLD = 0.5


def _doc_of(chunk_id: str) -> str:
    """`AAPL-10K-FY2025:ITEM_8:abc` -> `AAPL-10K-FY2025` (doc id has no colons)."""
    return chunk_id.split(":", 1)[0]


def structural_issues(row, chunk_map: dict) -> list[str]:
    """Hard, deterministic problems with one golden row (empty == clean)."""
    issues: list[str] = []
    answerable = row.type != GoldType.UNANSWERABLE

    missing = [cid for cid in row.gold_chunk_ids if cid not in chunk_map]
    if missing:
        issues.append(f"gold_chunk_id(s) not in corpus: {missing}")

    if answerable:
        if not row.gold_chunk_ids:
            issues.append("answerable row cites no gold chunks")
        if not row.gold_answer.strip():
            issues.append("answerable row has an empty gold_answer")
    else:
        if row.gold_chunk_ids:
            issues.append(
                f"unanswerable row should cite no chunks, has {len(row.gold_chunk_ids)}"
            )

    if row.type == GoldType.MULTI_HOP:
        docs = {_doc_of(cid) for cid in row.gold_chunk_ids}
        if len(docs) < 2:
            issues.append(
                f"multi_hop row spans only {len(docs)} filing(s); expected ≥2"
            )

    if row.difficulty not in VALID_DIFFICULTY:
        issues.append(f"unknown difficulty {row.difficulty!r}")

    return issues


def _as_evidence(row, chunk_map: dict) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=chunk_map[cid], score=1.0)
        for cid in row.gold_chunk_ids
        if cid in chunk_map
    ]


def verify(
    *,
    golden_path: Path,
    store_path: Path,
    use_judge: bool,
    judge_model: str,
    out_path: Path | None,
    strict: bool,
) -> bool:
    golden = load_golden(golden_path)
    chunk_map = load_chunk_map(store_path)

    judge = None
    if use_judge:
        from ragauge.eval.judge import Judge

        judge = Judge(judge_model)

    rows: list[dict] = []
    total_cost = 0.0
    for row in golden:
        issues = structural_issues(row, chunk_map)
        entry: dict = {
            "id": row.id,
            "type": row.type.value,
            "difficulty": row.difficulty,
            "structural_issues": issues,
            "grounding": None,
        }

        # Self-consistency: grade the gold answer against its own cited chunks.
        answerable = row.type != GoldType.UNANSWERABLE
        if judge is not None and answerable and not issues:
            evidence = _as_evidence(row, chunk_map)
            verdict, tel = judge.judge(
                row.question, Answer(text=row.gold_answer), evidence
            )
            total_cost += tel["cost_usd"]
            grounded = verdict.supported and verdict.score >= GROUNDING_THRESHOLD
            entry["grounding"] = {
                "supported": verdict.supported,
                "score": verdict.score,
                "grounded": grounded,
                "unsupported_claims": verdict.unsupported_claims,
            }
        rows.append(entry)

    ok_structural = all(not r["structural_issues"] for r in rows)
    flagged = [
        r for r in rows if r["grounding"] is not None and not r["grounding"]["grounded"]
    ]
    report = {
        "golden_path": str(golden_path),
        "corpus_chunks": len(chunk_map),
        "n_rows": len(golden),
        "judged": use_judge,
        "judge_model": judge_model if use_judge else None,
        "cost_usd": round(total_cost, 6),
        "distribution": _distribution(golden),
        "rows": rows,
    }
    md = _render(report, golden, flagged, ok_structural)
    print(md)
    if out_path is not None:
        report["report_markdown"] = md
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out_path}")

    # Structural failures are bugs -> non-zero. Grounding flags are warnings for
    # the human, unless --strict promotes them to failures.
    return ok_structural and (not strict or not flagged)


def _distribution(golden) -> dict:
    by_type: dict[str, int] = {}
    by_diff: dict[str, int] = {}
    by_company: dict[str, int] = {}
    for row in golden:
        by_type[row.type.value] = by_type.get(row.type.value, 0) + 1
        by_diff[row.difficulty] = by_diff.get(row.difficulty, 0) + 1
        for cid in row.gold_chunk_ids:
            comp = _doc_of(cid).split("-")[0]
            by_company[comp] = by_company.get(comp, 0) + 1
    return {"type": by_type, "difficulty": by_diff, "company_gold_chunks": by_company}


def _render(report: dict, golden, flagged: list[dict], ok_structural: bool) -> str:
    struct_errors = [r for r in report["rows"] if r["structural_issues"]]
    header = "✅ structurally clean" if ok_structural else "❌ structural errors"
    lines = [
        f"## Golden-set verification — {header}",
        "",
        f"`{report['golden_path']}` · {report['n_rows']} rows · "
        f"corpus {report['corpus_chunks']} chunks"
        + (f" · judge `{report['judge_model']}`" if report["judged"] else "")
        + ".",
        "",
        f"- distribution: {report['distribution']['type']} · "
        f"{report['distribution']['difficulty']}",
    ]

    if struct_errors:
        lines += ["", "### ❌ Structural errors (must fix)", ""]
        for r in struct_errors:
            for issue in r["structural_issues"]:
                lines.append(f"- **{r['id']}**: {issue}")
    else:
        lines += ["", "All rows pass the structural checks (ids resolve, "
                  "answerable/unanswerable cardinality, multi-hop spread, difficulty)."]

    if report["judged"]:
        graded = [r for r in report["rows"] if r["grounding"] is not None]
        lines += [
            "",
            f"### Self-consistency (gold answer vs. its own cited chunks) — "
            f"{len(graded) - len(flagged)}/{len(graded)} grounded",
            "",
        ]
        if flagged:
            lines.append(
                "These gold answers were **not** judged supported by the chunks "
                "the row cites — review them (the answer may be wrong, or it cites "
                "the wrong chunks):"
            )
            lines.append("")
            qmap = {row.id: row for row in golden}
            for r in flagged:
                g = r["grounding"]
                q = qmap[r["id"]].question
                lines.append(
                    f"- **{r['id']}** (score {g['score']:.2f}): {q}"
                )
                if g["unsupported_claims"]:
                    lines.append(f"  - judge flagged: {g['unsupported_claims']}")
            lines.append(
                "\n_Not auto-corrected — the ground truth is the human's to fix._"
            )
        else:
            lines.append(
                "Every answerable gold answer grounds in its cited chunks. ✅"
            )
        if report["cost_usd"]:
            lines.append(f"\n<sub>self-consistency judge cost: ${report['cost_usd']:.4f}</sub>")

    lines += ["", "> Final sign-off is still human (DESIGN.md §8); this narrows the "
              "read to flagged rows, it doesn't replace it.", ""]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragauge-verify", description=__doc__)
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument("--store", type=Path, default=STORE)
    p.add_argument("--judge", action="store_true",
                   help="also grade each gold answer against its cited chunks (LLM)")
    p.add_argument("--judge-model", default="claude-opus-4-8")
    p.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero on grounding flags too, not just structural errors")
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    ok = verify(
        golden_path=args.golden,
        store_path=args.store,
        use_judge=args.judge,
        judge_model=args.judge_model,
        out_path=args.out,
        strict=args.strict,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
