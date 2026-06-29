"""Eval harness entrypoint (T10 + T14–T16): golden set -> pipeline -> RunReport.

For each golden row it runs Retrieve -> Generate -> Judge, then computes:

* **Retrieval (no LLM):** recall@5, MRR, unanswerable-precision — always computed,
  reproducible from ``(config hash, corpus hash)``.
* **Generation (LLM-as-judge):** groundedness / supported-claim rate and
  unsupported-claim rate, from the structured judge verdicts.
* **Ops:** ``$/run`` from real provider token counts, p50/p95 latency.

Results are written to ``metrics.json`` (a ``RunReport``) and printed as a table.

The judge/generator are optional: with ``--no-judge``, no ``anthropic`` package, or
no ``ANTHROPIC_API_KEY``, the harness still produces the deterministic retrieval
baseline (DESIGN.md §7.3 — the metrics aren't hostage to the judge). Run with::

    python -m ragauge.eval.run            # full judged run (needs ANTHROPIC_API_KEY)
    python -m ragauge.eval.run --no-judge # deterministic retrieval baseline only
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from ragauge.config import PipelineConfig
from ragauge.contracts import GoldType, RunReport
from ragauge.eval import metrics
from ragauge.eval.cost import assert_judge_at_least_as_capable
from ragauge.eval.golden import DEFAULT_GOLDEN, load_golden
from ragauge.eval.judge import JUDGE_PROMPT_VERSION, Judge
from ragauge.generate.generator import GENERATOR_PROMPT_VERSION, Generator

DATA = Path("data")
MANIFEST = DATA / "manifest.json"
STORE = DATA / "chunks.jsonl"
INDEX_DIR = Path("indexes/dense")
DEFAULT_OUT = Path("metrics.json")

DEFAULT_GENERATOR_MODEL = "claude-opus-4-8"
DEFAULT_JUDGE_MODEL = "claude-opus-4-8"  # judge >= generator (PRD §7.3)


def _judging_available(no_judge: bool) -> tuple[bool, str]:
    if no_judge:
        return False, "--no-judge"
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, "anthropic package not installed"
    if not os.getenv("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY not set"
    return True, ""


def run(
    *,
    golden_path: Path,
    out_path: Path,
    generator_model: str,
    judge_model: str,
    no_judge: bool,
    top_k: int,
) -> RunReport:
    from ragauge.retrieve.embedder import BgeEmbedder
    from ragauge.retrieve.retriever import build_retriever

    cfg = PipelineConfig()
    cfg.retrieval.top_k = top_k
    corpus_hash = json.loads(MANIFEST.read_text())["corpus_hash"]

    golden = load_golden(golden_path)
    embedder = BgeEmbedder(cfg.embedding)
    retriever = build_retriever(
        embedder, index_dir=INDEX_DIR, store_path=STORE, config=cfg
    )
    chunk_map = retriever.chunk_map

    judge_on, judge_off_reason = _judging_available(no_judge)
    generator = judge = None
    if judge_on:
        assert_judge_at_least_as_capable(generator_model, judge_model)
        generator = Generator(generator_model)
        judge = Judge(judge_model)
        print(f"Judged run: generator={generator_model}  judge={judge_model}")
    else:
        print(f"Retrieval-only run (no generation/judge): {judge_off_reason}")

    per_question: list[dict] = []
    retrieval_ms: list[float] = []
    gen_ms: list[float] = []
    total_cost = 0.0

    for row in golden:
        answerable = row.type != GoldType.UNANSWERABLE

        t0 = time.perf_counter()
        results = retriever.retrieve(row.question, cfg.retrieval)
        dt = (time.perf_counter() - t0) * 1000
        retrieval_ms.append(dt)
        retrieved_ids = [r.chunk.chunk_id for r in results]

        rec = recip = None
        if answerable:
            rec = metrics.recall_at_k(row.gold_chunk_ids, retrieved_ids, top_k)
            recip = metrics.reciprocal_rank(row.gold_chunk_ids, retrieved_ids)

        entry: dict = {
            "id": row.id,
            "type": row.type.value,
            "difficulty": row.difficulty,
            "gold_chunk_ids": row.gold_chunk_ids,
            "retrieved_ids": retrieved_ids,
            "recall_at_k": rec,
            "reciprocal_rank": recip,
            "retrieval_ms": dt,
        }

        if judge_on:
            answer = generator.generate(row.question, results, cfg.retrieval)
            total_cost += answer.cost_usd
            gen_ms.append(answer.latency_ms)
            entry.update(
                {
                    "abstained": answer.abstained,
                    "answer": answer.text,
                    "citations": answer.citations,
                    "gen_cost_usd": answer.cost_usd,
                }
            )
            if not answer.abstained:
                verdict, tel = judge.judge(row.question, answer, results)
                total_cost += tel["cost_usd"]
                entry.update(
                    {
                        "supported": verdict.supported,
                        "unsupported_claims": verdict.unsupported_claims,
                        "groundedness_score": verdict.score,
                        "judge_cost_usd": tel["cost_usd"],
                    }
                )
        else:
            # Retrieval-only: abstention is the pre-generation evidence gate.
            min_score = cfg.retrieval.min_score
            entry["abstained"] = not results or (
                min_score is not None and results[0].score < min_score
            )

        per_question.append(entry)

    aggregates = _aggregate(per_question, golden, retrieval_ms, gen_ms, judge_on)

    report = RunReport(
        config_hash=cfg.hash(),
        corpus_hash=corpus_hash,
        embedding_model_id=cfg.embedding.model_id,
        generator_model_id=generator_model if judge_on else "",
        judge_model_id=judge_model if judge_on else "",
        timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        per_question=per_question,
        aggregates=aggregates,
        cost_usd=round(total_cost, 6),
    )
    out_path.write_text(report.model_dump_json(indent=2))
    _print_summary(report, judge_on, golden_path, out_path)
    return report


def _aggregate(per_question, golden, retrieval_ms, gen_ms, judge_on) -> dict:
    answerable = [e for e in per_question if e["recall_at_k"] is not None]
    recalls = [e["recall_at_k"] for e in answerable]
    rrs = [e["reciprocal_rank"] for e in answerable]

    abstained = [bool(e.get("abstained")) for e in per_question]
    is_unans = [e["type"] == "unanswerable" for e in per_question]
    uprec, n_abstained, n_correct = metrics.unanswerable_precision(abstained, is_unans)

    agg: dict = {
        "n_questions": len(per_question),
        "n_answerable": len(answerable),
        "recall_at_5": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "mrr": round(sum(rrs) / len(rrs), 4) if rrs else None,
        "unanswerable_precision": None if uprec is None else round(uprec, 4),
        "unanswerable_recall": metrics.unanswerable_recall(abstained, is_unans),
        "n_abstained": n_abstained,
        "n_correct_abstentions": n_correct,
        "retrieval_latency_ms": metrics.percentiles(retrieval_ms),
    }

    if judge_on:
        judged = [e for e in per_question if "supported" in e]
        n_answered = sum(1 for e in per_question if e.get("abstained") is False)
        supported = [e for e in judged if e["supported"]]
        with_unsupported = [e for e in judged if e["unsupported_claims"]]
        scores = [e["groundedness_score"] for e in judged]
        agg.update(
            {
                "n_answered": n_answered,
                "n_judged": len(judged),
                "groundedness_supported_rate": round(len(supported) / len(judged), 4)
                if judged
                else None,
                "unsupported_claim_rate": round(len(with_unsupported) / len(judged), 4)
                if judged
                else None,
                "mean_groundedness_score": round(sum(scores) / len(scores), 4)
                if scores
                else None,
                "generation_latency_ms": metrics.percentiles(gen_ms),
                "generator_prompt_version": GENERATOR_PROMPT_VERSION,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
            }
        )
    return agg


def _fmt(v) -> str:
    return "  n/a" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)


def _print_summary(report: RunReport, judge_on: bool, golden_path, out_path) -> None:
    a = report.aggregates
    print()
    print("=" * 60)
    print(f"  RAGauge eval — {golden_path}")
    print(f"  config={report.config_hash}  corpus={report.corpus_hash}")
    print("=" * 60)
    rows = [
        ("questions", a["n_questions"]),
        ("answerable", a["n_answerable"]),
        ("recall@5", a["recall_at_5"]),
        ("MRR", a["mrr"]),
        ("unanswerable-precision", a["unanswerable_precision"]),
        (f"  (abstained {a['n_abstained']}, correct {a['n_correct_abstentions']})", ""),
        ("retrieval p50/p95 ms",
         f"{a['retrieval_latency_ms']['p50']:.0f} / {a['retrieval_latency_ms']['p95']:.0f}"),
    ]
    if judge_on:
        rows += [
            ("answered", a["n_answered"]),
            ("groundedness / supported-rate", a["groundedness_supported_rate"]),
            ("unsupported-claim rate", a["unsupported_claim_rate"]),
            ("mean groundedness score", a["mean_groundedness_score"]),
            ("generation p50/p95 ms",
             f"{a['generation_latency_ms']['p50']:.0f} / {a['generation_latency_ms']['p95']:.0f}"),
        ]
    rows.append(("$/run", f"${report.cost_usd:.4f}"))
    for label, val in rows:
        print(f"  {label:<34}{_fmt(val)}")
    print("=" * 60)
    print(f"  wrote {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragauge-eval", description=__doc__)
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN,
                   help="golden-set JSONL (default: the committed candidate set)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="metrics.json path")
    p.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--no-judge", action="store_true",
                   help="deterministic retrieval metrics only (no LLM calls)")
    p.add_argument("-k", "--top-k", type=int, default=5)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    run(
        golden_path=args.golden,
        out_path=args.out,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        no_judge=args.no_judge,
        top_k=args.top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
