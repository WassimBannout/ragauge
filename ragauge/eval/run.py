"""Eval harness entrypoint (T10 + T14–T16): golden set -> pipeline -> RunReport.

For each golden row it runs Retrieve -> Generate -> Judge, then computes:

* **Retrieval (no LLM):** recall@5, MRR, unanswerable-precision — always computed,
  reproducible from ``(config hash, corpus hash)``.
* **Generation (LLM-as-judge):** groundedness / supported-claim rate and
  unsupported-claim rate, from the structured judge verdicts.
* **Ops:** ``$/run`` from real provider token counts, p50/p95 latency.

Results are written to ``metrics.json`` (a ``RunReport``) and printed as a table.

The retrieval stack is config-selectable via ``--retrieval`` so a single run can
target any rung of the ablation ladder (dense → hybrid → hybrid+rerank); the
3-config comparison table is driven by :mod:`ragauge.eval.ablation`.

The judge/generator are optional: with ``--no-judge``, no ``anthropic`` package, or
no ``ANTHROPIC_API_KEY``, the harness still produces the deterministic retrieval
baseline (DESIGN.md §7.3 — the metrics aren't hostage to the judge). Run with::

    python -m ragauge.eval.run                       # full judged dense run
    python -m ragauge.eval.run --retrieval hybrid+rerank --no-judge
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

# The ablation ladder: each preset is a config diff over the previous one, so a
# recall delta is attributable to the single stage that was switched on.
RETRIEVAL_PRESETS: dict[str, dict] = {
    "dense": {"dense": True, "bm25": False, "fusion": False, "rerank": False},
    "hybrid": {"dense": True, "bm25": True, "fusion": True, "rerank": False},
    "hybrid+rerank": {"dense": True, "bm25": True, "fusion": True, "rerank": True},
}


def make_pipeline_config(preset: str, top_k: int) -> PipelineConfig:
    """A ``PipelineConfig`` with the named retrieval preset applied. Embedding
    and chunking are held constant across presets, so the only config diff — and
    thus the only thing the ``config_hash`` separates — is the retrieval stack."""
    cfg = PipelineConfig()
    cfg.retrieval = cfg.retrieval.model_copy(
        update={**RETRIEVAL_PRESETS[preset], "top_k": top_k, "top_n": top_k}
    )
    return cfg


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


def build_components(no_judge: bool, generator_model: str, judge_model: str):
    """Load the embedder + retriever (and, when judging, the generator/judge)
    once. The retriever is preset-agnostic — embedding/chunking are constant — so
    the ablation reuses a single instance across all three configs."""
    from ragauge.retrieve.embedder import BgeEmbedder
    from ragauge.retrieve.retriever import build_retriever

    base = PipelineConfig()
    embedder = BgeEmbedder(base.embedding)
    retriever = build_retriever(
        embedder, index_dir=INDEX_DIR, store_path=STORE, config=base
    )

    judge_on, reason = _judging_available(no_judge)
    generator = judge = None
    if judge_on:
        assert_judge_at_least_as_capable(generator_model, judge_model)
        generator = Generator(generator_model)
        judge = Judge(judge_model)
    return retriever, generator, judge, judge_on, reason


def evaluate_config(
    *,
    retriever,
    generator,
    judge,
    judge_on: bool,
    golden,
    pipeline_cfg: PipelineConfig,
    corpus_hash: str,
    generator_model: str,
    judge_model: str,
) -> RunReport:
    """Run the full golden set through one retrieval config -> ``RunReport``."""
    retrieval_cfg = pipeline_cfg.retrieval
    top_k = retrieval_cfg.top_k

    per_question: list[dict] = []
    retrieval_ms: list[float] = []
    gen_ms: list[float] = []
    total_cost = 0.0

    for row in golden:
        answerable = row.type != GoldType.UNANSWERABLE

        t0 = time.perf_counter()
        results = retriever.retrieve(row.question, retrieval_cfg)
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
            answer = generator.generate(row.question, results, retrieval_cfg)
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
            min_score = retrieval_cfg.min_score
            entry["abstained"] = not results or (
                min_score is not None and results[0].score < min_score
            )

        per_question.append(entry)

    aggregates = _aggregate(per_question, golden, retrieval_ms, gen_ms, judge_on, top_k)

    return RunReport(
        config_hash=pipeline_cfg.hash(),
        corpus_hash=corpus_hash,
        embedding_model_id=pipeline_cfg.embedding.model_id,
        generator_model_id=generator_model if judge_on else "",
        judge_model_id=judge_model if judge_on else "",
        timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        per_question=per_question,
        aggregates=aggregates,
        cost_usd=round(total_cost, 6),
    )


def run(
    *,
    golden_path: Path,
    out_path: Path,
    generator_model: str,
    judge_model: str,
    no_judge: bool,
    top_k: int,
    retrieval: str = "dense",
    limit: int | None = None,
) -> RunReport:
    cfg = make_pipeline_config(retrieval, top_k)
    corpus_hash = json.loads(MANIFEST.read_text())["corpus_hash"]
    golden = load_golden(golden_path)
    if limit is not None:
        # Deterministic first-N subset: the golden set is stably ordered, so the
        # CI gate and the committed baseline grade the *same* questions.
        golden = golden[:limit]

    retriever, generator, judge, judge_on, reason = build_components(
        no_judge, generator_model, judge_model
    )
    if judge_on:
        print(f"Judged run [{retrieval}]: generator={generator_model}  judge={judge_model}")
    else:
        print(f"Retrieval-only run [{retrieval}] (no generation/judge): {reason}")

    report = evaluate_config(
        retriever=retriever,
        generator=generator,
        judge=judge,
        judge_on=judge_on,
        golden=golden,
        pipeline_cfg=cfg,
        corpus_hash=corpus_hash,
        generator_model=generator_model,
        judge_model=judge_model,
    )
    out_path.write_text(report.model_dump_json(indent=2))
    _print_summary(report, judge_on, golden_path, out_path, retrieval)
    return report


def _aggregate(per_question, golden, retrieval_ms, gen_ms, judge_on, top_k) -> dict:
    answerable = [e for e in per_question if e["recall_at_k"] is not None]
    recalls = [e["recall_at_k"] for e in answerable]
    rrs = [e["reciprocal_rank"] for e in answerable]

    abstained = [bool(e.get("abstained")) for e in per_question]
    is_unans = [e["type"] == "unanswerable" for e in per_question]
    uprec, n_abstained, n_correct = metrics.unanswerable_precision(abstained, is_unans)

    agg: dict = {
        "n_questions": len(per_question),
        "n_answerable": len(answerable),
        "top_k": top_k,
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


def _print_summary(report: RunReport, judge_on, golden_path, out_path, retrieval) -> None:
    a = report.aggregates
    print()
    print("=" * 60)
    print(f"  RAGauge eval — {golden_path}  [retrieval={retrieval}]")
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
    p.add_argument("--retrieval", choices=list(RETRIEVAL_PRESETS), default="dense",
                   help="retrieval stack preset (ablation rung)")
    p.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--no-judge", action="store_true",
                   help="deterministic retrieval metrics only (no LLM calls)")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.add_argument("--limit", type=int, default=None,
                   help="grade only the first N golden rows (CI subset)")
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
        retrieval=args.retrieval,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
