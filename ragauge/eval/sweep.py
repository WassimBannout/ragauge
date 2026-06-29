"""Model & cost sweep (PRD T21, §7.2): pick the generator by data, not guess.

Runs the **fixed** golden set through three Claude tiers — cheapest, balanced,
most capable — holding retrieval and the judge constant, so the *only* variable is
the generator model. For each tier it reports the things that actually trade off
with the model choice:

* **groundedness** (supported-claim rate) and unsupported-claim rate,
* **cost-per-query** and ``$/run`` from real provider token counts,
* **p50/p95 generation latency**,
* **recall@5** — included to show it is *invariant* to the model: recall is owned
  by the retrieval stack (see :mod:`ragauge.eval.ablation` for recall lift per
  stage), so a flat recall column is itself the signal that this sweep moves
  groundedness/cost/latency, not retrieval.

It also measures **prompt caching**: the stable instruction prefix carries a
``cache_control`` breakpoint while the per-question retrieved chunks are sent
*after* it (uncached). ``$/run`` is reported with caching **on vs. off** by
repricing the same provider token counts — no second API pass. (At our short
instruction prefix the realized saving may be ~0 because the prefix is below the
model's minimum cacheable size; the dashboard says so honestly rather than
faking a win.)

Output is a Markdown dashboard (``dashboard.md``) plus a JSON sidecar. Run with::

    python -m ragauge.eval.sweep                      # full 3-model judged sweep
    python -m ragauge.eval.sweep --limit 3            # cheap smoke run
    python -m ragauge.eval.sweep --models claude-haiku-4-5,claude-opus-4-8
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from ragauge.config import PipelineConfig
from ragauge.contracts import GoldType
from ragauge.eval import metrics
from ragauge.eval.cost import (
    PRICING,
    assert_judge_at_least_as_capable,
    cost_usd_uncached,
)
from ragauge.eval.golden import DEFAULT_GOLDEN, load_golden
from ragauge.eval.judge import Judge
from ragauge.eval.run import (
    INDEX_DIR,
    MANIFEST,
    STORE,
    make_pipeline_config,
    _judging_available,
)
from ragauge.generate.generator import Generator

DEFAULT_OUT = Path("dashboard.md")
DEFAULT_JSON_OUT = Path("metrics_sweep.json")

# Cheapest → balanced → most capable. The judge is held at the most-capable tier
# so it is >= every generator under test (PRD §7.3 capability gate).
DEFAULT_MODELS: list[str] = [
    "claude-haiku-4-5",  # cheapest
    "claude-sonnet-4-6",  # balanced
    "claude-opus-4-8",  # most capable
]
DEFAULT_JUDGE_MODEL = "claude-opus-4-8"

# Labelled by price, derived from PRICING so the tier name never drifts from $.
_TIER_NAMES = ["cheapest", "balanced", "most capable", "premium"]


def _tier_labels(models: list[str]) -> dict[str, str]:
    """Order models by input price and name them cheapest..most-capable."""
    ordered = sorted(models, key=lambda m: PRICING[m][0])
    names = _TIER_NAMES if len(ordered) <= len(_TIER_NAMES) else None
    return {
        m: (names[i] if names else f"tier {i + 1}") for i, m in enumerate(ordered)
    }


@dataclass
class _ModelAcc:
    """Per-model running totals over the golden set."""

    model: str
    tier: str
    gen_ms: list[float] = field(default_factory=list)
    supported: list[bool] = field(default_factory=list)
    unsupported: list[bool] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    cost_on: float = 0.0
    cost_off: float = 0.0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    n_answered: int = 0
    n_abstained: int = 0


def _build_retriever():
    """Embedder + retriever, built once (preset-agnostic; embedding is held
    constant across the sweep)."""
    from ragauge.retrieve.embedder import BgeEmbedder
    from ragauge.retrieve.retriever import build_retriever

    base = PipelineConfig()
    embedder = BgeEmbedder(base.embedding)
    return build_retriever(
        embedder, index_dir=INDEX_DIR, store_path=STORE, config=base
    )


def run_sweep(
    *,
    golden_path: Path,
    out_path: Path,
    json_out_path: Path,
    models: list[str],
    judge_model: str,
    retrieval: str,
    top_k: int,
    limit: int | None,
    cache: bool,
) -> dict:
    judge_on, reason = _judging_available(False)
    if not judge_on:
        raise SystemExit(
            f"model sweep needs the generator + judge ({reason}); "
            "set ANTHROPIC_API_KEY and install `anthropic`."
        )
    for m in models:
        assert_judge_at_least_as_capable(m, judge_model)

    corpus_hash = json.loads(MANIFEST.read_text())["corpus_hash"]
    golden = load_golden(golden_path)
    if limit is not None:
        golden = golden[:limit]

    pipeline_cfg = make_pipeline_config(retrieval, top_k)
    retrieval_cfg = pipeline_cfg.retrieval

    retriever = _build_retriever()
    judge = Judge(judge_model, cache_system=cache)
    generators = {m: Generator(m, cache_system=cache) for m in models}
    tiers = _tier_labels(models)
    accs = {m: _ModelAcc(model=m, tier=tiers[m]) for m in models}

    print(
        f"Model sweep over {len(golden)} golden rows · retrieval={retrieval} · "
        f"judge={judge_model} · caching={'on' if cache else 'off'}\n"
        f"  models: {', '.join(f'{m} [{tiers[m]}]' for m in models)}\n"
    )

    # Retrieval is model-independent — retrieve once per question and reuse the
    # identical evidence across every model, so the comparison is strictly fair
    # and recall is constant by construction.
    recalls: list[float] = []
    rrs: list[float] = []
    for row in golden:
        answerable = row.type != GoldType.UNANSWERABLE
        results = retriever.retrieve(row.question, retrieval_cfg)
        retrieved_ids = [r.chunk.chunk_id for r in results]
        if answerable:
            recalls.append(metrics.recall_at_k(row.gold_chunk_ids, retrieved_ids, top_k))
            rrs.append(metrics.reciprocal_rank(row.gold_chunk_ids, retrieved_ids))

        for m in models:
            acc = accs[m]
            answer = generators[m].generate(row.question, results, retrieval_cfg)
            acc.gen_ms.append(answer.latency_ms)
            acc.cost_on += answer.cost_usd
            acc.cost_off += cost_usd_uncached(
                m,
                answer.input_tokens,
                answer.output_tokens,
                cache_creation_tokens=answer.cache_creation_input_tokens,
                cache_read_tokens=answer.cache_read_input_tokens,
            )
            acc.cache_write_tokens += answer.cache_creation_input_tokens
            acc.cache_read_tokens += answer.cache_read_input_tokens

            if answer.abstained:
                acc.n_abstained += 1
                continue
            acc.n_answered += 1

            verdict, tel = judge.judge(row.question, answer, results)
            acc.cost_on += tel["cost_usd"]
            acc.cost_off += cost_usd_uncached(
                judge_model,
                tel["input_tokens"],
                tel["output_tokens"],
                cache_creation_tokens=tel["cache_creation_input_tokens"],
                cache_read_tokens=tel["cache_read_input_tokens"],
            )
            acc.cache_write_tokens += tel["cache_creation_input_tokens"]
            acc.cache_read_tokens += tel["cache_read_input_tokens"]
            acc.supported.append(verdict.supported)
            acc.unsupported.append(bool(verdict.unsupported_claims))
            acc.scores.append(verdict.score)

    n_q = len(golden)
    shared = {
        "recall_at_5": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "mrr": round(sum(rrs) / len(rrs), 4) if rrs else None,
    }
    rows = [_summarize(acc, n_q, shared) for acc in accs.values()]
    rows.sort(key=lambda r: PRICING[r["model"]][0])  # cheapest first

    report = {
        "corpus_hash": corpus_hash,
        "config_hash": pipeline_cfg.hash(),
        "retrieval": retrieval,
        "judge_model_id": judge_model,
        "caching": cache,
        "n_questions": n_q,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "models": rows,
    }
    dashboard = _render_dashboard(report, golden_path)
    report["dashboard_markdown"] = dashboard

    out_path.write_text(dashboard)
    json_out_path.write_text(json.dumps(report, indent=2))
    print(dashboard)
    print(f"\nwrote {out_path} and {json_out_path}")
    return report


def _summarize(acc: _ModelAcc, n_questions: int, shared: dict) -> dict:
    judged = len(acc.supported)
    return {
        "model": acc.model,
        "tier": acc.tier,
        "recall_at_5": shared["recall_at_5"],  # retrieval-owned; same across models
        "mrr": shared["mrr"],
        "n_answered": acc.n_answered,
        "n_abstained": acc.n_abstained,
        "groundedness_supported_rate": round(sum(acc.supported) / judged, 4)
        if judged
        else None,
        "unsupported_claim_rate": round(sum(acc.unsupported) / judged, 4)
        if judged
        else None,
        "mean_groundedness_score": round(sum(acc.scores) / len(acc.scores), 4)
        if acc.scores
        else None,
        "generation_latency_ms": metrics.percentiles(acc.gen_ms),
        "cost_run_usd": round(acc.cost_on, 6),
        "cost_run_uncached_usd": round(acc.cost_off, 6),
        "cost_per_query_usd": round(acc.cost_on / n_questions, 6) if n_questions else 0.0,
        "cache_read_tokens": acc.cache_read_tokens,
        "cache_write_tokens": acc.cache_write_tokens,
    }


# --------------------------------------------------------------------------- #
# Dashboard rendering
# --------------------------------------------------------------------------- #
def _f(v, nd: int = 4) -> str:
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(r) + " |" for r in rows),
        ]
    )


def _render_dashboard(report: dict, golden_path: Path) -> str:
    rows = report["models"]
    quality = _md_table(
        ["tier", "model", "recall@5", "MRR", "grounded-rate", "unsupp-rate",
         "mean score", "gen p50/p95 ms", "$/query", "$/run"],
        [
            [
                r["tier"], r["model"], _f(r["recall_at_5"]), _f(r["mrr"]),
                _f(r["groundedness_supported_rate"]), _f(r["unsupported_claim_rate"]),
                _f(r["mean_groundedness_score"]),
                f"{r['generation_latency_ms']['p50']:.0f} / {r['generation_latency_ms']['p95']:.0f}",
                f"${r['cost_per_query_usd']:.4f}", f"${r['cost_run_usd']:.4f}",
            ]
            for r in rows
        ],
    )

    caching = _md_table(
        ["model", "$/run cache-off", "$/run cache-on", "saving", "cache-read tok", "cache-write tok"],
        [
            [
                r["model"],
                f"${r['cost_run_uncached_usd']:.4f}",
                f"${r['cost_run_usd']:.4f}",
                _saving(r["cost_run_uncached_usd"], r["cost_run_usd"]),
                str(r["cache_read_tokens"]),
                str(r["cache_write_tokens"]),
            ]
            for r in rows
        ],
    )

    realized = sum(r["cache_read_tokens"] + r["cache_write_tokens"] for r in rows)
    if not report["caching"]:
        cache_note = "_Caching disabled (`--no-cache`); the on/off columns are equal by definition._"
    elif realized == 0:
        cache_note = (
            "_The provider reported **zero** cached tokens: the stable instruction "
            "prefix is below the model's minimum cacheable size (1024–4096 tokens), "
            "so it silently does not cache and the realized saving is $0. The "
            "breakpoint is placed correctly (instructions cached, per-question "
            "evidence after it) — caching would pay off here only with a larger "
            "shared prefix (e.g. a big few-shot block or a shared document)._"
        )
    else:
        cache_note = (
            "_Cost is repriced from the provider's own token counts (cache write "
            "1.25×, read 0.10× the input rate) — the same call, two price models._"
        )

    rec = _recommendation(rows)
    a = report
    return "\n".join(
        [
            "# RAGauge — model & cost sweep",
            "",
            f"Generator picked by data, not guess (PRD T21). Golden set `{golden_path}` "
            f"({a['n_questions']} questions), retrieval held at `{a['retrieval']}`, "
            f"judge fixed at `{a['judge_model_id']}` (≥ every generator under test).",
            "",
            f"- corpus `{a['corpus_hash']}` · config `{a['config_hash']}` · "
            f"caching `{'on' if a['caching'] else 'off'}` · {a['timestamp']}",
            f"- cost from the provider's billed token `usage` (never a generic "
            f"tokenizer); latency is wall-clock per generation call.",
            "",
            "## Quality vs. cost vs. latency",
            "",
            quality,
            "",
            f"**Recommendation.** {rec}",
            "",
            "## Prompt caching — $/run on vs. off",
            "",
            caching,
            "",
            cache_note,
            "",
            "## Note on recall",
            "",
            "`recall@5` is **identical across models** — it is a property of the "
            "retrieval stack, not the generator, so the model sweep cannot move it. "
            "Recall *lift* is the headline of the retrieval ablation "
            "(`metrics_ablation.json` / `python -m ragauge.eval.ablation`); it is "
            "surfaced here only to make the separation of concerns explicit.",
            "",
        ]
    )


def _saving(off: float, on: float) -> str:
    if off <= 0:
        return "n/a"
    pct = (off - on) / off * 100
    return f"${off - on:.4f} ({pct:.1f}%)"


def _recommendation(rows: list[dict]) -> str:
    graded = [r for r in rows if r["groundedness_supported_rate"] is not None]
    if not graded:
        return "No judged rows — run with an API key to compare groundedness."
    best = max(graded, key=lambda r: r["groundedness_supported_rate"])
    best_rate = best["groundedness_supported_rate"]
    cheapest = min(graded, key=lambda r: r["cost_run_usd"])
    # Cheapest model whose groundedness is within 5 points of the best.
    within = sorted(
        (r for r in graded if best_rate - r["groundedness_supported_rate"] <= 0.05),
        key=lambda r: r["cost_run_usd"],
    )
    pick = within[0] if within else best

    if pick["model"] == cheapest["model"]:
        # Cheapest model is within tolerance of the best — the clear pick.
        rate = pick["groundedness_supported_rate"]
        lead = (
            "top groundedness *and* lowest cost"
            if pick["model"] == best["model"]
            else f"groundedness within 5 points of the best (`{best['model']}`, {_f(best_rate)})"
        )
        return (
            f"`{pick['model']}` ({pick['tier']}) — {lead} "
            f"({_f(rate)} at ${pick['cost_per_query_usd']:.4f}/query). Clear pick."
        )
    if pick["model"] == best["model"]:
        # Best groundedness, but it costs more than the cheapest — worth it.
        return (
            f"`{pick['model']}` ({pick['tier']}) leads on groundedness "
            f"({_f(best_rate)}) at ${pick['cost_per_query_usd']:.4f}/query vs "
            f"${cheapest['cost_per_query_usd']:.4f} for `{cheapest['model']}` — the "
            f"groundedness gain justifies the higher cost."
        )
    return (
        f"`{pick['model']}` ({pick['tier']}) matches the best groundedness within "
        f"5 points ({_f(pick['groundedness_supported_rate'])} vs "
        f"{_f(best_rate)} for `{best['model']}`) at "
        f"${pick['cost_per_query_usd']:.4f}/query vs "
        f"${best['cost_per_query_usd']:.4f} — the cheaper model is the better buy."
    )


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragauge-sweep", description=__doc__)
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="dashboard.md path")
    p.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS),
                   help="comma-separated generator model ids (cheapest..most capable)")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                   help="fixed grader; must be >= every generator (PRD §7.3)")
    p.add_argument("--retrieval", default="dense",
                   help="retrieval preset held constant across the sweep")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.add_argument("--limit", type=int, default=None,
                   help="cap golden rows (cheap smoke run)")
    p.add_argument("--no-cache", action="store_true",
                   help="disable prompt caching (on/off columns become equal)")
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    run_sweep(
        golden_path=args.golden,
        out_path=args.out,
        json_out_path=args.json_out,
        models=models,
        judge_model=args.judge_model,
        retrieval=args.retrieval,
        top_k=args.top_k,
        limit=args.limit,
        cache=not args.no_cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
