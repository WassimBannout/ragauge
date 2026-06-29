"""Retrieval ablation (PRD T20, DESIGN.md §5.2).

Runs the golden set through the three rungs of the retrieval ladder —
``dense`` → ``hybrid`` (BM25 + RRF) → ``hybrid+rerank`` (+ cross-encoder) — and
emits one comparison table: recall@5, MRR, groundedness, and p95 latency per
config. The point is **recall lift per stage**: a stage that doesn't move the
metrics gets cut, and saying so is itself the signal (DESIGN.md §5.3).

The embedder, retriever, and (when judging) generator/judge are built **once**
and reused across all three configs, so the only thing that varies between rows
is the retrieval stack. Run with::

    python -m ragauge.eval.ablation --no-judge   # deterministic recall ablation
    python -m ragauge.eval.ablation              # + groundedness (needs API key)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from ragauge.contracts import RunReport
from ragauge.eval.golden import DEFAULT_GOLDEN, load_golden
from ragauge.eval.run import (
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_JUDGE_MODEL,
    MANIFEST,
    RETRIEVAL_PRESETS,
    build_components,
    evaluate_config,
    make_pipeline_config,
)

DEFAULT_OUT = Path("metrics_ablation.json")


def run_ablation(
    *,
    golden_path: Path,
    out_path: Path,
    generator_model: str,
    judge_model: str,
    no_judge: bool,
    top_k: int,
) -> dict[str, RunReport]:
    corpus_hash = json.loads(MANIFEST.read_text())["corpus_hash"]
    golden = load_golden(golden_path)

    retriever, generator, judge, judge_on, reason = build_components(
        no_judge, generator_model, judge_model
    )
    mode = (
        f"judged (generator={generator_model}, judge={judge_model})"
        if judge_on
        else f"retrieval-only ({reason})"
    )
    print(f"Ablation over {len(golden)} golden rows — {mode}\n")

    reports: dict[str, RunReport] = {}
    for preset in RETRIEVAL_PRESETS:
        print(f"  running [{preset}] ...", flush=True)
        cfg = make_pipeline_config(preset, top_k)
        reports[preset] = evaluate_config(
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

    table = _comparison_table(reports, judge_on)
    print("\n" + table + "\n")

    out_path.write_text(
        json.dumps(
            {
                "corpus_hash": corpus_hash,
                "judged": judge_on,
                "configs": {p: r.model_dump() for p, r in reports.items()},
                "table_markdown": table,
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")
    _print_headline(reports)
    return reports


def _comparison_table(reports: dict[str, RunReport], judge_on: bool) -> str:
    headers = ["config", "recall@5", "MRR", "retr p95 ms"]
    if judge_on:
        headers += ["grounded-rate", "unsupp-rate", "gen p95 ms", "$/run"]

    rows: list[list[str]] = []
    for preset, rep in reports.items():
        a = rep.aggregates
        cells = [
            preset,
            _f(a.get("recall_at_5")),
            _f(a.get("mrr")),
            f"{a['retrieval_latency_ms']['p95']:.0f}",
        ]
        if judge_on:
            cells += [
                _f(a.get("groundedness_supported_rate")),
                _f(a.get("unsupported_claim_rate")),
                f"{a['generation_latency_ms']['p95']:.0f}",
                f"${rep.cost_usd:.4f}",
            ]
        rows.append(cells)

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: "| " + " | ".join(  # noqa: E731
        c.ljust(widths[i]) for i, c in enumerate(cells)
    ) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def _print_headline(reports: dict[str, RunReport]) -> None:
    presets = list(reports)
    base = reports[presets[0]].aggregates.get("recall_at_5")
    best_preset = max(
        presets, key=lambda p: reports[p].aggregates.get("recall_at_5") or -1
    )
    best = reports[best_preset].aggregates.get("recall_at_5")
    if base is None or best is None:
        return
    delta = best - base
    print(
        f"\nHeadline: {best_preset} lifted recall@5 from {base:.2f} ({presets[0]}) "
        f"to {best:.2f} (+{delta:.2f})."
    )


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragauge-ablation", description=__doc__)
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--no-judge", action="store_true",
                   help="deterministic recall/MRR ablation only (no LLM calls)")
    p.add_argument("-k", "--top-k", type=int, default=5)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    run_ablation(
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
