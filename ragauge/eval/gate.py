"""Regression gate: compare a run's metrics against a committed baseline (T22).

The CI gate runs the 25-question eval subset, then calls this to decide whether
the build passes. It **fails** (exit 1) when a gated metric drops more than its
tolerance below ``evals/baseline.json``:

* ``recall@5`` — deterministic, always available (no API key needed).
* ``groundedness`` (supported-claim rate) — only when the judged run produced it;
  if either side is missing it (e.g. a fork PR with no key), that metric is
  **skipped**, not failed, so the deterministic recall gate still holds.

Both metrics are higher-is-better, so a regression is ``current < baseline -
tolerance``. Improvements and within-tolerance noise pass. The Markdown delta it
renders is what the workflow posts as the PR comment, so the same artifact is the
human-readable diff and the machine verdict.

    python -m ragauge.eval.gate --baseline evals/baseline.json --metrics metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (aggregate key, label, default tolerance). Higher is better for both.
GATED_METRICS: list[tuple[str, str, float]] = [
    ("recall_at_5", "recall@5", 0.05),
    ("groundedness_supported_rate", "groundedness", 0.05),
]
# Shown in the delta for context, never gated.
INFO_METRICS: list[tuple[str, str]] = [
    ("mrr", "MRR"),
    ("unsupported_claim_rate", "unsupported-claim rate"),
]


def load_aggregates(path: str | Path) -> dict:
    """The ``aggregates`` block of a ``RunReport`` JSON (``{}`` if absent)."""
    data = json.loads(Path(path).read_text())
    return data.get("aggregates", data)  # tolerate a bare aggregates dict too


def _meta(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    return {
        "config_hash": data.get("config_hash"),
        "corpus_hash": data.get("corpus_hash"),
        "generator_model_id": data.get("generator_model_id"),
        "judge_model_id": data.get("judge_model_id"),
    }


def compare(
    baseline: dict, current: dict, tolerances: dict[str, float]
) -> tuple[list[dict], bool]:
    """Build per-metric delta rows and the overall pass/fail.

    A metric is *gated* only when both sides have a numeric value; a ``None`` on
    either side is reported as ``skipped`` and never fails the build (so the
    recall gate survives a key-less, groundedness-free run).
    """
    rows: list[dict] = []
    ok = True
    for key, label, default_tol in GATED_METRICS:
        tol = tolerances.get(key, default_tol)
        base = baseline.get(key)
        cur = current.get(key)
        row = {
            "metric": label,
            "key": key,
            "baseline": base,
            "current": cur,
            "tolerance": tol,
            "delta": None,
            "gated": True,
        }
        if base is None or cur is None:
            row["status"] = "skipped"
            row["reason"] = (
                "no baseline yet" if base is None else "not evaluated this run"
            )
        else:
            delta = cur - base
            row["delta"] = delta
            if delta < -tol:
                row["status"] = "FAIL"
                ok = False
            else:
                row["status"] = "pass"
        rows.append(row)
    return rows, ok


def _info_rows(baseline: dict, current: dict) -> list[dict]:
    rows = []
    for key, label in INFO_METRICS:
        base, cur = baseline.get(key), current.get(key)
        delta = (cur - base) if (base is not None and cur is not None) else None
        rows.append(
            {"metric": label, "baseline": base, "current": cur, "delta": delta}
        )
    return rows


def _num(v) -> str:
    return "—" if v is None else f"{v:.4f}"


_STATUS_ICON = {"pass": "✅ pass", "FAIL": "❌ **FAIL**", "skipped": "⚪ skipped"}


def render_markdown(
    rows: list[dict],
    info: list[dict],
    ok: bool,
    *,
    baseline_meta: dict,
    current_meta: dict,
    n_questions: int | None,
) -> str:
    title = "## RAGauge eval gate — " + ("✅ pass" if ok else "❌ regression")
    lines = [title, ""]

    head = "Quality gate vs. `evals/baseline.json`"
    if n_questions is not None:
        head += f" ({n_questions}-question subset)"
    lines += [head + ".", ""]

    lines.append("| metric | baseline | current | Δ | tolerance | status |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        status = _STATUS_ICON.get(r["status"], r["status"])
        if r["status"] == "skipped":
            status += f" ({r['reason']})"
        lines.append(
            f"| {r['metric']} | {_num(r['baseline'])} | {_num(r['current'])} "
            f"| {_signed(r['delta'])} | −{r['tolerance']:.2f} | {status} |"
        )

    if any(i["baseline"] is not None or i["current"] is not None for i in info):
        lines += ["", "<sub>For context (not gated):</sub>", ""]
        lines.append("| metric | baseline | current | Δ |")
        lines.append("| --- | --- | --- | --- |")
        for i in info:
            lines.append(
                f"| {i['metric']} | {_num(i['baseline'])} | {_num(i['current'])} "
                f"| {_signed(i['delta'])} |"
            )

    # Provenance: warn loudly if the baseline graded a different corpus/config,
    # which would make the delta meaningless.
    notes = []
    if (
        baseline_meta.get("corpus_hash")
        and current_meta.get("corpus_hash")
        and baseline_meta["corpus_hash"] != current_meta["corpus_hash"]
    ):
        notes.append(
            f"⚠️ corpus changed since baseline "
            f"(`{baseline_meta['corpus_hash']}` → `{current_meta['corpus_hash']}`) "
            "— regenerate `evals/baseline.json`."
        )
    if (
        baseline_meta.get("config_hash")
        and current_meta.get("config_hash")
        and baseline_meta["config_hash"] != current_meta["config_hash"]
    ):
        notes.append(
            f"⚠️ pipeline config changed since baseline "
            f"(`{baseline_meta['config_hash']}` → `{current_meta['config_hash']}`); "
            "a chunking change moves `chunk_id`s, so any recall delta may be stale "
            "— regenerate `evals/baseline.json`."
        )
    if current_meta.get("generator_model_id"):
        notes.append(
            f"generator `{current_meta['generator_model_id']}`, "
            f"judge `{current_meta.get('judge_model_id') or '—'}`."
        )
    if notes:
        lines += [""] + [f"> {n}" for n in notes]

    verdict = (
        "All gated metrics within tolerance."
        if ok
        else "**A gated metric dropped beyond its tolerance — failing the build.**"
    )
    lines += ["", verdict, ""]
    return "\n".join(lines)


def _signed(v) -> str:
    return "—" if v is None else f"{v:+.4f}"


def run_gate(
    *,
    baseline_path: Path,
    metrics_path: Path,
    out_path: Path | None,
    tolerances: dict[str, float],
) -> bool:
    base_agg = load_aggregates(baseline_path)
    cur_agg = load_aggregates(metrics_path)
    rows, ok = compare(base_agg, cur_agg, tolerances)
    info = _info_rows(base_agg, cur_agg)
    md = render_markdown(
        rows,
        info,
        ok,
        baseline_meta=_meta(baseline_path),
        current_meta=_meta(metrics_path),
        n_questions=cur_agg.get("n_questions"),
    )
    print(md)
    if out_path is not None:
        out_path.write_text(md)
    return ok


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragauge-gate", description=__doc__)
    p.add_argument("--baseline", type=Path, default=Path("evals/baseline.json"))
    p.add_argument("--metrics", type=Path, default=Path("metrics.json"))
    p.add_argument("--out", type=Path, default=None,
                   help="write the Markdown delta here (for the PR comment)")
    p.add_argument("--recall-tolerance", type=float, default=0.05,
                   help="max allowed recall@5 drop below baseline")
    p.add_argument("--groundedness-tolerance", type=float, default=0.05,
                   help="max allowed groundedness drop below baseline")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ok = run_gate(
        baseline_path=args.baseline,
        metrics_path=args.metrics,
        out_path=args.out,
        tolerances={
            "recall_at_5": args.recall_tolerance,
            "groundedness_supported_rate": args.groundedness_tolerance,
        },
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
