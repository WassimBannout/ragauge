"""CI regression-gate logic (T22).

Pure functions over two aggregate dicts — no network, no model — so they gate
themselves in the deterministic suite. The pass/fail decision is the load-bearing
claim of the CI gate, so the threshold edges and the skip-don't-fail behaviour are
pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragauge.eval.gate import compare, load_aggregates, render_markdown, run_gate

TOL = {"recall_at_5": 0.05, "groundedness_supported_rate": 0.05}


def _status(rows, key):
    return next(r["status"] for r in rows if r["key"] == key)


def test_within_tolerance_passes():
    base = {"recall_at_5": 0.51, "groundedness_supported_rate": 0.90}
    cur = {"recall_at_5": 0.48, "groundedness_supported_rate": 0.87}  # small drops
    rows, ok = compare(base, cur, TOL)
    assert ok
    assert _status(rows, "recall_at_5") == "pass"


def test_recall_drop_beyond_tolerance_fails():
    base = {"recall_at_5": 0.51, "groundedness_supported_rate": 0.90}
    cur = {"recall_at_5": 0.40, "groundedness_supported_rate": 0.90}  # -0.11
    rows, ok = compare(base, cur, TOL)
    assert not ok
    assert _status(rows, "recall_at_5") == "FAIL"


def test_groundedness_drop_beyond_tolerance_fails():
    base = {"recall_at_5": 0.51, "groundedness_supported_rate": 0.90}
    cur = {"recall_at_5": 0.51, "groundedness_supported_rate": 0.80}  # -0.10
    rows, ok = compare(base, cur, TOL)
    assert not ok
    assert _status(rows, "groundedness_supported_rate") == "FAIL"


def test_exact_threshold_is_allowed():
    # delta == -tolerance is within tolerance (fail only when it drops *more*).
    base = {"recall_at_5": 0.50}
    cur = {"recall_at_5": 0.45}  # exactly -0.05
    rows, ok = compare(base, cur, TOL)
    assert ok
    assert _status(rows, "recall_at_5") == "pass"


def test_improvement_passes():
    base = {"recall_at_5": 0.40, "groundedness_supported_rate": 0.70}
    cur = {"recall_at_5": 0.55, "groundedness_supported_rate": 0.95}
    _, ok = compare(base, cur, TOL)
    assert ok


def test_missing_groundedness_is_skipped_not_failed():
    # Deterministic-only run (no judge): groundedness absent on the current side.
    base = {"recall_at_5": 0.51, "groundedness_supported_rate": 0.90}
    cur = {"recall_at_5": 0.51}
    rows, ok = compare(base, cur, TOL)
    assert ok  # recall still gates; groundedness skipped, not failed
    assert _status(rows, "groundedness_supported_rate") == "skipped"


def test_missing_baseline_metric_is_skipped():
    base = {"recall_at_5": 0.51}  # groundedness not yet baselined
    cur = {"recall_at_5": 0.51, "groundedness_supported_rate": 0.90}
    rows, ok = compare(base, cur, TOL)
    assert ok
    assert _status(rows, "groundedness_supported_rate") == "skipped"


def test_render_markdown_flags_failure_and_corpus_drift():
    base = {"recall_at_5": 0.51, "groundedness_supported_rate": 0.90}
    cur = {"recall_at_5": 0.30}
    rows, ok = compare(base, cur, TOL)
    md = render_markdown(
        rows, [], ok,
        baseline_meta={"corpus_hash": "aaa"},
        current_meta={"corpus_hash": "bbb", "generator_model_id": "claude-opus-4-8"},
        n_questions=25,
    )
    assert "❌" in md and "FAIL" in md
    assert "corpus changed" in md
    assert "25-question subset" in md


def test_run_gate_end_to_end(tmp_path: Path):
    (tmp_path / "baseline.json").write_text(
        json.dumps({"corpus_hash": "x", "aggregates": {"recall_at_5": 0.51}})
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps({"corpus_hash": "x", "aggregates": {"recall_at_5": 0.30}})
    )
    out = tmp_path / "delta.md"
    ok = run_gate(
        baseline_path=tmp_path / "baseline.json",
        metrics_path=tmp_path / "metrics.json",
        out_path=out,
        tolerances=TOL,
    )
    assert not ok
    assert out.exists() and "FAIL" in out.read_text()


def test_load_aggregates_tolerates_bare_dict(tmp_path: Path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"recall_at_5": 0.5}))  # no "aggregates" wrapper
    assert load_aggregates(p) == {"recall_at_5": 0.5}
