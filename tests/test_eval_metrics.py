"""T10/T14-T16 — deterministic eval metrics, cost, and judge-capability gate.

These are the LLM-free guarantees of the harness: recall@5 / MRR /
unanswerable-precision are pure functions, cost is exact from provider token
counts, and a weaker model must never be allowed to judge a stronger one.
"""

import pytest

from ragauge.eval import metrics
from ragauge.eval.cost import (
    assert_judge_at_least_as_capable,
    cost_usd,
)


def test_recall_at_k_single_and_multi_gold():
    assert metrics.recall_at_k(["a"], ["x", "a", "y"], 5) == 1.0
    assert metrics.recall_at_k(["a", "b"], ["x", "a", "y"], 5) == 0.5
    # gold at rank 6 is outside top-5
    assert metrics.recall_at_k(["a"], ["1", "2", "3", "4", "5", "a"], 5) == 0.0
    # no gold chunks (unanswerable) -> defined as 0.0; callers exclude these rows
    assert metrics.recall_at_k([], ["a"], 5) == 0.0


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(["a"], ["a"]) == 1.0
    assert metrics.reciprocal_rank(["a"], ["x", "a", "y"]) == 0.5
    assert metrics.reciprocal_rank(["a"], ["x", "y"]) == 0.0


def test_unanswerable_precision_and_recall():
    abstained = [True, True, False]
    is_unans = [True, False, False]
    prec, n_abs, n_correct = metrics.unanswerable_precision(abstained, is_unans)
    assert (prec, n_abs, n_correct) == (0.5, 2, 1)
    assert metrics.unanswerable_recall(abstained, is_unans) == 1.0


def test_unanswerable_precision_undefined_when_never_abstains():
    assert metrics.unanswerable_precision([False, False], [True, False]) == (None, 0, 0)


def test_percentiles():
    p = metrics.percentiles([10, 20, 30, 40, 100])
    assert p["p50"] == 30 and p["max"] == 100
    assert metrics.percentiles([]) == {"p50": 0.0, "p95": 0.0, "max": 0.0}


def test_cost_from_provider_token_counts():
    # opus-4-8: $5 in / $25 out per 1M
    assert cost_usd("claude-opus-4-8", 1_000_000, 1_000_000) == pytest.approx(30.0)
    assert cost_usd("claude-opus-4-8", 0, 0) == 0.0
    with pytest.raises(KeyError):
        cost_usd("not-a-model", 1, 1)


def test_judge_capability_gate():
    # equal is allowed
    assert_judge_at_least_as_capable("claude-opus-4-8", "claude-opus-4-8") is None
    # stronger judge is allowed
    assert_judge_at_least_as_capable("claude-sonnet-4-6", "claude-opus-4-8") is None
    # weaker judge is rejected
    with pytest.raises(ValueError):
        assert_judge_at_least_as_capable("claude-opus-4-8", "claude-haiku-4-5")
