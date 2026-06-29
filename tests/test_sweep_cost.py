"""Cache-aware cost math + sweep wiring (T21).

These are pure-function checks — no network, no model — so they run in CI with the
deterministic suite. The dollar figures are the load-bearing claim of the sweep,
so the write/read multipliers and the on-vs-off baseline are pinned here.
"""

from __future__ import annotations

import pytest

from ragauge.eval.cost import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    cost_usd,
    cost_usd_uncached,
)
from ragauge.eval.sweep import _recommendation, _saving, _tier_labels


def test_cost_usd_no_cache_matches_plain_rate():
    # opus 4.8: $5 / $25 per 1M
    assert cost_usd("claude-opus-4-8", 1_000_000, 0) == pytest.approx(5.0)
    assert cost_usd("claude-opus-4-8", 0, 1_000_000) == pytest.approx(25.0)


def test_cache_read_is_cheaper_than_full_input():
    # 1M tokens served from cache cost 0.10x the input rate.
    cached = cost_usd("claude-opus-4-8", 0, 0, cache_read_tokens=1_000_000)
    assert cached == pytest.approx(5.0 * CACHE_READ_MULTIPLIER)


def test_cache_write_carries_a_premium():
    written = cost_usd("claude-opus-4-8", 0, 0, cache_creation_tokens=1_000_000)
    assert written == pytest.approx(5.0 * CACHE_WRITE_MULTIPLIER)


def test_uncached_baseline_reprices_cache_at_full_rate():
    # The same token counts: caching-off bills every cached token at full input.
    on = cost_usd(
        "claude-sonnet-4-6", 100, 50, cache_creation_tokens=200, cache_read_tokens=400
    )
    off = cost_usd_uncached(
        "claude-sonnet-4-6", 100, 50, cache_creation_tokens=200, cache_read_tokens=400
    )
    # off = (100+200+400) input + 50 output, all at full rate
    in_rate, out_rate = 3.0, 15.0
    assert off == pytest.approx((700 * in_rate + 50 * out_rate) / 1e6)
    # A cache read is cheaper than full input, so on must beat off here.
    assert on < off


def test_uncached_equals_cost_usd_when_no_cache_tokens():
    args = ("claude-haiku-4-5", 321, 123)
    assert cost_usd(*args) == pytest.approx(cost_usd_uncached(*args))


def test_tier_labels_order_by_price():
    labels = _tier_labels(
        ["claude-opus-4-8", "claude-haiku-4-5", "claude-sonnet-4-6"]
    )
    assert labels["claude-haiku-4-5"] == "cheapest"
    assert labels["claude-sonnet-4-6"] == "balanced"
    assert labels["claude-opus-4-8"] == "most capable"


def test_saving_formats_zero_when_equal():
    assert _saving(0.0, 0.0) == "n/a"
    assert "0.0%" in _saving(1.0, 1.0)


def test_recommendation_prefers_cheaper_within_tolerance():
    rows = [
        {"model": "claude-haiku-4-5", "tier": "cheapest",
         "groundedness_supported_rate": 0.92, "cost_run_usd": 0.10,
         "cost_per_query_usd": 0.003},
        {"model": "claude-opus-4-8", "tier": "most capable",
         "groundedness_supported_rate": 0.94, "cost_run_usd": 0.80,
         "cost_per_query_usd": 0.026},
    ]
    rec = _recommendation(rows)
    # within 5 points and far cheaper → recommend haiku
    assert "claude-haiku-4-5" in rec
