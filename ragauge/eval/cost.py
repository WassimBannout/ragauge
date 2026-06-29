"""Real-dollar cost from provider token counts (PRD §7.2 Ops, T15 / T21).

Cost is derived from the **provider's own token counts** returned on every
response (``usage.input_tokens`` / ``usage.output_tokens`` and the cache
breakdown ``cache_creation_input_tokens`` / ``cache_read_input_tokens``) — never
a generic tokenizer — so ``$/run`` is the real *billed* figure, not an estimate.
Using the billed ``usage`` is strictly more accurate than a pre-flight
``messages.count_tokens`` call: it captures output tokens and the cache split,
which a token count of the prompt cannot.

``PRICING`` is per-1M-token (input, output) in USD, verified against the live
Claude API reference (models cached 2026-06-04). Update from the same source if a
model is added — do not guess. ``CAPABILITY_RANK`` orders models so the harness
can enforce "the judge is at least as capable as the generator" (PRD §7.3).

Prompt-cache economics (5-minute ephemeral cache): a cache **write** bills at
``1.25×`` the base input rate, a cache **read** at ``0.10×``. The sweep (T21)
reports ``$/run`` with caching on vs. off by repricing the same provider token
counts under both multipliers — no second API pass needed.
"""

from __future__ import annotations

# (input $/1M, output $/1M) — from the Claude API model/pricing reference.
PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.00, 50.00),
    "claude-mythos-5": (10.00, 50.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Cache-pricing multipliers on the base input rate (5-minute ephemeral cache).
CACHE_WRITE_MULTIPLIER = 1.25  # writing the prefix into the cache
CACHE_READ_MULTIPLIER = 0.10  # reading a cached prefix on a later request

# Higher = more capable. Only ordering matters (judge >= generator gate).
CAPABILITY_RANK: dict[str, int] = {
    "claude-haiku-4-5": 1,
    "claude-sonnet-4-6": 2,
    "claude-opus-4-6": 3,
    "claude-opus-4-7": 4,
    "claude-opus-4-8": 5,
    "claude-fable-5": 6,
    "claude-mythos-5": 6,
}


def _rates(model_id: str) -> tuple[float, float]:
    if model_id not in PRICING:
        raise KeyError(
            f"no pricing for {model_id!r}; add it from the live pricing reference"
        )
    return PRICING[model_id]


def cost_usd(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Billed dollar cost of one call from the provider's token counts.

    ``input_tokens`` is the *uncached* remainder the provider charges at the full
    input rate; cached tokens are billed separately at the write/read multipliers.
    With caching off both cache counts are 0 and this reduces to the plain
    ``input × rate + output × rate`` figure used by the rest of the harness.
    """
    in_rate, out_rate = _rates(model_id)
    return (
        input_tokens * in_rate
        + cache_creation_tokens * in_rate * CACHE_WRITE_MULTIPLIER
        + cache_read_tokens * in_rate * CACHE_READ_MULTIPLIER
        + output_tokens * out_rate
    ) / 1e6


def cost_usd_uncached(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """What the *same* call would have cost with caching off — every cached token
    re-billed at the full input rate. The cache-on/off baseline for T21: subtract
    :func:`cost_usd` from this to get the realized saving on identical token counts.
    """
    in_rate, out_rate = _rates(model_id)
    full_input = input_tokens + cache_creation_tokens + cache_read_tokens
    return (full_input * in_rate + output_tokens * out_rate) / 1e6


def assert_judge_at_least_as_capable(generator_model: str, judge_model: str) -> None:
    """Enforce PRD §7.3: a weaker model must not grade a stronger one's output."""
    for m in (generator_model, judge_model):
        if m not in CAPABILITY_RANK:
            raise KeyError(f"unranked model {m!r}; add it to CAPABILITY_RANK")
    if CAPABILITY_RANK[judge_model] < CAPABILITY_RANK[generator_model]:
        raise ValueError(
            f"judge {judge_model!r} is less capable than generator "
            f"{generator_model!r}; the judge must be >= the generator (PRD §7.3)"
        )
