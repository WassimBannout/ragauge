"""Real-dollar cost from provider token counts (PRD §7.2 Ops, T15).

Cost is derived from the **provider's own token counts** returned on every
response (``usage.input_tokens`` / ``usage.output_tokens``) — never a generic
tokenizer — so ``$/run`` is the real billed figure, not an estimate.

``PRICING`` is per-1M-token (input, output) in USD, verified against the live
Claude API reference (models cached 2026-06-04). Update from the same source if a
model is added — do not guess. ``CAPABILITY_RANK`` orders models so the harness
can enforce "the judge is at least as capable as the generator" (PRD §7.3).
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


def cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one call from the provider's token counts."""
    if model_id not in PRICING:
        raise KeyError(
            f"no pricing for {model_id!r}; add it from the live pricing reference"
        )
    in_rate, out_rate = PRICING[model_id]
    return input_tokens / 1e6 * in_rate + output_tokens / 1e6 * out_rate


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
