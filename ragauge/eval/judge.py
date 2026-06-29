"""LLM-as-judge with a structured, validated schema (DESIGN.md §7.3, T14).

The judge emits a **Pydantic** verdict — ``{supported, unsupported_claims,
score}`` — not free text, so groundedness and unsupported-claim rates aggregate
without parsing prose. It grades the answer strictly against the evidence the
generator was given, never its own world knowledge. Prompt + schema are
versioned with the run so a score is reproducible.

The judge runs **only on answered rows** — abstentions have no claims to grade and
are scored by the cheap, deterministic unanswerable-precision instead.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ragauge.contracts import Answer, RetrievedChunk
from ragauge.eval.cost import cost_usd

# Bump when the prompt or schema changes — versioned with the run for repro.
JUDGE_PROMPT_VERSION = "judge-v1"

SYSTEM_PROMPT = """\
You are a strict groundedness judge for a retrieval-augmented QA system over SEC \
10-K filings. You are given a QUESTION, the system's ANSWER, and the EVIDENCE \
passages the system was allowed to use (each tagged with its chunk_id).

Decide, using ONLY the evidence (never your own knowledge), whether every factual \
claim in the ANSWER is supported by the EVIDENCE.

- `supported`: true only if ALL factual claims in the answer are directly \
supported by the evidence.
- `unsupported_claims`: list each specific claim in the answer that the evidence \
does NOT support (a hallucinated figure, an unsupported comparison, etc.). Empty \
if the answer is fully grounded.
- `score`: a number from 0.0 to 1.0 — the fraction of the answer's claims that are \
supported by the evidence (1.0 = fully grounded, 0.0 = nothing supported)."""


class JudgeVerdict(BaseModel):
    """Structured judge output — the unit the generation metrics roll up from."""

    supported: bool = Field(
        description="True iff every claim in the answer is supported by evidence."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Specific answer claims not supported by the evidence.",
    )
    score: float = Field(
        description="Fraction of the answer's claims supported by evidence (0..1)."
    )


def _format_evidence(retrieved: list[RetrievedChunk]) -> str:
    blocks = []
    for rc in retrieved:
        blocks.append(f"(chunk_id={rc.chunk.chunk_id})\n{rc.chunk.text}")
    return "\n\n".join(blocks)


class Judge:
    """Wraps the judge call behind ``judge(question, answer, evidence)``."""

    def __init__(self, model_id: str, *, max_tokens: int = 1024):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._client = None  # lazy

    def _client_or_init(self):
        if self._client is None:
            import anthropic  # lazy

            self._client = anthropic.Anthropic()
        return self._client

    def judge(
        self,
        question: str,
        answer: Answer,
        retrieved: list[RetrievedChunk],
    ) -> tuple[JudgeVerdict, dict]:
        """Return (verdict, telemetry). Telemetry carries token cost + latency."""
        import time

        client = self._client_or_init()
        user = (
            f"QUESTION: {question}\n\n"
            f"ANSWER: {answer.text}\n\n"
            f"EVIDENCE:\n{_format_evidence(retrieved)}"
        )
        t0 = time.perf_counter()
        response = client.messages.parse(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
            output_format=JudgeVerdict,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = response.usage
        telemetry = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": cost_usd(self.model_id, usage.input_tokens, usage.output_tokens),
            "latency_ms": latency_ms,
        }
        verdict = response.parsed_output
        if verdict is None:
            # No parseable verdict (e.g. a refusal): fail closed — treat the answer
            # as unsupported rather than silently counting it grounded.
            verdict = JudgeVerdict(
                supported=False,
                unsupported_claims=["judge returned no structured verdict"],
                score=0.0,
            )
        return verdict, telemetry
