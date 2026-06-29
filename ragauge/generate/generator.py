"""Generate: grounded, cited answers or a first-class abstention (DESIGN.md §6).

The generator answers **only** from the retrieved evidence and cites the
``chunk_id``(s) backing the answer, or abstains. Two abstention triggers
(DESIGN.md §6.2):

1. **Pre-generation gate** — if there is no evidence (or the top score is below
   ``retrieval.min_score``), short-circuit to an abstention *without* spending an
   LLM call.
2. **Post-generation honesty** — even with evidence, the model may conclude the
   specific fact isn't present and set ``abstained=true``.

Output is **structured** (``client.messages.parse`` + a Pydantic schema), so
citations and the abstention flag are parsed, never regexed out of prose. The
``anthropic`` import is lazy so the deterministic retrieval metrics run with no
SDK and no API key.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from ragauge.config import RetrievalConfig
from ragauge.contracts import Answer, RetrievedChunk
from ragauge.eval.cost import cost_usd

# Bump when the prompt or schema changes — versioned with the run for repro.
GENERATOR_PROMPT_VERSION = "gen-v1"

SYSTEM_PROMPT = """\
You answer questions about SEC 10-K filings using ONLY the numbered evidence \
passages provided. Each passage is labelled [n] and tagged with its chunk_id.

Rules:
- Use only facts stated in the evidence. Never rely on outside or prior knowledge.
- Cite the chunk_id(s) that support your answer in the `citations` field.
- If the evidence does not contain the answer, set `abstained` to true, leave \
`answer` empty, and cite nothing. Do not guess or fabricate figures.
- Keep the answer concise and factual; report numbers with their units as given."""


class GenerationOutput(BaseModel):
    """Structured generator result (the schema the model fills in)."""

    abstained: bool = Field(
        description="True if the evidence does not support an answer."
    )
    answer: str = Field(
        default="", description="The grounded answer, or empty if abstained."
    )
    citations: list[str] = Field(
        default_factory=list,
        description="chunk_ids from the evidence supporting the answer.",
    )


def _format_evidence(retrieved: list[RetrievedChunk]) -> str:
    blocks = []
    for i, rc in enumerate(retrieved, start=1):
        blocks.append(f"[{i}] (chunk_id={rc.chunk.chunk_id})\n{rc.chunk.text}")
    return "\n\n".join(blocks)


class Generator:
    """Wraps the generation call behind ``generate(question, evidence) -> Answer``."""

    def __init__(self, model_id: str, *, max_tokens: int = 1024):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._client = None  # lazy

    def _client_or_init(self):
        if self._client is None:
            import anthropic  # lazy: deterministic metrics don't need the SDK

            self._client = anthropic.Anthropic()
        return self._client

    def generate(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        config: RetrievalConfig | None = None,
    ) -> Answer:
        config = config or RetrievalConfig()
        t0 = time.perf_counter()

        # Trigger 1: pre-generation evidence gate — abstain before the LLM call.
        if self._evidence_too_weak(retrieved, config.min_score):
            return Answer(
                text="",
                citations=[],
                abstained=True,
                evidence_used=[rc.chunk.chunk_id for rc in retrieved],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        client = self._client_or_init()
        user = f"Question: {question}\n\nEvidence:\n{_format_evidence(retrieved)}"
        response = client.messages.parse(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
            output_format=GenerationOutput,
        )
        out: GenerationOutput | None = response.parsed_output
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = response.usage
        valid_ids = {rc.chunk.chunk_id for rc in retrieved}

        if out is None:
            # No parseable structured output (e.g. a safety refusal) — abstain
            # rather than fabricate. The honest outcome, not an error.
            return Answer(
                text="",
                citations=[],
                abstained=True,
                evidence_used=list(valid_ids),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost_usd(
                    self.model_id, usage.input_tokens, usage.output_tokens
                ),
                latency_ms=latency_ms,
            )

        # Drop any citation the model invented that isn't in the supplied evidence.
        citations = [c for c in out.citations if c in valid_ids]

        return Answer(
            text="" if out.abstained else out.answer,
            citations=[] if out.abstained else citations,
            abstained=out.abstained,
            evidence_used=list(valid_ids),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost_usd(self.model_id, usage.input_tokens, usage.output_tokens),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _evidence_too_weak(
        retrieved: list[RetrievedChunk], min_score: float | None
    ) -> bool:
        if not retrieved:
            return True
        if min_score is not None and retrieved[0].score < min_score:
            return True
        return False
