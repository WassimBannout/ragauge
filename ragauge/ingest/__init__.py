"""Ingest component — raw filings -> structure-aware Chunks + dense index inputs.

Pipeline: acquire (EDGAR) -> parse (HTML -> blocks, table discrimination) ->
segment (Item boundaries) -> chunk (size-bounded, structure-aware) -> store
(JSONL keyed by chunk_id). See PRD §S1.
"""

from ragauge.ingest.pipeline import ingest_filing, ingest_corpus  # noqa: F401
