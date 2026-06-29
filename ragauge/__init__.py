"""RAGauge — eval-first RAG over SEC 10-K filings.

Slice 1 (this code): ingest + dense retrieval. The serving pipeline is four
bounded components — ``ingest`` / ``retrieve`` / ``generate`` / ``eval`` — that
exchange typed records (see :mod:`ragauge.contracts`) and never hidden global
state. Generation and the eval harness are stubbed out for later slices.
"""

__version__ = "0.1.0"
