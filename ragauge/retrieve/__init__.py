"""Retrieve component — query -> ranked, deduped evidence (DESIGN.md §5).

Slice 1 wires the **dense** stage only. ``Retrieve(query, config)`` is the seam
the harness and CLI talk to; BM25 / fusion / rerank attach behind config toggles
in later slices without changing that surface.
"""

from ragauge.retrieve.retriever import Retriever, build_retriever  # noqa: F401
