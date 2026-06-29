"""Chunk store — the source of truth for text + metadata (PRD §S1.5).

Inspectable **JSONL keyed by ``chunk_id``**: one chunk per line. The vector
index owns only geometry; this store owns everything human-readable, so the T6
inspect CLI re-loads chunks without re-parsing the filings.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragauge.contracts import Chunk


def write_chunks(path: str | Path, chunks: list[Chunk]) -> int:
    """Write chunks as JSONL, de-duplicated by ``chunk_id`` (stable order)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            fh.write(chunk.model_dump_json() + "\n")
            n += 1
    return n


def load_chunks(path: str | Path) -> list[Chunk]:
    path = Path(path)
    out: list[Chunk] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Chunk.model_validate_json(line))
    return out


def load_chunk_map(path: str | Path) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in load_chunks(path)}
