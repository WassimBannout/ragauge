"""Load the hand-verified golden set as typed ``GoldRow`` records (T9 -> T10).

JSONL in, validated ``GoldRow``s out. Defaults to the committed candidate set;
point ``--golden`` at the verified file once it lands.
"""

from __future__ import annotations

from pathlib import Path

from ragauge.contracts import GoldRow

DEFAULT_GOLDEN = Path("data/golden/candidates.jsonl")


def load_golden(path: str | Path = DEFAULT_GOLDEN) -> list[GoldRow]:
    path = Path(path)
    rows: list[GoldRow] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(GoldRow.model_validate_json(line))
    ids = [r.id for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate ids in golden set {path}")
    return rows
