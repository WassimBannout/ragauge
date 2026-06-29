"""Exact flat vector index (PRD §S1.5 / FR7).

At this corpus scale (hundreds–low-thousands of chunks) ANN approximation buys
nothing and would inject error into the very recall@5 we measure — so we use
**exact** search. Implemented as a brute-force inner-product scan over
L2-normalized vectors (cosine), which *is* an exact flat index. We use NumPy
rather than FAISS `IndexFlatIP`: identical semantics, no native dependency, fully
CPU-portable (NFR3) — a deliberate, documented substitution.

Artifacts are **self-describing**: the directory is stamped with
``embedding_model_id + corpus_hash + chunking_config_hash`` so a stale or
mismatched index cannot be silently served (DESIGN.md §9).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_VECTORS = "vectors.npy"
_IDS = "ids.json"
_META = "meta.json"


def build_dense_index(
    chunks: list,
    embedder,
    *,
    corpus_hash: str,
    chunking_config_hash: str,
    index_dir: str | "Path",
) -> "ExactFlatIndex":
    """Embed all chunk texts and build + persist the stamped exact flat index."""
    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    embeddings = embedder.encode_passages(texts)
    index = ExactFlatIndex.build(
        embeddings,
        chunk_ids,
        embedding_model_id=embedder.model_id,
        corpus_hash=corpus_hash,
        chunking_config_hash=chunking_config_hash,
        dim=embedder.dim,
    )
    index.save(index_dir)
    return index


class IndexStampMismatch(RuntimeError):
    """Raised when a built index does not match the requested config/corpus."""


class ExactFlatIndex:
    def __init__(self, vectors: np.ndarray, chunk_ids: list[str], meta: dict):
        if vectors.shape[0] != len(chunk_ids):
            raise ValueError("vectors / chunk_ids length mismatch")
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.chunk_ids = chunk_ids
        self.meta = meta

    # ----------------------------------------------------------------- #
    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        chunk_ids: list[str],
        *,
        embedding_model_id: str,
        corpus_hash: str,
        chunking_config_hash: str,
        dim: int,
    ) -> "ExactFlatIndex":
        meta = {
            "embedding_model_id": embedding_model_id,
            "corpus_hash": corpus_hash,
            "chunking_config_hash": chunking_config_hash,
            "dim": dim,
            "n_vectors": len(chunk_ids),
        }
        return cls(embeddings, chunk_ids, meta)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return the top-``k`` ``(chunk_id, score)`` by inner product.

        Ties broken by index order for determinism (NFR1)."""
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        scores = self.vectors @ q
        k = min(k, len(self.chunk_ids))
        if k == 0:
            return []
        # argpartition for speed, then a stable sort of the shortlist.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.lexsort((top, -scores[top]))]
        return [(self.chunk_ids[i], float(scores[i])) for i in top]

    # ----------------------------------------------------------------- #
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / _VECTORS, self.vectors)
        (directory / _IDS).write_text(json.dumps(self.chunk_ids), encoding="utf-8")
        (directory / _META).write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expect_model_id: str | None = None,
        expect_corpus_hash: str | None = None,
        expect_chunking_hash: str | None = None,
    ) -> "ExactFlatIndex":
        directory = Path(directory)
        vectors = np.load(directory / _VECTORS)
        chunk_ids = json.loads((directory / _IDS).read_text(encoding="utf-8"))
        meta = json.loads((directory / _META).read_text(encoding="utf-8"))

        def _check(name: str, expected: str | None) -> None:
            if expected is not None and meta.get(name) != expected:
                raise IndexStampMismatch(
                    f"index {name}={meta.get(name)!r} != expected {expected!r}; "
                    "rebuild the index"
                )

        _check("embedding_model_id", expect_model_id)
        _check("corpus_hash", expect_corpus_hash)
        _check("chunking_config_hash", expect_chunking_hash)
        return cls(vectors, chunk_ids, meta)
