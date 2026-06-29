"""Embedding backbone (PRD §S1.4).

Default: local ``BAAI/bge-base-en-v1.5`` via sentence-transformers — 768-dim,
512-token max, **asymmetric** query/passage prompting (bge prepends an
instruction to the query, not to passages). Local + deterministic is the
decisive property: a RunReport reproduces from ``(config, corpus, model id)``
forever, with zero marginal ingest cost (PRD §S1.4, §NFR1/NFR5).

A :class:`HashingEmbedder` provides deterministic vectors with no model download
so tests and CI run fast and offline.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from ragauge.config import EmbeddingConfig


class Embedder(Protocol):
    model_id: str
    dim: int

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...
    def count_tokens(self, text: str) -> int: ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class BgeEmbedder:
    """Local sentence-transformers embedder with bge asymmetric prompting."""

    def __init__(self, config: EmbeddingConfig | None = None):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.config = config or EmbeddingConfig()
        self.model_id = self.config.model_id
        self._model = SentenceTransformer(self.model_id)
        self._model.max_seq_length = self.config.max_seq_length
        # Method was renamed across sentence-transformers versions; support both.
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        prefixed = [self.config.query_instruction + t for t in texts]
        return self._encode(prefixed)

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer.encode(text, add_special_tokens=True))


class HashingEmbedder:
    """Deterministic hash-based embedder for tests/CI — no weights, no network.

    Not semantically meaningful; it only guarantees stable, normalized vectors so
    index/retrieval plumbing can be exercised reproducibly.
    """

    def __init__(self, dim: int = 64, model_id: str = "hashing-test-embedder"):
        self.dim = dim
        self.model_id = model_id

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return v

    def _encode(self, texts: list[str]) -> np.ndarray:
        return _l2_normalize(np.vstack([self._vec(t) for t in texts]))

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def count_tokens(self, text: str) -> int:
        return len(text.split())
