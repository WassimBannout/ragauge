"""BM25 sparse retrieval (PRD T17, DESIGN.md §5.1).

Lexical match over a **tokenizer-versioned** sparse index, built in-memory from
the same chunk store the dense index embeds. At this corpus scale
(hundreds–low-thousands of chunks) an Okapi BM25 builds in milliseconds, so —
unlike the persisted dense index — there is no separate artifact that can go
stale; the binding invariant we *do* record is the tokenizer version, which
travels in the run's provenance (``TOKENIZER_VERSION``).

Implemented directly rather than via ``rank_bm25`` for the same reason we use
NumPy over FAISS (see ``index.py``): zero native/third-party deps, fully
portable, and the scoring stays inspectable. BM25 is strong exactly where dense
is weak — exact financial terms, defined entities, and precise figures that a
paraphrase-tuned embedder blurs (DESIGN.md §5.1).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

# Bump this whenever ``tokenize`` changes — a sparse run's results are only
# reproducible against the tokenizer that produced them.
TOKENIZER_VERSION = "bm25-tok-v1"

# Alphanumeric word/number tokens, lowercased. The internal ``.`` clause keeps
# decimals ("383.3") and dotted identifiers as single lexical tokens so a
# precise figure survives as one term instead of fragmenting.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Bm25Index:
    """Okapi BM25 over a fixed chunk corpus.

    Standard parameters (``k1=1.5``, ``b=0.75``); the smoothed IDF form
    ``ln(1 + (N - n + 0.5)/(n + 0.5))`` is always non-negative, so common terms
    can't pull a document's score below zero.
    """

    def __init__(
        self,
        chunk_ids: list[str],
        docs_tokens: list[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if len(chunk_ids) != len(docs_tokens):
            raise ValueError("chunk_ids / docs_tokens length mismatch")
        self.chunk_ids = chunk_ids
        self.k1 = k1
        self.b = b
        self.tokenizer_version = TOKENIZER_VERSION

        self.doc_len = [len(toks) for toks in docs_tokens]
        n_docs = len(docs_tokens)
        self.avgdl = (sum(self.doc_len) / n_docs) if n_docs else 0.0

        # Inverted index term -> [(doc_idx, term_freq), ...] so search only
        # touches documents that contain a query term.
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, toks in enumerate(docs_tokens):
            for term, tf in Counter(toks).items():
                self._postings[term].append((doc_idx, tf))

        self._idf: dict[str, float] = {}
        for term, postings in self._postings.items():
            n = len(postings)
            self._idf[term] = math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5))

    @classmethod
    def from_chunks(cls, chunks: list, **kwargs) -> "Bm25Index":
        chunk_ids = [c.chunk_id for c in chunks]
        docs_tokens = [tokenize(c.text) for c in chunks]
        return cls(chunk_ids, docs_tokens, **kwargs)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-``k`` ``(chunk_id, score)`` by BM25, score descending.

        Documents with no query-term overlap (score 0) are excluded; ties break
        by document index order for determinism (NFR1)."""
        q_terms = tokenize(query)
        if not q_terms or k <= 0:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in set(q_terms):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for doc_idx, tf in postings:
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl
                )
                scores[doc_idx] += idf * (tf * (self.k1 + 1.0)) / denom

        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.chunk_ids[i], float(s)) for i, s in ranked]
