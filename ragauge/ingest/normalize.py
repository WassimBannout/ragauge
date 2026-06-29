"""Deterministic text normalization (PRD §S1.1.6).

Applied **before** ``chunk_id`` hashing, so ids are stable across rebuilds. No
randomness, no locale dependence — same input always yields the same output.
"""

from __future__ import annotations

import re
import unicodedata

# Characters that are visually whitespace but not ASCII space.
_UNICODE_SPACES = {
    "\xa0": " ",  # non-breaking space
    " ": " ",  # figure space
    " ": " ",  # narrow no-break space
    " ": " ",  # thin space
    " ": " ",  # hair space
    "﻿": "",  # zero-width no-break space / BOM
    "​": "",  # zero-width space
}

# Smart punctuation -> ASCII so hashes don't depend on typography.
_PUNCT = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",  # en dash
    "—": "-",  # em dash
    "−": "-",  # minus sign
    "…": "...",
}

_WS_RUN = re.compile(r"[ \t\f\v]+")
_NEWLINE_RUN = re.compile(r"\n{3,}")
# Hyphenated line-break split: "supply-\nchain" -> "supplychain".
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def normalize_text(text: str, *, collapse_whitespace: bool = True) -> str:
    """Normalize Unicode + whitespace deterministically.

    ``collapse_whitespace=True`` (the default, for prose) collapses runs of
    spaces and trims; tables pass ``False`` to keep their grid newlines.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    for bad, good in {**_UNICODE_SPACES, **_PUNCT}.items():
        text = text.replace(bad, good)

    text = text.replace("\x0c", "\n")  # form feed (page break) -> newline
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    if collapse_whitespace:
        text = _WS_RUN.sub(" ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = _NEWLINE_RUN.sub("\n\n", text)
        text = text.strip()
    else:
        # Tables: collapse only intra-line spaces, preserve line structure.
        text = "\n".join(_WS_RUN.sub(" ", line).strip() for line in text.split("\n"))
        text = text.strip()

    return text
