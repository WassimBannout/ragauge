"""Normalization is deterministic and runs before chunk_id hashing (PRD §S1.1.6)."""

from ragauge.ingest.normalize import normalize_text


def test_smart_punctuation_and_spaces_normalized():
    raw = "“Total” revenue was – − great…"
    out = normalize_text(raw)
    assert '"Total"' in out
    assert " " not in out and "–" not in out and "−" not in out
    assert out.endswith("great...")


def test_hyphenated_linebreak_repaired_and_whitespace_collapsed():
    out = normalize_text("supply-\nchain   risk\t\tmatters")
    assert "supplychain risk matters" == out


def test_deterministic():
    raw = "Foo   bar baz\n\n\nqux"
    assert normalize_text(raw) == normalize_text(raw)


def test_table_mode_preserves_newlines():
    out = normalize_text("a | b\nc | d", collapse_whitespace=False)
    assert "\n" in out
