"""Acquire & vendor 2–3 static 10-Ks from EDGAR (PRD §S1.6 FR1 / T3).

Downloads the **primary 10-K HTML document** (not the SGML full-submission
wrapper) once, stores it under the gitignored ``data/raw/``, and writes a
manifest recording accession/URL + a ``corpus_hash``. SEC requires a declared
User-Agent and polite rate limiting.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from ragauge.config import stable_hash

# Default corpus: ≥2 companies (PRD §S1.6). NVDA included for a 3rd filing.
DEFAULT_CORPUS: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
}

_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{doc}"


def _user_agent() -> str:
    # SEC fair-access policy requires a real contact. Override via env.
    return os.environ.get(
        "SEC_USER_AGENT", "RAGauge research wassimbannout20@gmail.com"
    )


def _get(url: str) -> requests.Response:
    resp = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=30)
    resp.raise_for_status()
    time.sleep(0.2)  # stay well under SEC's 10 req/s
    return resp


@dataclass
class ManifestEntry:
    company: str
    cik: str
    accession: str
    primary_doc: str
    url: str
    fiscal_year: int
    doc_id: str
    source_path: str
    sha256: str


def _latest_10k(cik: str) -> dict:
    data = _get(_SUBMISSIONS.format(cik=cik)).json()
    recent = data["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return {
                "accession": recent["accessionNumber"][i],
                "primary_doc": recent["primaryDocument"][i],
                "report_date": recent["reportDate"][i],
            }
    raise RuntimeError(f"no 10-K found for CIK {cik}")


def acquire_corpus(
    corpus: dict[str, str] | None = None,
    raw_dir: str | Path = "data/raw",
    manifest_path: str | Path = "data/manifest.json",
) -> list[ManifestEntry]:
    """Download each filing's primary HTML, write files + manifest, return it."""
    corpus = corpus or DEFAULT_CORPUS
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    entries: list[ManifestEntry] = []
    for ticker, cik in corpus.items():
        info = _latest_10k(cik)
        acc_nodash = info["accession"].replace("-", "")
        url = _ARCHIVE.format(cik_int=int(cik), acc=acc_nodash, doc=info["primary_doc"])
        html = _get(url).content
        fiscal_year = int(info["report_date"][:4])
        doc_id = f"{ticker}-10K-FY{fiscal_year}"
        dest = raw_dir / f"{doc_id}.htm"
        dest.write_bytes(html)
        entries.append(
            ManifestEntry(
                company=ticker,
                cik=cik,
                accession=info["accession"],
                primary_doc=info["primary_doc"],
                url=url,
                fiscal_year=fiscal_year,
                doc_id=doc_id,
                source_path=str(dest),
                sha256=stable_hash(html.decode("latin-1")),
            )
        )

    corpus_hash = stable_hash(sorted(e.sha256 for e in entries))
    manifest_path = Path(manifest_path)
    manifest_path.write_text(
        json.dumps(
            {"corpus_hash": corpus_hash, "filings": [asdict(e) for e in entries]},
            indent=2,
        ),
        encoding="utf-8",
    )
    return entries


def load_manifest(manifest_path: str | Path = "data/manifest.json") -> dict:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))
