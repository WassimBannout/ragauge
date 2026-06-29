"""RAGauge CLI — the demoable surface for Slice 1 (ingest + dense retrieval).

Subcommands map to the build-order milestones (DESIGN.md §12):

    ragauge acquire       # T3  download 2–3 10-Ks + manifest
    ragauge ingest        # T4/T5/T6  parse -> segment -> chunk -> store
    ragauge inspect       # T6  dump chunks for a doc/section (no re-parse)
    ragauge build-index   # T7  embed chunks + build the exact dense index
    ragauge query         # T8  dense top-k for a question
    ragauge eval          # T10/T14-T16  golden set -> metrics.json + summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from ragauge.config import PipelineConfig
from ragauge.eval.golden import DEFAULT_GOLDEN
from ragauge.eval.run import DEFAULT_GENERATOR_MODEL, DEFAULT_JUDGE_MODEL

DATA = Path("data")
MANIFEST = DATA / "manifest.json"
STORE = DATA / "chunks.jsonl"
INDEX_DIR = Path("indexes/dense")


def _bge():
    from ragauge.retrieve.embedder import BgeEmbedder

    cfg = PipelineConfig()
    return BgeEmbedder(cfg.embedding), cfg


# --------------------------------------------------------------------------- #
def cmd_acquire(args: argparse.Namespace) -> int:
    from ragauge.ingest.acquire import DEFAULT_CORPUS, acquire_corpus

    corpus = DEFAULT_CORPUS
    if args.tickers:
        corpus = {t: DEFAULT_CORPUS[t] for t in args.tickers if t in DEFAULT_CORPUS}
    entries = acquire_corpus(corpus, raw_dir=DATA / "raw", manifest_path=MANIFEST)
    for e in entries:
        print(f"  {e.doc_id:20s} {e.accession}  -> {e.source_path}")
    print(f"\n{len(entries)} filings vendored; manifest at {MANIFEST}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from ragauge.ingest.acquire import load_manifest
    from ragauge.ingest.pipeline import ingest_corpus

    embedder, cfg = _bge()
    manifest = load_manifest(MANIFEST)
    chunks = ingest_corpus(
        manifest,
        store_path=STORE,
        config=cfg.chunking,
        count_tokens=embedder.count_tokens,
    )
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.doc_id] = by_doc.get(c.doc_id, 0) + 1
    for doc, n in sorted(by_doc.items()):
        print(f"  {doc:20s} {n:5d} chunks")
    print(f"\n{len(chunks)} chunks -> {STORE}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from ragauge.contracts import Section
    from ragauge.ingest.store import load_chunks

    chunks = load_chunks(STORE)
    if args.doc:
        chunks = [c for c in chunks if args.doc.upper() in c.doc_id.upper()]
    if args.section:
        want = Section(args.section)
        chunks = [c for c in chunks if c.section == want]
    chunks = chunks[: args.limit]
    for c in chunks:
        print("─" * 80)
        print(f"{c.chunk_id}  [{c.content_type.value}]  {c.token_count} tok")
        print(f"anchor: {c.anchor}")
        body = c.text if args.full else (c.text[:600] + ("…" if len(c.text) > 600 else ""))
        print(body)
    print("─" * 80)
    print(f"{len(chunks)} chunk(s) shown")
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    from ragauge.ingest.acquire import load_manifest
    from ragauge.ingest.store import load_chunks
    from ragauge.retrieve.index import build_dense_index

    embedder, cfg = _bge()
    manifest = load_manifest(MANIFEST)
    chunks = load_chunks(STORE)
    print(f"embedding {len(chunks)} chunks with {embedder.model_id} (dim={embedder.dim})…")
    build_dense_index(
        chunks,
        embedder,
        corpus_hash=manifest["corpus_hash"],
        chunking_config_hash=cfg.chunking.hash(),
        index_dir=INDEX_DIR,
    )
    print(f"index built -> {INDEX_DIR} (stamped model+corpus+chunking)")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    import time

    from ragauge.retrieve.retriever import build_retriever

    embedder, cfg = _bge()
    cfg.retrieval.top_k = args.k
    retriever = build_retriever(
        embedder, index_dir=INDEX_DIR, store_path=STORE, config=cfg
    )
    t0 = time.perf_counter()
    results = retriever.retrieve(args.question, cfg.retrieval)
    dt = (time.perf_counter() - t0) * 1000
    print(f'Q: {args.question}   (dense top-{args.k}, {dt:.0f} ms)\n')
    for r in results:
        c = r.chunk
        preview = " ".join(c.text.split())[:220]
        print(f"[{r.score:.3f}] {c.chunk_id}")
        print(f"        {c.company} FY{c.fiscal_year} · {c.section.value} · {c.content_type.value}")
        print(f"        {preview}…\n")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from ragauge.eval.run import run

    run(
        golden_path=args.golden,
        out_path=args.out,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        no_judge=args.no_judge,
        top_k=args.k,
    )
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragauge", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("acquire", help="download 2–3 10-Ks + manifest")
    a.add_argument("--tickers", nargs="*", help="subset of AAPL MSFT NVDA")
    a.set_defaults(func=cmd_acquire)

    i = sub.add_parser("ingest", help="parse -> segment -> chunk -> store")
    i.set_defaults(func=cmd_ingest)

    n = sub.add_parser("inspect", help="dump chunks for a doc/section")
    n.add_argument("--doc", help="filter by doc_id substring, e.g. AAPL")
    n.add_argument("--section", help="ITEM_1 | ITEM_1A | ITEM_7 | ITEM_8 | OTHER")
    n.add_argument("--limit", type=int, default=10)
    n.add_argument("--full", action="store_true", help="print full chunk text")
    n.set_defaults(func=cmd_inspect)

    b = sub.add_parser("build-index", help="embed chunks + build dense index")
    b.set_defaults(func=cmd_build_index)

    q = sub.add_parser("query", help="dense top-k for a question")
    q.add_argument("question")
    q.add_argument("-k", type=int, default=5)
    q.set_defaults(func=cmd_query)

    e = sub.add_parser("eval", help="run the golden set -> metrics.json + summary")
    e.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN,
                   help="golden-set JSONL (default: committed candidate set)")
    e.add_argument("--out", type=Path, default=Path("metrics.json"))
    e.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    e.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    e.add_argument("--no-judge", action="store_true",
                   help="deterministic retrieval metrics only (no LLM calls)")
    e.add_argument("-k", type=int, default=5)
    e.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
