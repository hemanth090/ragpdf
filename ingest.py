"""CLI: ingest a PDF or a directory of PDFs, page-level, incremental, resumable, auto-scaling."""

import argparse
import os
import sys
import time

from autoscale import AdaptiveBatch
from config import (
    COLLECTION,
    EMBED_DIM,
    EMBED_MODEL,
    WRITE_BATCH_MAX,
    WRITE_BATCH_START,
    WRITE_TARGET_S,
)
from extract import extract
from store import (
    count,
    delete_stale_pages,
    ensure_collection,
    existing_hashes,
    upsert_pages,
)
from writer import AdaptiveWriter


def find_pdfs(target: str) -> list[str]:
    if os.path.isfile(target):
        return [target]
    found = []
    for root, _, files in os.walk(target):
        for f in files:
            if f.lower().endswith(".pdf"):  # also catches .PDF
                found.append(os.path.join(root, f))
    return sorted(found)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest PDFs into Qdrant, one point per page.")
    ap.add_argument("target", help="PDF file or directory")
    ap.add_argument("--doc-id", help="Override doc_id (single PDF only)")
    ap.add_argument("--recreate", action="store_true",
                    help="DROP THE WHOLE COLLECTION (all documents) and rebuild")
    ap.add_argument("--force", action="store_true",
                    help="Re-embed every page of the given PDF(s), ignoring the unchanged-page check")
    ap.add_argument("--text-only", action="store_true", help="Skip pages with no extractable text")
    args = ap.parse_args()

    pdfs = find_pdfs(args.target)
    if not pdfs:
        print(f"no PDFs found at {args.target}")
        return 1
    if args.doc_id and len(pdfs) > 1:
        print("--doc-id only works with a single PDF (otherwise every file would share one doc_id).")
        return 1

    if args.recreate:
        print("WARNING: --recreate drops the entire collection, not just this target.")
    ensure_collection(recreate=args.recreate)
    print(f"collection {COLLECTION}: {EMBED_MODEL} @ {EMBED_DIM} dims, {len(pdfs)} file(s)")

    from embeddings import embed_stream, status_line

    total_pages = total_skipped = total_embedded = total_no_text = 0
    t0 = time.time()

    for pdf in pdfs:
        pages = extract(pdf)
        if args.doc_id:
            for p in pages:
                p.doc_id = args.doc_id
        source = pages[0].source if pages else os.path.abspath(pdf)
        total_pages += len(pages)

        if args.text_only:
            pages = [p for p in pages if p.has_text]

        # Drop pages that no longer exist (PDF got shorter) and find what changed.
        if pages:
            delete_stale_pages(source, pages[0].page_count)
        stored = {} if args.force else existing_hashes(source)
        todo = [p for p in pages if stored.get(p.page_no) != p.fingerprint]
        with_text = [p for p in todo if p.has_text]
        skipped = len(pages) - len(todo)
        total_skipped += skipped
        total_no_text += len(todo) - len(with_text)
        print(
            f"\n{os.path.basename(pdf)}: {len(pages)} pages | {len(todo)} to index "
            f"({len(with_text)} embed, {len(todo) - len(with_text)} text-less) | {skipped} unchanged"
        )

        # Embedding (scaled by network/compute) feeds a background writer (scaled by upload speed).
        # Whatever is already uploaded survives a crash; re-running resumes from there.
        writer = AdaptiveWriter(
            upsert_pages,
            AdaptiveBatch(WRITE_BATCH_START, hi=WRITE_BATCH_MAX, target_s=WRITE_TARGET_S),
        )
        try:
            for p in todo:
                if not p.has_text:
                    writer.put(p, None)
            done, t_doc, last_print = 0, time.time(), time.time()
            for i, vec in embed_stream([p.embed_text for p in with_text]):
                writer.put(with_text[i], vec)
                done += 1
                if time.time() - last_print >= 5:
                    last_print = time.time()
                    rate = done / max(last_print - t_doc, 1e-9)
                    print(f"  {done}/{len(with_text)} embedded | {rate:.1f} pages/s | {status_line()} "
                          f"| saved {writer.written} (batch {writer.batch.size})")
            total_embedded += done
        finally:
            werr = writer.close()   # flush what is buffered, even if embedding failed
        if werr:
            raise werr
        print(f"  saved {writer.written} points ({time.time() - t0:.0f}s elapsed)")

    print(f"\ndone: {total_pages} pages across {len(pdfs)} file(s) in {time.time() - t0:.1f}s")
    print(f"  embedded: {total_embedded}   text-less (no vectors): {total_no_text}   unchanged/skipped: {total_skipped}")
    print(f"  collection now holds {count()} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
