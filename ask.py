"""CLI: question -> retrieved pages -> VLM -> cited answer."""

import argparse
import os
import sys

from config import ADJACENT_PAGES, MAX_RENDER_PAGES, RETRIEVE_LIMIT, VLM_TOP_PAGES
from render import render_pages
from retrieve import adjacent_pages, dedupe, search
from vlm import answer


def main() -> int:
    ap = argparse.ArgumentParser(description="Ask a question about indexed PDFs.")
    ap.add_argument("question")
    ap.add_argument("--doc-id", help="Restrict to one document")
    ap.add_argument("--source", help="Restrict to one PDF path")
    ap.add_argument("--limit", type=int, default=RETRIEVE_LIMIT, help="Hybrid candidates to retrieve")
    ap.add_argument("--pages", type=int, default=VLM_TOP_PAGES, help="Best-scoring pages to use")
    ap.add_argument("--adjacent", type=int, default=ADJACENT_PAGES, help="Neighbouring pages to add around them")
    ap.add_argument("--max-render", type=int, default=MAX_RENDER_PAGES,
                    help="Max images sent to the VLM (best hits first, then best neighbours)")
    ap.add_argument("--show-hits", action="store_true")
    args = ap.parse_args()

    hits = search(args.question, doc_id=args.doc_id, source=args.source, limit=args.limit)
    if not hits:
        print("no hits")
        return 1

    hits = dedupe(hits)
    if args.show_hits:
        print(f"query: {args.question}\n")
        for h in hits[: args.limit]:
            print(f"  {h.score:.4f}  {os.path.basename(h.source)} p{h.page_no:<4} text={h.char_count}")

    # Real hits always come first; neighbours fill whatever budget is left, best first.
    top = hits[: args.pages]
    neighbours = []
    if args.adjacent:
        neighbours = [h for h in adjacent_pages(top, depth=args.adjacent) if h.is_neighbor]
        neighbours.sort(key=lambda h: -h.score)
    candidates = (top + neighbours)[: max(args.max_render, 1)]

    # Group by PDF (best-scoring PDF first), pages in reading order within each PDF.
    by_source: dict[str, list[int]] = {}
    doc_of: dict[str, str] = {}
    for h in sorted(candidates, key=lambda h: (h.is_neighbor, -h.score)):
        by_source.setdefault(h.source, []).append(h.page_no)
        doc_of[h.source] = h.doc_id

    rendered = []  # (doc_id, page_no, png_path)
    for source, nums in by_source.items():
        for n, path in render_pages(source, sorted(set(nums))):
            rendered.append((doc_of[source], n, path))

    if not rendered:
        print("INSUFFICIENT EVIDENCE: no pages could be rendered.")
        return 1

    print(f"\nrendered pages: {[(d, p) for d, p, _ in rendered]}")

    res = answer(args.question, rendered)
    print("\n" + "=" * 70)
    print(res["text"])
    if res["cited"]:
        print("\nsources: " + ", ".join(f"{d} p{p}" for d, p in res["cited"]))
    print("=" * 70)
    return 0 if res["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
