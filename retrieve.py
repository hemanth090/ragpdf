"""Hybrid retrieval: dense + sparse prefetch, fused by Qdrant's RRF.

RRF is delegated to Qdrant rather than reimplemented. Hand-rolled RRF needs
score normalization to work at all, and the docs are explicit that an
unnormalized alpha-weighted blend of cosine and BM25 gets dominated by
whichever retriever has larger raw magnitudes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from qdrant_client import models

from config import (
    ADJACENT_PAGES,
    BM25_MODEL,
    COLLECTION,
    DENSE_VECTOR,
    RETRIEVE_LIMIT,
    SPARSE_VECTOR,
    qdrant_client,
)


@dataclass
class Hit:
    doc_id: str
    source: str
    page_no: int
    has_text: bool
    score: float
    char_count: int
    page_count: int = 0
    is_neighbor: bool = False  # True for pages added by adjacent_pages()


def _filter(doc_id: str | None, source: str | None, has_text: bool | None):
    must = []
    if doc_id:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
    if source:
        # payload stores absolute paths, so normalise whatever the user typed
        must.append(models.FieldCondition(key="source", match=models.MatchValue(value=os.path.abspath(source))))
    if has_text is not None:
        must.append(models.FieldCondition(key="has_text", match=models.MatchValue(value=has_text)))
    return models.Filter(must=must) if must else None


def search(
    query: str,
    doc_id: str | None = None,
    source: str | None = None,
    limit: int = RETRIEVE_LIMIT,
    rrf_k: int = 2,
) -> list[Hit]:
    from embeddings import embed_query

    client = qdrant_client()
    flt = _filter(doc_id, source, None)

    dense = embed_query(query)

    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=models.Document(text=query, model=BM25_MODEL),
                using=SPARSE_VECTOR,
                limit=limit,
                filter=flt,
            ),
            models.Prefetch(
                query=dense,
                using=DENSE_VECTOR,
                limit=limit,
                filter=flt,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=rrf_k)),
        limit=limit,
        with_payload=True,
    )

    return [
        Hit(
            doc_id=p.payload["doc_id"],
            source=p.payload["source"],
            page_no=p.payload["page_no"],
            has_text=p.payload.get("has_text", False),
            score=p.score,
            char_count=p.payload.get("char_count", 0),
            page_count=p.payload.get("page_count", 0),
        )
        for p in results.points
    ]


def dedupe(hits: list[Hit]) -> list[Hit]:
    """Cheap safety net; point IDs are unique per page so this is normally a no-op."""
    seen, out = set(), []
    for h in hits:
        if (h.source, h.page_no) in seen:
            continue
        seen.add((h.source, h.page_no))
        out.append(h)
    return out


def adjacent_pages(hits: list[Hit], depth: int = ADJACENT_PAGES) -> list[Hit]:
    """Widen around strong hits so split tables/notes stay contiguous.

    Returns hits + neighbours. Neighbours are flagged is_neighbor=True and
    scored at half the parent per step of distance, so callers can rank them
    below real hits.
    """
    out = list(hits)
    have = {(h.source, h.page_no) for h in out}
    totals: dict[str, int] = {}

    for h in hits:
        total = h.page_count or totals.get(h.source, 0)
        if not total:
            try:
                from pypdf import PdfReader

                total = totals[h.source] = len(PdfReader(h.source).pages)
            except Exception:
                continue
        for d in range(1, depth + 1):
            for p in (h.page_no - d, h.page_no + d):
                if 1 <= p <= total and (h.source, p) not in have:
                    have.add((h.source, p))
                    out.append(
                        Hit(
                            doc_id=h.doc_id,
                            source=h.source,
                            page_no=p,
                            has_text=True,
                            score=h.score * (0.5 ** d),
                            char_count=0,
                            page_count=total,
                            is_neighbor=True,
                        )
                    )
    return out
