"""Document registry: distinct documents in the collection, with page counts.

Backed by a scroll over the collection (cheap: payload only, no vectors).
Results are cached in-memory for DOCS_TTL_S so the UI picker stays fast.
"""
from __future__ import annotations

import os
import time

_DOCS_CACHE: dict = {"at": 0.0, "docs": []}
DOCS_TTL_S = float(os.environ.get("RAGPDF_DOCS_TTL", "60"))


def list_documents(force: bool = False) -> list[dict]:
    """Return [{doc_id, source, pages, chars}] sorted by doc_id."""
    from config import COLLECTION, qdrant_client

    now = time.time()
    if not force and now - _DOCS_CACHE["at"] < DOCS_TTL_S and _DOCS_CACHE["docs"]:
        return _DOCS_CACHE["docs"]

    client = qdrant_client()
    agg: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=512,
            offset=offset,
            with_payload=["doc_id", "source", "page_no", "char_count"],
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            doc = pl.get("doc_id", "?")
            e = agg.setdefault(doc, {"doc_id": doc, "source": pl.get("source", ""),
                                     "pages": 0, "chars": 0})
            e["pages"] += 1
            e["chars"] += pl.get("char_count", 0) or 0
            # prefer the longest source string (absolute path) if doc_id repeats
            if len(pl.get("source", "")) > len(e["source"]):
                e["source"] = pl["source"]
        if offset is None:
            break

    docs = sorted(agg.values(), key=lambda d: d["doc_id"].lower())
    _DOCS_CACHE.update(at=now, docs=docs)
    return docs
