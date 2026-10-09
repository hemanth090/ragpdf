"""Qdrant collection lifecycle, upsert, delete, metadata filters."""
from __future__ import annotations

from qdrant_client import models

from config import (
    BM25_MODEL,
    COLLECTION,
    DENSE_VECTOR,
    EMBED_DIM,
    EMBED_MODEL,
    SPARSE_VECTOR,
    page_point_id,
    qdrant_client,
)

_INDEXES = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "source": models.PayloadSchemaType.KEYWORD,   # was missing: needed for --source / delete / scroll
    "emb_model": models.PayloadSchemaType.KEYWORD,
    "has_text": models.PayloadSchemaType.BOOL,
    "page_no": models.PayloadSchemaType.INTEGER,  # needed to drop stale pages by range
}


def _ensure_indexes(client) -> None:
    """Create any missing payload index (also upgrades collections made by older versions)."""
    existing = set((client.get_collection(COLLECTION).payload_schema or {}).keys())
    for field, schema in _INDEXES.items():
        if field not in existing:
            client.create_payload_index(
                collection_name=COLLECTION, field_name=field, field_schema=schema
            )


def ensure_collection(client=None, recreate: bool = False) -> None:
    client = client or qdrant_client()

    if recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=EMBED_DIM, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                    modifier=models.Modifier.IDF,
                )
            },
        )
    _check_dim(client)
    _ensure_indexes(client)


def _check_dim(client) -> None:
    """Fail early (and clearly) if the collection was built with a different embedding size."""
    try:
        cfg = client.get_collection(COLLECTION).config.params.vectors
        size = cfg[DENSE_VECTOR].size if isinstance(cfg, dict) else cfg.size
    except Exception:
        return
    if isinstance(size, int) and size != EMBED_DIM:
        raise RuntimeError(
            f"Collection {COLLECTION!r} stores {size}-dim vectors but the current embedding model "
            f"({EMBED_MODEL}) makes {EMBED_DIM}-dim ones. Use a different RAGPDF_COLLECTION, "
            f"or re-ingest with --recreate."
        )


def make_point(page, dense: list[float] | None):
    """Payload always present so text-less pages stay addressable.

    has_text=False pages carry no vectors (nothing to embed). `vector` is a
    required field on PointStruct, so an empty dict is passed rather than
    omitting it. (Verify once against your qdrant-client version.)
    """
    payload = {
        "doc_id": page.doc_id,
        "source": page.source,
        "page_no": page.page_no,
        "page_count": page.page_count,
        "char_count": page.char_count,
        "has_text": page.has_text,
        "emb_model": EMBED_MODEL,
        "content_hash": page.fingerprint,
    }
    pid = page_point_id(page.source, page.page_no)

    if not page.has_text or dense is None:
        return models.PointStruct(id=pid, vector={}, payload=payload)

    return models.PointStruct(
        id=pid,
        vector={
            DENSE_VECTOR: dense,
            SPARSE_VECTOR: models.Document(text=page.sparse_text, model=BM25_MODEL),
        },
        payload=payload,
    )


def upsert_pages(pages, dense_vectors: list[list[float] | None]) -> int:
    """Upsert exactly these pages in ONE request (collection must already exist).
    Batch sizing is handled by writer.AdaptiveWriter. Returns number written."""
    points = [make_point(p, d) for p, d in zip(pages, dense_vectors)]
    if points:
        qdrant_client().upsert(COLLECTION, points=points, wait=True)
    return len(points)


def _source_filter(source: str, extra: list | None = None) -> "models.Filter":
    must = [models.FieldCondition(key="source", match=models.MatchValue(value=source))]
    return models.Filter(must=must + (extra or []))


def existing_hashes(source: str) -> dict[int, str]:
    """{page_no: content_hash} already stored for this PDF (for incremental ingest)."""
    client = qdrant_client()
    out, offset = {}, None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=_source_filter(source),
            limit=256,
            offset=offset,
            with_payload=["page_no", "content_hash"],
            with_vectors=False,
        )
        for p in points:
            out[p.payload["page_no"]] = p.payload.get("content_hash", "")
        if offset is None:
            return out


def delete_stale_pages(source: str, page_count: int) -> None:
    """Remove pages beyond the end of a PDF that got shorter since last ingest."""
    qdrant_client().delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(
            filter=_source_filter(
                source,
                [models.FieldCondition(key="page_no", range=models.Range(gt=page_count))],
            )
        ),
        wait=True,
    )


def delete_document(source: str) -> int:
    import os

    source = os.path.abspath(source)
    before = count(source=source)
    qdrant_client().delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(filter=_source_filter(source)),
        wait=True,
    )
    return before


def count(source: str | None = None, doc_id: str | None = None) -> int:
    client = qdrant_client()
    must = []
    if source:
        must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
    if doc_id:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
    return client.count(
        collection_name=COLLECTION,
        count_filter=models.Filter(must=must) if must else None,
        exact=True,
    ).count
