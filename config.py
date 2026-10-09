from __future__ import annotations
import os
import re
import uuid
from functools import lru_cache
from pathlib import Path

BASE = Path(__file__).resolve().parent


def _load_env() -> None:
    """Minimal .env loader. MUST run before the settings below are read."""
    env = BASE / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]  # strip matching quotes: KEY="abc" -> abc
        os.environ.setdefault(k, v)


_load_env()  # <-- was at the bottom of the file, so .env overrides were ignored

# --------------------------------------------------------------------------
# Embedding provider
#   local  : sentence-transformers on this machine (free, no network, default)
#   ollama : local Ollama server (free; also works against a remote Ollama box)
#   gemini : Google API (rate-limited remote)
# --------------------------------------------------------------------------
PROVIDER = os.environ.get("RAGPDF_PROVIDER", "local").lower()

_QWEN_Q = "Instruct: Given a question, retrieve the document page that answers it\nQuery: "

LOCAL_PRESETS = {
    # MIT, 8k context, multilingual. Safest all-rounder.
    "bge-m3":     dict(model="BAAI/bge-m3", dim=1024, max_len=4096),
    # Apache-2.0, strongest open family. 0.6B is laptop-friendly, 4B wants a GPU.
    "qwen3-0.6b": dict(model="Qwen/Qwen3-Embedding-0.6B", dim=1024, max_len=4096, query_prefix=_QWEN_Q),
    "qwen3-4b":   dict(model="Qwen/Qwen3-Embedding-4B", dim=2560, max_len=4096, query_prefix=_QWEN_Q),
    # Small and fast on CPU, 8k context, English-focused.
    "nomic":      dict(model="nomic-ai/nomic-embed-text-v1.5", dim=768, max_len=4096,
                       doc_prefix="search_document: ", query_prefix="search_query: ", trust_remote_code=True),
}

OLLAMA_PRESETS = {
    "bge-m3":               dict(dim=1024),
    "qwen3-embedding:0.6b": dict(dim=1024, query_prefix=_QWEN_Q),
    "qwen3-embedding:4b":   dict(dim=2560, query_prefix=_QWEN_Q),
    "nomic-embed-text":     dict(dim=768, doc_prefix="search_document: ", query_prefix="search_query: "),
}


def _resolve_spec() -> dict:
    base = dict(model="", dim=None, max_len=4096, doc_prefix="", query_prefix="", trust_remote_code=False)
    if PROVIDER == "gemini":
        spec = dict(base, model=os.environ.get("RAGPDF_EMBED_MODEL", "gemini-embedding-2"), dim=1536)
    elif PROVIDER == "local":
        key = os.environ.get("RAGPDF_LOCAL_MODEL", "bge-m3")
        spec = dict(base, **(LOCAL_PRESETS.get(key) or dict(model=key)))  # unknown key = any HF model id
    elif PROVIDER == "ollama":
        key = os.environ.get("RAGPDF_OLLAMA_MODEL", "bge-m3")
        spec = dict(base, model=key, **OLLAMA_PRESETS.get(key, {}))
    else:
        raise RuntimeError(f"RAGPDF_PROVIDER must be local, ollama or gemini (got {PROVIDER!r})")

    if os.environ.get("RAGPDF_EMBED_DIM"):
        spec["dim"] = int(os.environ["RAGPDF_EMBED_DIM"])
    if "RAGPDF_DOC_PREFIX" in os.environ:
        spec["doc_prefix"] = os.environ["RAGPDF_DOC_PREFIX"]
    if "RAGPDF_QUERY_PREFIX" in os.environ:
        spec["query_prefix"] = os.environ["RAGPDF_QUERY_PREFIX"]
    if not spec["dim"]:
        raise RuntimeError(f"Unknown model {spec['model']!r}: set RAGPDF_EMBED_DIM (and prefixes if it needs any).")
    return spec


EMBED_SPEC = _resolve_spec()
EMBED_MODEL = f"{PROVIDER}:{EMBED_SPEC['model']}"   # stored in payload + part of the page fingerprint
EMBED_DIM = EMBED_SPEC["dim"]

# Collection name encodes model + dim, so switching models never collides with old vectors.
# (Set RAGPDF_COLLECTION=ragpdf_pages_v1 to keep using an older Gemini-built collection.)
_slug = re.sub(r"[^a-z0-9]+", "_", EMBED_MODEL.lower()).strip("_")
COLLECTION = os.environ.get("RAGPDF_COLLECTION", f"ragpdf_{_slug}_{EMBED_DIM}")

# "-latest" is a moving alias; set RAGPDF_VLM_MODEL to a pinned model for reproducible answers.
VLM_MODEL = os.environ.get("RAGPDF_VLM_MODEL", "gemini-flash-latest")

BM25_MODEL = "qdrant/bm25"
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

RENDER_DPI = int(os.environ.get("RAGPDF_DPI", "200"))
CACHE_DIR = Path(os.environ.get("RAGPDF_CACHE", BASE / "cache"))

RETRIEVE_LIMIT = int(os.environ.get("RAGPDF_LIMIT", "20"))
VLM_TOP_PAGES = int(os.environ.get("RAGPDF_VLM_PAGES", "4"))    # best-scoring pages
ADJACENT_PAGES = int(os.environ.get("RAGPDF_ADJACENT", "1"))    # neighbours added around them
# Hard cap on images sent to the VLM. Real hits are kept first, then the best neighbours.
MAX_RENDER_PAGES = int(os.environ.get("RAGPDF_MAX_RENDER", "8"))

MAX_EMBED_CHARS = 28000
MIN_TEXT_CHARS = 50

# --------------------------------------------------------------------------
# Auto-scaling (see autoscale.py)
# --------------------------------------------------------------------------
# Remote providers (gemini / ollama): concurrent requests are scaled between these bounds.
EMBED_START_WORKERS = int(os.environ.get("RAGPDF_START_WORKERS", "2"))
EMBED_MAX_WORKERS = int(os.environ.get("RAGPDF_MAX_WORKERS", "16"))
RATE_HZ = float(os.environ.get("RAGPDF_RATE_HZ", "0"))          # optional hard cap, 0 = off
REQ_BATCH = int(os.environ.get("RAGPDF_REQ_BATCH", "4"))        # texts per request (ollama only)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# Local provider: starting batch (auto if unset); halves itself on out-of-memory.
LOCAL_BATCH = int(os.environ["RAGPDF_LOCAL_BATCH"]) if os.environ.get("RAGPDF_LOCAL_BATCH") else None
LOCAL_DEVICE = os.environ.get("RAGPDF_DEVICE")                  # cuda | mps | cpu (auto if unset)
# Qdrant writes: batch size grows while uploads are fast, shrinks when slow or failing.
WRITE_BATCH_START = int(os.environ.get("RAGPDF_WRITE_BATCH", "32"))
WRITE_BATCH_MAX = int(os.environ.get("RAGPDF_WRITE_BATCH_MAX", "256"))
WRITE_TARGET_S = float(os.environ.get("RAGPDF_WRITE_TARGET_S", "2.0"))


@lru_cache(maxsize=1)
def qdrant_client():
    from qdrant_client import QdrantClient

    # Hard-pin the Qdrant IP when local DNS is flaky (QDRANT_PIN_IP=1).
    # TLS/SNI still uses the real hostname, so certs validate normally.
    if os.environ.get("QDRANT_PIN_IP") == "1":
        import socket as _socket

        _real_getaddrinfo = _socket.getaddrinfo

        def _pinned(host, port, *a, **k):
            if "qdrant.io" in str(host):
                for ip in ("15.156.10.57", "99.79.27.66", "16.54.160.227"):
                    try:
                        return _real_getaddrinfo(ip, port, *a, **k)
                    except OSError:
                        continue
            return _real_getaddrinfo(host, port, *a, **k)

        _socket.getaddrinfo = _pinned

    url = os.environ.get("QDRANT_URL")
    key = os.environ.get("QDRANT_API_KEY")
    if not url:
        raise RuntimeError("QDRANT_URL missing from .env")
    return QdrantClient(url=url, api_key=key, timeout=60)


@lru_cache(maxsize=1)
def gemini_client():
    from google.genai import Client

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing from .env")
    return Client(api_key=key)


def page_point_id(source: str, page_no: int) -> str:
    ns = uuid.NAMESPACE_URL
    return str(uuid.uuid5(ns, f"ragpdf::{os.path.abspath(source)}::{page_no}"))
