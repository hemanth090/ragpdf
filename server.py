"""ragpdf server: HTTP query console over the indexed PDFs.

  GET  /                  -> static/index.html (query console)
  GET  /api/health        -> {"ok", "collection", "points", "provider", ...}
  GET  /api/docs          -> {"docs": [{doc_id, source, pages, chars}]}
  POST /api/classify      -> {"clear", "reason", "suggestions"} (Jev gate)
  POST /api/ask           -> {"answer", "cited", "shown", "hits", "ok"} (blocking)
  GET  /api/ask/stream?q= -> Server-Sent Events: start -> hits -> pages -> token* -> done
  GET  /img/<token>        -> cached PNG for a rendered page

Only stdlib (http.server). Run:  python3 server.py [--port 8787]
Env: QDRANT_PIN_IP=1 bypasses flaky local DNS for *.qdrant.io (see config.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"

# Rate limiting: token bucket per IP for the expensive endpoints
# (classify/ask/upload). Generous defaults; tune via env.
_RATE = {"max": int(os.environ.get("RAGPDF_RL_MAX", "30")),
         "window": float(os.environ.get("RAGPDF_RL_WINDOW", "60"))}
_BUCKETS: dict[str, list[float]] = {}
_BUCKETS_LOCK = threading.Lock()


def _rate_ok(ip: str) -> bool:
    now = time.time()
    with _BUCKETS_LOCK:
        hits = [t for t in _BUCKETS.get(ip, []) if now - t < _RATE["window"]]
        if len(hits) >= _RATE["max"]:
            _BUCKETS[ip] = hits
            return False
        _BUCKETS[ip] = hits + [now]
        return True


# Concurrency: cap simultaneous vision (VLM) calls. Extra ask requests wait
# briefly, then get a 503 instead of dogpiling the quota.
_VLM_SLOTS = threading.BoundedSemaphore(
    int(os.environ.get("RAGPDF_MAX_VLM", "3")))

_IMG_TOKENS: dict[str, str] = {}  # token -> absolute png path
_JOBS: dict[str, dict] = {}       # job_id -> ingest progress


def _parse_multipart(raw: bytes, boundary: bytes) -> tuple[str, bytes]:
    """Extract (filename, content) of the first file part. Minimal parser."""
    for part in raw.split(b"--" + boundary):
        if b'filename="' not in part:
            continue
        head, _, body = part.partition(b"\r\n\r\n")
        fname = head.split(b'filename="')[1].split(b'"')[0].decode("utf-8", "replace")
        body = body.rsplit(b"\r\n", 1)[0]
        return fname, body
    return "", b""


def _ingest_job(job_id: str, pdf_path: str):
    """Background ingest for one uploaded PDF. Updates _JOBS as it goes."""
    import config
    from store import (count, delete_stale_pages, ensure_collection,
                       existing_hashes, upsert_pages)
    from extract import extract
    from embeddings import embed_stream, status_line
    from autoscale import AdaptiveBatch
    from writer import AdaptiveWriter

    job = _JOBS[job_id]
    try:
        ensure_collection()
        pages = extract(pdf_path)
        job.update(status="extracted", total=len(pages), detail=f"{len(pages)} pages")
        stored = existing_hashes(pages[0].source) if pages else {}
        todo = [p for p in pages if stored.get(p.page_no) != p.fingerprint]
        with_text = [p for p in todo if p.has_text]
        if pages:
            delete_stale_pages(pages[0].source, pages[0].page_count)
        skipped = len(pages) - len(todo)
        job.update(status="embedding", total=len(with_text),
                   detail=f"{skipped} unchanged, {len(with_text)} to embed")
        writer = AdaptiveWriter(
            upsert_pages,
            AdaptiveBatch(config.WRITE_BATCH_START, hi=config.WRITE_BATCH_MAX,
                          target_s=config.WRITE_TARGET_S))
        try:
            for p in todo:
                if not p.has_text:
                    writer.put(p, None)
            done = 0
            for i, vec in embed_stream([p.embed_text for p in with_text]):
                writer.put(with_text[i], vec)
                done += 1
                if done % 5 == 0 or done == len(with_text):
                    job.update(done=done,
                               detail=f"{done}/{len(with_text)} embedded | {status_line()}")
        finally:
            werr = writer.close()
        if werr:
            raise werr
        from docs import list_documents
        list_documents(force=True)  # bust the registry cache
        job.update(status="done", done=len(with_text),
                   detail=f"indexed {len(pages)} pages, {count()} total points")
    except Exception as e:
        job.update(status="error", detail=f"{type(e).__name__}: {str(e)[:300]}")


def _img_token(png_path: str) -> str:
    tok = hashlib.sha1(png_path.encode()).hexdigest()[:16]
    _IMG_TOKENS[tok] = png_path
    return tok


def _ask_pipeline(question, log, limit=None, pages=None, adjacent=None,
                  max_render=None, doc_id=None, source=None):
    """Run the ask.py pipeline, calling log(event: dict) at each stage."""
    import config
    from retrieve import adjacent_pages, dedupe, search
    from render import render_pages
    from vlm import answer as vlm_answer

    limit = config.RETRIEVE_LIMIT if limit is None else limit
    pages = config.VLM_TOP_PAGES if pages is None else pages
    adjacent = config.ADJACENT_PAGES if adjacent is None else adjacent
    max_render = config.MAX_RENDER_PAGES if max_render is None else max_render

    log({"type": "stage", "stage": "retrieve", "message": "Searching the index…"})
    hits = search(question, doc_id=doc_id or None, source=source or None, limit=limit)
    if not hits:
        raise RuntimeError("no hits in the index for this question")
    hits = dedupe(hits)
    log({"type": "hits", "hits": [
        {"doc_id": h.doc_id, "source": os.path.basename(h.source),
         "page_no": h.page_no, "score": round(h.score, 4),
         "chars": h.char_count} for h in hits[:limit]]})

    top = hits[:pages]
    neighbours = []
    if adjacent:
        neighbours = [h for h in adjacent_pages(top, depth=adjacent) if h.is_neighbor]
        neighbours.sort(key=lambda h: -h.score)
    candidates = (top + neighbours)[: max(max_render, 1)]

    by_source: dict[str, list[int]] = {}
    doc_of: dict[str, str] = {}
    for h in sorted(candidates, key=lambda h: (h.is_neighbor, -h.score)):
        by_source.setdefault(h.source, []).append(h.page_no)
        doc_of[h.source] = h.doc_id

    log({"type": "stage", "stage": "render",
         "message": f"Rendering {sum(len(set(v)) for v in by_source.values())} pages…"})
    rendered = []  # (doc_id, source, page_no, png_path)
    for src, nums in by_source.items():
        for n, path in render_pages(src, sorted(set(nums))):
            rendered.append((doc_of[src], src, n, path))
    if not rendered:
        raise RuntimeError("no pages could be rendered")
    log({"type": "pages", "pages": [
        {"doc_id": d, "source": os.path.basename(s), "page_no": n,
         "img": _img_token(path)} for d, s, n, path in rendered]})

    log({"type": "stage", "stage": "answer",
         "message": "Reading the pages with the vision model…"})
    if not _VLM_SLOTS.acquire(timeout=10):
        raise RuntimeError("server is busy — retry in a few seconds")
    try:
        res = vlm_answer(question, [(d, n, p) for d, _, n, p in rendered])
    finally:
        _VLM_SLOTS.release()

    text = res.get("text", "")
    words = text.split(" ")
    for i in range(0, len(words), 8):
        log({"type": "token",
             "text": " ".join(words[i:i + 8]) + (" " if i + 8 < len(words) else "")})
        time.sleep(0.02)

    return {
        "ok": res.get("ok", False),
        "answer": text,
        "cited": [{"document": d, "page": p} for d, p in res.get("cited", [])],
        "shown": [{"document": d, "page": p} for d, p in sorted(res.get("shown", []))],
        "hits": [{"doc_id": h.doc_id, "source": os.path.basename(h.source),
                  "page_no": h.page_no, "score": round(h.score, 4)}
                 for h in hits[:limit]],
    }

class Handler(BaseHTTPRequestHandler):
    server_version = "ragpdf/1.0"

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _route_static(self, path):
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (STATIC / rel).resolve()
        if not str(target).startswith(str(STATIC.resolve())) or not target.is_file():
            return False
        ctype = {".html": "text/html; charset=utf-8", ".css": "text/css",
                 ".js": "text/javascript"}.get(target.suffix, "application/octet-stream")
        self._send(200, target.read_bytes(), ctype)
        return True

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path, qs = url.path, urllib.parse.parse_qs(url.query)
        if path == "/api/health":
            try:
                import config
                client = config.qdrant_client()
                pts = client.count(config.COLLECTION, exact=True).count
                from docs import list_documents
                ndocs = len(list_documents())
                self._json(200, {"ok": True, "collection": config.COLLECTION,
                                 "points": pts, "docs": ndocs,
                                 "provider": config.EMBED_MODEL,
                                 "vlm": config.VLM_MODEL,
                                 "version": "2.0"})
            except Exception as e:
                self._json(503, {"ok": False, "error": str(e)[:300]})
            return
        if path == "/api/docs":
            try:
                from docs import list_documents
                docs = list_documents()
                short = [{**d, "source": os.path.basename(d["source"])} for d in docs]
                self._json(200, {"ok": True, "docs": short, "count": len(short)})
            except Exception as e:
                self._json(503, {"ok": False, "error": str(e)[:300]})
            return
        if path == "/api/ask/stream":
            q = (qs.get("q") or [""])[0]
            if not q.strip():
                self._json(400, {"ok": False, "error": "missing ?q="})
                return
            self._sse(q, qs)
            return
        if path.startswith("/img/"):
            tok = path[len("/img/"):]
            png = _IMG_TOKENS.get(tok)
            if png and os.path.exists(png):
                self._send(200, Path(png).read_bytes(), "image/png")
            else:
                self._json(404, {"ok": False, "error": "unknown image token"})
            return
        if path == "/api/jobs":
            jobs = sorted(_JOBS.values(), key=lambda j: j.get("at", 0), reverse=True)[:20]
            self._json(200, {"ok": True, "jobs": jobs})
            return
        if path.startswith("/api/jobs/"):
            job = _JOBS.get(path[len("/api/jobs/"):])
            if job:
                self._json(200, {"ok": True, **job})
            else:
                self._json(404, {"ok": False, "error": "unknown job"})
            return
        if self._route_static(path):
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path in ("/api/classify", "/api/ask", "/api/upload"):
            if not _rate_ok(self.client_address[0]):
                self._json(429, {"ok": False,
                                 "error": "rate limited — slow down and retry"})
                return
        if url.path == "/api/upload":
            self._handle_upload()
            return
        if url.path == "/api/classify":
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                self._json(400, {"ok": False, "error": "invalid JSON body"})
                return
            q = str(body.get("question") or body.get("q") or "").strip()
            if not q:
                self._json(400, {"ok": False, "error": "missing 'question'"})
                return
            try:
                from jev import classify
                from docs import list_documents
                try:
                    docs = list_documents()
                    ctx = "Documents: " + ", ".join(
                        f"{d['doc_id']} ({d['pages']} pages)" for d in docs) + "."
                except Exception:
                    ctx = ""
                if body.get("doc_id"):
                    ctx += f" Searching within: {body['doc_id']}."
                res = classify(q, ctx)
                self._json(200, {"ok": True, **res})
            except Exception as e:
                self._json(502, {"ok": False,
                                 "error": f"{type(e).__name__}: {str(e)[:300]}"})
            return
        if url.path != "/api/ask":
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._json(400, {"ok": False, "error": "invalid JSON body"})
            return
        q = str(body.get("question") or body.get("q") or "").strip()
        if not q:
            self._json(400, {"ok": False, "error": "missing 'question'"})
            return
        events: list[dict] = []
        try:
            t0 = time.time()
            payload = _ask_pipeline(q, events.append,
                                    limit=body.get("limit"), pages=body.get("pages"),
                                    adjacent=body.get("adjacent"),
                                    max_render=body.get("max_render"),
                                    doc_id=body.get("doc_id"), source=body.get("source"))
            payload["elapsed_s"] = round(time.time() - t0, 1)
            self._json(200, payload)
        except Exception as e:
            self._json(502, {"ok": False,
                             "error": f"{type(e).__name__}: {str(e)[:500]}",
                             "events": events})

    def _sse(self, question, qs):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def emit(obj):
            try:
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise StopIteration

        def num(key):
            try:
                v = (qs.get(key) or [None])[0]
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        try:
            emit({"type": "start", "question": question})
            t0 = time.time()
            # Clamp expensive knobs so one request can't hog the VLM.
            def capped(key, cap):
                v = num(key)
                return max(1, min(v, cap)) if v is not None else None
            payload = _ask_pipeline(
                question, emit,
                limit=capped("limit", 50), pages=capped("pages", 8),
                adjacent=num("adjacent"), max_render=capped("max_render", 12),
                doc_id=(qs.get("doc_id") or [None])[0],
                source=(qs.get("source") or [None])[0])
            payload["elapsed_s"] = round(time.time() - t0, 1)
            emit({"type": "done", **payload})
        except StopIteration:
            pass
        except Exception as e:
            try:
                emit({"type": "error",
                      "error": f"{type(e).__name__}: {str(e)[:500]}"})
            except StopIteration:
                pass

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    # -- upload + background ingest -------------------------------------
    def _handle_upload(self):
        import threading
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype or "boundary=" not in ctype:
            self._json(400, {"ok": False,
                             "error": "send multipart/form-data with a 'file' field"})
            return
        try:
            boundary = ctype.split("boundary=")[1].strip().strip('"').encode()
            n = int(self.headers.get("Content-Length") or 0)
            if n > 200 * 1024 * 1024:
                self._json(413, {"ok": False, "error": "file too large (max 200MB)"})
                return
            raw = self.rfile.read(n)
            fname, pdf = _parse_multipart(raw, boundary)
            if not pdf or not fname.lower().endswith(".pdf"):
                self._json(400, {"ok": False, "error": "need a 'file' field with a .pdf"})
                return
        except Exception as e:
            self._json(400, {"ok": False, "error": f"bad upload: {e}"})
            return
        safe = "".join(c for c in os.path.basename(fname) if c.isalnum() or c in "._-")[:120]
        uploads = BASE / "uploads"
        uploads.mkdir(exist_ok=True)
        dest = uploads / safe
        i = 1
        while dest.exists():
            dest = uploads / f"{dest.stem}-{i}{dest.suffix}"
            i += 1
        dest.write_bytes(pdf)
        job_id = hashlib.sha1(f"{dest}:{time.time()}".encode()).hexdigest()[:12]
        _JOBS[job_id] = {"id": job_id, "file": dest.name, "status": "queued",
                         "total": 0, "done": 0, "detail": "", "at": time.time()}
        threading.Thread(target=_ingest_job, args=(job_id, str(dest)),
                         daemon=True).start()
        self._json(202, {"ok": True, "job_id": job_id, "file": dest.name})


def main():
    import argparse
    ap = argparse.ArgumentParser(description="ragpdf query server")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8787")))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    if not STATIC.joinpath("index.html").exists():
        print(f"WARNING: {STATIC}/index.html missing — API only.")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ragpdf server on http://{args.host}:{args.port}  (static: {STATIC})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
