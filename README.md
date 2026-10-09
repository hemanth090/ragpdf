# ragpdf — ask any PDF, cite the page

Drop in annual reports, contracts, papers, manuals — ask questions, get
answers read off the rendered page image by a vision model, with citations.
No chunking guesswork: one Qdrant point per page, hybrid retrieval, refusal
instead of hallucination.

```
PDFs → page index (dense + BM25 → RRF) → top pages rendered → VLM reads → cited answer
```

## 60-second start

```bash
pip install -r requirements.txt   # + poppler-utils for pdftotext/pdftoppm
cp .env.example .env             # QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY
python ingest.py ./docs          # any PDF or folder; re-runs skip unchanged pages
python server.py --port 8787     # open http://127.0.0.1:8787
```

Or `docker compose up --build` (needs `.env`).

## Why this, not another RAG demo

- **Page-level, not chunks.** No splitter to tune, no cross-chunk stitching.
  Tables survive because they are never split.
- **RRF done right.** Fusion is delegated to Qdrant (`Rrf(k=2)`), not a
hand-rolled alpha blend that one retriever dominates.
- **Vision answers, not snippet pasting.** The LLM never sees extracted text
— it reads page PNGs and cites `(document, page N)`. Checked against what
it was shown; uncited answers are flagged, missing evidence is refused.
- **Jev gate.** A classifier asks clarifying questions for vague queries
("what was the profit?" → which metric, which year?) before burning
retrieval + vision calls. Zero-cost heuristic first, LLM second, fail-open.
- **Crash-safe ingest.** Content-hash fingerprints + background writer:
kill it mid-run, re-run, it resumes. Adaptive batching on both embed and
upload sides (AIMD, like TCP congestion control).

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | minimal query console (SSE streaming) |
| GET | `/api/health` | `{ok, collection, points, docs, provider, vlm}` |
| GET | `/api/docs` | indexed documents + page counts |
| POST | `/api/classify` | Jev: `{clear, reason, suggestions[], via}` |
| POST | `/api/ask` | blocking ask → `{answer, cited, shown, hits}` |
| GET | `/api/ask/stream?q=` | SSE: `start → hits → pages → token* → done` |
| POST | `/api/upload` | PDF upload → background ingest `{job_id}` |
| GET | `/api/jobs[/<id>]` | ingest progress |

Knobs: `limit ≤ 50`, `pages ≤ 8`, `max_render ≤ 12`, `adjacent`, `doc_id`.

## Architecture

```
static/        zero-dependency console (HTML/CSS/JS, SSE client)
server.py      stdlib HTTP: static + JSON/SSE API, rate limit, VLM semaphore,
               upload + background ingest jobs
jev.py         ambiguity classifier (regex heuristic → Gemini, fail-open)
ingest.py       CLI ingest: folder → extract → embed → upsert (resumable)
extract.py      pdftotext page-split, slugified doc_ids, content fingerprints
retrieve.py    dense + BM25 prefetch, Qdrant RRF, neighbour expansion
vlm.py         Gemini reads page images → JSON {answer, citations, sufficient}
store.py       Qdrant lifecycle, payload indexes, incremental helpers
docs.py        document registry (scroll + aggregate, cached)
providers.py   embeddings: local (BGE-M3/Qwen/nomic) · ollama · gemini
embeddings.py  embed_query / embed_stream + retry + adaptive limiter
autoscale.py   AIMD controllers for concurrency and batch size
writer.py      background Qdrant writer with retry
render.py      pdftoppm page → cached PNG (query time only)
```

## Production notes

- Stateless HTTP layer — scale replicas behind a LB; Qdrant is the shared
state. `cache/` + `uploads/` shared (NFS) or per-replica (recomputed).
- Guards: 30 req/min/IP, max 3 concurrent vision calls, 200MB upload cap.
- Each embedding model gets its own collection (`ragpdf_<model>_<dim>`).
- Gemini free tier is 20 req/day — Jev's heuristic keeps the gate useful
with zero API spend; bring a paid key (or self-hosted VLM) for real load.

## License

MIT.
