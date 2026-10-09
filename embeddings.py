"""Embedding front door: embed_query() and embed_stream().

Local provider  -> batched on this machine, batch size self-adjusts to memory.
Remote provider -> many requests in parallel; AdaptiveLimiter scales the number in
                   flight up while the network / API is healthy and cuts it on
                   429s, timeouts and latency spikes.
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from autoscale import AdaptiveLimiter
from config import EMBED_MAX_WORKERS, EMBED_START_WORKERS, RATE_HZ
from providers import get_embedder
from util import http_status, is_retryable

_limiter = AdaptiveLimiter(start=EMBED_START_WORKERS, hi=EMBED_MAX_WORKERS, rate_hz=RATE_HZ)


def _with_retry(fn, attempts: int = 6):
    """Run one remote request under the limiter; retry only transient failures."""
    last = None
    for i in range(attempts):
        with _limiter.slot() as s:
            try:
                out = fn()
                s.outcome = "ok"
                return out
            except Exception as e:
                last = e
                status = http_status(e)
                if not is_retryable(status):
                    s.outcome = "neutral"
                    raise RuntimeError(f"embedding failed (HTTP {status}, not retryable): {str(e)[:200]}") from e
                s.outcome = "throttled" if status == 429 else "error"
        if i < attempts - 1:
            time.sleep(min(0.5 * 2 ** i + random.random() * 0.5, 15))
    raise RuntimeError(f"embedding failed after {attempts} attempts: {str(last)[:200]}")


def embed_query(text: str) -> list[float]:
    emb = get_embedder()
    if emb.kind == "local":
        return emb.embed_query(text)
    return _with_retry(lambda: emb.embed_query(text))


def embed_stream(texts: list[str]):
    """Yield (index, vector) as results become ready (order not guaranteed)."""
    if not texts:
        return
    emb = get_embedder()
    if emb.kind == "local":
        yield from emb.stream(texts)
        return

    rb = emb.req_batch
    chunks = [(s, texts[s : s + rb]) for s in range(0, len(texts), rb)]
    pool = ThreadPoolExecutor(max_workers=_limiter.hi)   # threads wait on the limiter, not on the pool
    futs = {pool.submit(_with_retry, lambda c=c: emb.embed_docs(c)): s for s, c in chunks}
    try:
        for f in as_completed(futs):
            start = futs[f]
            for j, v in enumerate(f.result()):
                yield start + j, v
    finally:
        pool.shutdown(wait=False, cancel_futures=True)   # on error/Ctrl-C: stop queuing new work


def status_line() -> str:
    emb = get_embedder()
    if emb.kind == "local":
        return f"{emb.device} batch={emb.batch}"
    return f"requests in flight={_limiter.inflight}/limit {_limiter.limit} (max {_limiter.hi}) 429s={_limiter.throttles}"
