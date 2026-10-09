"""Background Qdrant writer: embedding and uploading overlap, upload batch size auto-scales."""
from __future__ import annotations

import queue
import threading
import time

from autoscale import AdaptiveBatch

_DONE = object()
MAX_BUFFER_AGE_S = 10.0   # slow embedders still get their progress saved regularly


class AdaptiveWriter:
    def __init__(self, upsert_fn, batch: AdaptiveBatch, queue_size: int = 1024, attempts: int = 4):
        self._upsert = upsert_fn
        self.batch = batch
        self.written = 0
        self.error: Exception | None = None
        self._attempts = attempts
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)   # full queue = backpressure on the embedder
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def put(self, page, vec) -> None:
        if self.error:
            raise self.error
        self._q.put((page, vec))

    def close(self) -> Exception | None:
        """Flush everything still buffered, wait for the thread, return the error (if any)."""
        self._q.put(_DONE)
        self._t.join()
        return self.error

    def _flush(self, pages: list, vecs: list) -> None:
        n = len(pages)
        for attempt in range(self._attempts):
            t0 = time.monotonic()
            try:
                self._upsert(pages, vecs)
            except Exception as e:
                self.batch.record(n, time.monotonic() - t0, ok=False)
                if attempt == self._attempts - 1:
                    self.error = e
                    return
                time.sleep(min(2 ** attempt, 15))
            else:
                self.batch.record(n, time.monotonic() - t0, ok=True)
                self.written += n
                return

    def _run(self) -> None:
        pages, vecs, last_flush = [], [], time.monotonic()
        while True:
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                item = None
            if item is _DONE:
                break
            if self.error:
                continue                      # keep draining so the producer never deadlocks
            if item is not None:
                pages.append(item[0])
                vecs.append(item[1])
            due = len(pages) >= self.batch.size or (pages and time.monotonic() - last_flush > MAX_BUFFER_AGE_S)
            if due:
                self._flush(pages, vecs)
                pages, vecs, last_flush = [], [], time.monotonic()
        if pages and not self.error:
            self._flush(pages, vecs)
