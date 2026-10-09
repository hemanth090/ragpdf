"""Auto-scaling controllers.

AdaptiveLimiter  - how many embedding requests may be in flight (remote providers).
AdaptiveBatch    - how many points go into one Qdrant upsert.

Both follow the same idea as TCP congestion control (AIMD):
  * everything is going fine  -> grow slowly (additive increase)
  * rate limit / timeout / slow responses -> shrink fast (multiplicative decrease)
so throughput climbs until the network, the API quota or the server pushes back,
then settles just below that point, and keeps probing as conditions change.
"""

import threading
import time
from contextlib import contextmanager


class _Slot:
    """Set .outcome inside `with limiter.slot() as s:` to one of
    "ok" | "throttled" (HTTP 429) | "error" (timeout / 5xx / network) | "neutral" (ignore)."""

    __slots__ = ("outcome",)

    def __init__(self):
        self.outcome = "neutral"


class AdaptiveLimiter:
    def __init__(self, start: int = 2, lo: int = 1, hi: int = 16,
                 rate_hz: float = 0.0, latency_factor: float = 3.0):
        self.lo, self.hi = lo, hi
        self._limit = float(min(max(start, lo), hi))
        self._inflight = 0
        self._cond = threading.Condition()
        self._pause_until = 0.0        # global cool-down after a 429
        self._consec_throttle = 0
        self._acks = 0                 # successes since the last change
        self._last_decrease = 0.0
        self._fail_at = 0              # limit at which we last got throttled
        self._fail_time = 0.0
        self._base = None              # best-case latency (slowly drifts up)
        self._ewma = None              # recent latency
        self._factor = latency_factor  # ewma > base*factor => path is congested
        self._min_gap = 1.0 / rate_hz if rate_hz > 0 else 0.0
        self._next_start = 0.0
        self.throttles = 0

    @property
    def limit(self) -> int:
        return max(self.lo, int(self._limit))

    @property
    def inflight(self) -> int:
        return self._inflight

    def acquire(self) -> float:
        with self._cond:
            while True:
                now = time.monotonic()
                if now < self._pause_until:
                    self._cond.wait(self._pause_until - now)
                    continue
                if self._inflight < self.limit:
                    break
                self._cond.wait(0.5)
            self._inflight += 1
            if self._min_gap:
                start = max(now, self._next_start)
                self._next_start = start + self._min_gap
            else:
                start = now
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        return time.monotonic()

    def _decrease(self, factor: float, now: float) -> bool:
        """Shrink, but at most once per ~2 round-trips so one burst of failures counts once."""
        cooldown = min(max(0.05, 2 * self._ewma), 5.0) if self._ewma else 1.0
        if now - self._last_decrease < cooldown:
            return False
        self._limit = max(float(self.lo), self._limit * factor)
        self._last_decrease = now
        self._acks = 0
        return True

    def release(self, t0: float, outcome: str) -> None:
        now = time.monotonic()
        dt = now - t0
        with self._cond:
            self._inflight -= 1
            if outcome == "ok":
                self._consec_throttle = 0
                self._ewma = dt if self._ewma is None else 0.8 * self._ewma + 0.2 * dt
                self._base = dt if self._base is None else min(dt, self._base * 1.01)
                if self._ewma > self._base * self._factor:
                    self._decrease(0.9, now)             # latency blow-up = early congestion signal
                else:
                    self._acks += 1
                    # Near the level that got us throttled, probe 4x more slowly (like TCP
                    # congestion avoidance); the memory fades after a minute.
                    near_ceiling = self._fail_at and self.limit >= self._fail_at - 1 and now - self._fail_time < 60
                    if self._acks >= max(self.limit, 1) * (4 if near_ceiling else 1):
                        self._limit = min(float(self.hi), self._limit + 1)
                        self._acks = 0
            elif outcome == "throttled":
                self.throttles += 1
                before = self.limit
                if self._decrease(0.5, now):
                    self._fail_at, self._fail_time = before, now
                    self._consec_throttle += 1
                    pause = min(0.5 * 2 ** (self._consec_throttle - 1), 20.0)
                    self._pause_until = max(self._pause_until, now + pause)
            elif outcome == "error":
                self._decrease(0.7, now)
            self._cond.notify_all()

    @contextmanager
    def slot(self):
        t0 = self.acquire()
        s = _Slot()
        try:
            yield s
        finally:
            self.release(t0, s.outcome)


class AdaptiveBatch:
    """Qdrant upsert batch size: grow while uploads are quick, shrink when slow or failing."""

    def __init__(self, start: int = 32, lo: int = 4, hi: int = 256, target_s: float = 2.0):
        self.lo, self.hi, self.target = lo, hi, target_s
        self._size = min(max(start, lo), hi)

    @property
    def size(self) -> int:
        return self._size

    def record(self, n: int, seconds: float, ok: bool) -> None:
        if not ok:
            self._size = max(self.lo, self._size // 2)
        elif seconds > self.target:
            self._size = max(self.lo, int(self._size * 0.7))
        elif seconds < self.target / 2 and n >= self._size:
            self._size = min(self.hi, int(self._size * 1.5) + 1)
