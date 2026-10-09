"""Small shared helpers for classifying API errors."""
from __future__ import annotations

import re

_CODE = re.compile(r"\b([45]\d\d)\b")


def http_status(exc: Exception) -> int | None:
    """Best-effort HTTP status from a google-genai / httpx exception."""
    for attr in ("code", "status_code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    s = str(exc)
    if "RESOURCE_EXHAUSTED" in s:
        return 429
    if any(w in s for w in ("UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL")):
        return 503
    m = _CODE.search(s)
    return int(m.group(1)) if m else None


def is_retryable(status: int | None) -> bool:
    """429 and 5xx are transient. Other 4xx (bad key, bad request) never fix themselves.
    Unknown status (network blip) is treated as retryable."""
    if status is None:
        return True
    return status == 429 or status >= 500
