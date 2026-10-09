"""Gemini VLM over rendered pages. Prompt is built to refuse rather than guess.

The model returns structured JSON (answer + citations + sufficient flag) and
every citation is checked against the pages it was actually shown.
"""
from __future__ import annotations

import json
import random
import time

from config import VLM_MODEL, gemini_client
from util import http_status, is_retryable

SYSTEM = """You answer questions about documents by reading page images.
Each image is preceded by a label like [document: NAME | page N].

Rules:
1. Read content directly off the page image. Layouts may be dense, rotated, or use
   small type — figures, tables, diagrams, and prose alike. Read carefully.
2. Cite every claim inline as (document, page N), and list each page you used in
   `citations` using the exact document name and page number from the labels.
3. If the pages do not contain the answer, set sufficient=false and write: INSUFFICIENT EVIDENCE
   followed by what is missing. Do not estimate, interpolate, or use outside knowledge.
4. If content is ambiguous, say so explicitly and state what you can read.
5. Quote figures, names, and dates exactly as printed, with their stated units or context.

Respond with JSON only: {"sufficient": bool, "answer": str, "citations": [{"document": str, "page": int}]}"""

_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "sufficient": {"type": "BOOLEAN"},
        "answer": {"type": "STRING"},
        "citations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"document": {"type": "STRING"}, "page": {"type": "INTEGER"}},
                "required": ["document", "page"],
            },
        },
    },
    "required": ["sufficient", "answer", "citations"],
}


def _parts(pages: list[tuple[str, int, str]], question: str):
    """pages = [(doc_id, page_no, png_path)]. The label goes BEFORE its image."""
    from google.genai import types

    parts = []
    for doc, page_no, path in pages:
        with open(path, "rb") as f:
            img = f.read()
        parts.append(types.Part.from_text(text=f"[document: {doc} | page {page_no}]"))
        parts.append(types.Part.from_bytes(data=img, mime_type="image/png"))
    parts.append(
        types.Part.from_text(
            text=f"Question: {question}\n\nAnswer using only the page images above."
        )
    )
    return parts


def _config():
    from google.genai import types

    kwargs = dict(
        system_instruction=SYSTEM,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=_SCHEMA,
    )
    try:  # higher image resolution helps with dense tables; not all SDK versions have it
        return types.GenerateContentConfig(
            **kwargs, media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH
        )
    except Exception:
        return types.GenerateContentConfig(**kwargs)


def check_citations(data: dict, shown: set[tuple[str, int]]):
    """Split the model's citations into (valid, invalid) against what it was shown."""
    shown_lc = {(d.lower(), p): (d, p) for d, p in shown}
    valid, invalid = [], []
    for c in data.get("citations") or []:
        try:
            key = (str(c["document"]).strip().lower(), int(c["page"]))
        except (KeyError, TypeError, ValueError):
            invalid.append(c)
            continue
        if key in shown_lc:
            if shown_lc[key] not in valid:
                valid.append(shown_lc[key])
        else:
            invalid.append(c)
    return sorted(valid), invalid


def _generate(client, contents, attempts: int = 4):
    last = None
    for i in range(attempts):
        try:
            return client.models.generate_content(
                model=VLM_MODEL, contents=contents, config=_config()
            )
        except Exception as e:
            last = e
            if not is_retryable(http_status(e)) or i == attempts - 1:
                raise
            time.sleep(min(2 ** i * 2 + random.random(), 30))
    raise last  # unreachable


def answer(question: str, pages: list[tuple[str, int, str]]) -> dict:
    """pages = [(doc_id, page_no, png_path)]"""
    if not pages:
        return {"text": "INSUFFICIENT EVIDENCE: no pages could be rendered for this question.",
                "cited": [], "invalid": [], "shown": [], "ok": False}

    shown = {(d, p) for d, p, _ in pages}
    try:
        res = _generate(gemini_client(), [_parts(pages, question)])
        data = json.loads(res.text or "{}")
        if not isinstance(data, dict):
            raise ValueError("model did not return a JSON object")
    except Exception as e:
        return {"text": f"VLM error: {e}", "cited": [], "invalid": [],
                "shown": sorted(shown), "ok": False}

    text = str(data.get("answer", "")).strip()
    sufficient = bool(data.get("sufficient", False))
    cited, invalid = check_citations(data, shown)

    ok = sufficient and bool(cited)
    if sufficient and not cited:
        text += "\n\n[warning: the answer has no valid page citations; do not trust it without checking]"
    if invalid:
        text += f"\n\n[warning: ignored citations to pages that were not shown: {invalid}]"

    return {"text": text, "cited": cited, "invalid": invalid,
            "shown": sorted(shown), "sufficient": sufficient, "ok": ok}
