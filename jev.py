"""Query classifier ("Jev"): clear vs ambiguous, before any retrieval.

Ambiguous = the question admits materially different readings that would send
retrieval at different pages (missing entity, period, scope, or comparison leg).
A question that is merely broad ("summarize the risk section") is still clear —
it maps to one place. A question missing the thing being asked about is not.

Returns {"clear": bool, "reason": str, "suggestions": [str, ...]}.
Suggestions are 2-3 concrete rewrites the user can tap instead of typing.
"""
from __future__ import annotations

import json
import re

_PRONOUNS = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|he|she|him|her|his)\b",
    re.IGNORECASE)
_BARE_METRIC = re.compile(
    r"^(what('s| is)?|show|give|tell)\b.{0,40}\b"
    r"(profit|revenue|margin|dividend|income|loss|growth|ebitda|sales|turnover)\b"
    r"\??$",
    re.IGNORECASE)
_COMPARATIVE_DANGLING = re.compile(
    r"\b(vs|versus|compared to|against|than last|over last|change|growth|increase|decrease)\b",
    re.IGNORECASE)


def _heuristic(question: str) -> dict | None:
    """Cheap local gate. Returns a verdict or None (defer to the LLM)."""
    q = (question or "").strip()
    words = q.split()
    if len(words) < 2:
        return {"clear": False, "reason": "That is too short to search on.",
                "suggestions": [], "via": "heuristic"}
    low = q.lower()
    if low in ("tell me more", "more", "why", "how come", "explain",
               "what about it", "and then", "continue"):
        return {"clear": False, "reason": "There is nothing to search on yet.",
                "suggestions": [], "via": "heuristic"}
    if _PRONOUNS.search(q) and len(words) <= 7:
        return {"clear": False,
                "reason": "It is unclear what that refers to.",
                "suggestions": [], "via": "heuristic"}
    if _BARE_METRIC.match(q.strip()):
        return {"clear": False,
                "reason": "That metric appears in several places — which one?",
                "suggestions": [], "via": "heuristic"}
    return None

_CLASSIFY_SYSTEM = """You are Jev, a query classifier for a document Q&A system.
The index holds one or more long PDF documents (annual reports, contracts,
papers, manuals). Retrieval is page-level hybrid search; a vision model then
reads the top pages.

Classify the user's question as CLEAR or AMBIGUOUS.

AMBIGUOUS when the question admits materially different readings that would
retrieve different pages:
- missing entity: "what was the revenue?" when several segments/entities exist
- missing period: "what was the profit/revenue?" without a year/quarter — financial
  documents always carry multiple periods, so a bare metric with no period is AMBIGUOUS
- dangling comparison: "how did it change vs last year?" without saying what "it" is
- pronoun with no antecedent: "tell me more about it", "what about them?"
- scope split: could mean a summary table OR a detailed note in another section
- bare metric + no scope: "what is the dividend?", "what was the margin?" without
  saying which kind (interim/final, standalone/consolidated, which segment)

CLEAR only when the question names what it wants precisely enough to pull one place:
- names an entity + period/section ("revenue from operations FY2025",
  "final dividend for FY2025", "standalone vs consolidated PAT FY2025")
- asks about a unique thing ("who is the chairman?", "when is the AGM?")
- broad but single-target ("summarize the risk factors", "list the directors")

Respond with JSON only:
{"clear": bool, "reason": str, "suggestions": [str, ...]}
- reason: one short sentence, plain words, no jargon.
- suggestions: if AMBIGUOUS, 2-3 concrete rewrites naming the likely readings
  (use the document/period names from context when given). If CLEAR, [].
- Keep each suggestion under 90 characters so it fits a button."""


def classify(question: str, doc_context: str = "") -> dict:
    """Classify a question. Never raises — on failure returns clear=True.

    doc_context: e.g. "Documents: AIAENG-FY25 (321 pages)." Helps ground
    suggestions without a retrieval call.
    """
    from config import VLM_MODEL, gemini_client

    fallback = {"clear": True, "reason": "", "suggestions": [], "via": "llm-fallback"}
    q = (question or "").strip()
    fast = _heuristic(q)
    if fast is not None:
        return fast
    try:
        from google.genai import types
        user = f"Question: {q}"
        if doc_context:
            user = f"Context: {doc_context}\n{user}"
        res = gemini_client().models.generate_content(
            model=VLM_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=_CLASSIFY_SYSTEM,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        data = json.loads(res.text or "{}")
        if not isinstance(data, dict) or "clear" not in data:
            return fallback
        out = dict(fallback)
        out["clear"] = bool(data.get("clear", True))
        out["reason"] = str(data.get("reason", ""))[:200]
        sug = data.get("suggestions") or []
        out["suggestions"] = [str(s)[:120] for s in sug[:3] if str(s).strip()]
        out["via"] = "llm"
        if not out["clear"] and not out["suggestions"]:
            # ambiguous but no rewrites offered — still gate, UI shows retry
            pass
        return out
    except Exception:
        return fallback
