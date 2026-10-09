"""PDF -> one Page per physical page. No splitting, no merging, no image chunks."""

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from pypdf import PdfReader

from config import EMBED_DIM, EMBED_MODEL, MIN_TEXT_CHARS


def clean_text(text: str) -> str:
    """Collapse the padding that `pdftotext -layout` adds. Fewer junk tokens,
    and the 28k-char embedding guard trips far less often."""
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class Page:
    doc_id: str
    source: str
    page_no: int
    page_count: int
    text: str
    char_count: int
    has_text: bool

    @property
    def embed_text(self) -> str:
        """Dense text: includes a header so the vector knows which doc/page it is."""
        head = f"[document: {self.doc_id} | page {self.page_no} of {self.page_count}]"
        return f"{head}\n{self.text}"

    @property
    def sparse_text(self) -> str:
        """BM25 text: doc name for doc-name queries, but no 'page'/'of' noise words."""
        return f"{self.doc_id}\n{self.text}"

    @property
    def fingerprint(self) -> str:
        """Changes when anything that affects the stored point changes."""
        h = hashlib.sha1()
        for part in (self.doc_id, EMBED_MODEL, str(EMBED_DIM), str(self.has_text), self.text):
            h.update(part.encode("utf-8", "replace"))
            h.update(b"\x00")
        return h.hexdigest()


def doc_id_for(path: str) -> str:
    """Human-readable id: stem cleaned to [A-Za-z0-9._-], capped at 64 chars.

    Falls back to a short hash when the stem is empty or a bare hash already.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-_.")[:64]
    if not slug or re.fullmatch(r"[0-9a-f-]{16,}", slug):
        h = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:8]
        slug = (slug[:48] + "-" + h) if slug else f"doc-{h}"
    return slug


def extract(pdf_path: str) -> list[Page]:
    """One subprocess for the whole document; split on form feed.

    Calling pdftotext per page would cost one process spawn per page. The
    form-feed split preserves exact page boundaries.
    """
    if shutil.which("pdftotext") is None:
        raise RuntimeError("pdftotext not found. Install poppler-utils (apt) / poppler (brew).")

    reader = PdfReader(pdf_path)
    if reader.is_encrypted and reader.decrypt("") == 0:
        raise RuntimeError(f"{pdf_path} is password-protected; decrypt it first.")
    page_count = len(reader.pages)

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "doc.txt")
        try:
            subprocess.run(
                ["pdftotext", "-layout", pdf_path, out],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            msg = (e.stderr or b"").decode("utf-8", "replace").strip()
            raise RuntimeError(f"pdftotext failed on {pdf_path}: {msg}") from e
        with open(out, encoding="utf-8", errors="replace") as f:
            chunks = f.read().split("\f")

    # pdftotext emits a trailing empty chunk after the final form feed.
    if chunks and not chunks[-1].strip():
        chunks.pop()

    source = os.path.abspath(pdf_path)
    doc_id = doc_id_for(pdf_path)

    pages = []
    for i in range(page_count):
        text = clean_text(chunks[i]) if i < len(chunks) else ""
        pages.append(
            Page(
                doc_id=doc_id,
                source=source,
                page_no=i + 1,
                page_count=page_count,
                text=text,
                char_count=len(text),
                has_text=len(text) >= MIN_TEXT_CHARS,
            )
        )
    return pages


def extract_many(paths: list[str]) -> list[Page]:
    out = []
    for p in paths:
        out.extend(extract(p))
    return out
