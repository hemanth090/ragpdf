"""Render specific PDF pages to PNG. Called at query time only, never at ingest."""

import hashlib
import os
import shutil
import subprocess

from config import CACHE_DIR, RENDER_DPI


def render_page(pdf_path: str, page_no: int, dpi: int = RENDER_DPI) -> str:
    """Return a PNG path for one 1-indexed page. Cached by path+mtime+page+dpi.

    Renders straight from the original PDF with pdftoppm -f/-l. (The old
    pypdf single-page-PDF detour could drop fonts/resources and was slower.)
    """
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("pdftoppm not found. Install poppler-utils (apt) / poppler (brew).")

    abspath = os.path.abspath(pdf_path)
    mtime = os.path.getmtime(abspath)
    tag = hashlib.sha1(f"{abspath}:{mtime}:{page_no}:{dpi}".encode()).hexdigest()[:16]
    out_dir = CACHE_DIR / os.path.basename(abspath)
    stem = f"p{page_no:05d}_{tag}"
    png = out_dir / f"{stem}.png"
    if png.exists():
        return str(png)

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["pdftoppm", "-f", str(page_no), "-l", str(page_no), "-r", str(dpi),
             "-png", "-singlefile", abspath, str(out_dir / stem)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"pdftoppm failed for page {page_no}: {msg}") from e

    if not png.exists():
        raise RuntimeError(f"render produced no file for page {page_no}")
    return str(png)


def render_pages(pdf_path: str, page_nos: list[int], dpi: int = RENDER_DPI) -> list[tuple[int, str]]:
    """Page-number keyed so the VLM can cite each one it was shown."""
    out = []
    for n in page_nos:
        try:
            out.append((n, render_page(pdf_path, n, dpi)))
        except Exception as e:
            print(f"  ! render failed page {n}: {e}")
    return out
