# ragpdf on Python 3.11 (poppler for pdftotext/pdftoppm, torch CPU for embeddings)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY *.py ./
COPY static/ ./static/

ENV PORT=8787 \
    RAGPDF_CACHE=/app/cache \
    HF_HOME=/app/cache/hf \
    SENTENCE_TRANSFORMERS_HOME=/app/cache/st

VOLUME ["/app/cache", "/app/uploads"]
EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8787/api/health || exit 1

CMD ["python", "-u", "server.py", "--port", "8787", "--host", "0.0.0.0"]
