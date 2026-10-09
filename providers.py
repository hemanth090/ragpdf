"""Embedding back-ends. All return plain lists of floats.

kind = "local"  : runs on this machine; scaling = batch size (halves on out-of-memory).
kind = "remote" : one HTTP request per call; scaling = concurrent requests (AdaptiveLimiter).
"""

import json
import urllib.request
from functools import lru_cache

from config import (
    EMBED_SPEC,
    LOCAL_BATCH,
    LOCAL_DEVICE,
    MAX_EMBED_CHARS,
    OLLAMA_HOST,
    PROVIDER,
    REQ_BATCH,
    gemini_client,
)


def _clip(t: str) -> str:
    if len(t) > MAX_EMBED_CHARS:
        print(f"  ! embed text truncated {len(t)} -> {MAX_EMBED_CHARS} chars")
        return t[:MAX_EMBED_CHARS]
    return t


# ---------------------------------------------------------------- local (sentence-transformers)
class LocalEmbedder:
    kind = "local"

    def __init__(self, spec: dict):
        self.spec = spec
        self._model = None
        self.device = "cpu"
        self.batch = LOCAL_BATCH or 4

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("pip install sentence-transformers (and torch) for RAGPDF_PROVIDER=local") from e

        device = LOCAL_DEVICE
        if not device:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else (
                    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            except ImportError:
                device = "cpu"
        self.device = device
        if not LOCAL_BATCH:
            self.batch = {"cuda": 32, "mps": 8}.get(device, 4)   # long pages: keep CPU batches small

        print(f"  loading {self.spec['model']} on {device} (first run downloads the weights)...")
        m = SentenceTransformer(self.spec["model"], device=device,
                                trust_remote_code=self.spec.get("trust_remote_code", False))
        m.max_seq_length = self.spec["max_len"]
        real = m.get_sentence_embedding_dimension()
        if real != self.spec["dim"]:
            raise RuntimeError(f"{self.spec['model']} produces {real}-dim vectors but config says "
                               f"{self.spec['dim']}. Set RAGPDF_EMBED_DIM={real}.")
        self._model = m
        return m

    @staticmethod
    def _is_oom(e: Exception) -> bool:
        return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"

    def _free(self):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _encode(self, texts: list[str]) -> list[list[float]]:
        out = self._load().encode(texts, batch_size=self.batch, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)
        return out.tolist()

    def stream(self, texts: list[str]):
        """Yield (index, vector). Auto-halves the batch on out-of-memory and retries."""
        pre = self.spec["doc_prefix"]
        prepared = [pre + _clip(t) for t in texts]
        i = 0
        while i < len(prepared):
            window = prepared[i : i + max(self.batch * 4, 8)]
            try:
                vecs = self._encode(window)
            except Exception as e:
                if self._is_oom(e) and self.batch > 1:
                    self.batch = max(1, self.batch // 2)
                    self._free()
                    print(f"  ! out of memory -> local batch size now {self.batch}")
                    continue
                raise
            for j, v in enumerate(vecs):
                yield i + j, v
            i += len(window)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.spec["query_prefix"] + text])[0]


# ---------------------------------------------------------------- ollama
class OllamaEmbedder:
    kind = "remote"

    def __init__(self, spec: dict):
        self.spec = spec
        self.req_batch = max(1, REQ_BATCH)
        self.url = OLLAMA_HOST.rstrip("/") + "/api/embed"

    def _post(self, inputs: list[str]) -> list[list[float]]:
        body = json.dumps({
            "model": self.spec["model"],
            "input": inputs,
            "truncate": True,
            "options": {"num_ctx": self.spec["max_len"]},
        }).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:   # HTTPError carries .code for retry logic
            data = json.loads(r.read())
        return data["embeddings"]

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        return self._post([self.spec["doc_prefix"] + _clip(t) for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._post([self.spec["query_prefix"] + text])[0]


# ---------------------------------------------------------------- gemini
class GeminiEmbedder:
    kind = "remote"
    req_batch = 1   # the model merges multiple inputs into ONE vector, so never batch

    def __init__(self, spec: dict):
        self.spec = spec

    def _one(self, text: str, task_type: str) -> list[float]:
        from google.genai import types

        res = gemini_client().models.embed_content(
            model=self.spec["model"],
            contents=_clip(text),
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.spec["dim"]),
        )
        return list(res.embeddings[0].values)

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        return [self._one(texts[0], "RETRIEVAL_DOCUMENT")]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text, "RETRIEVAL_QUERY")


@lru_cache(maxsize=1)
def get_embedder():
    return {"local": LocalEmbedder, "ollama": OllamaEmbedder, "gemini": GeminiEmbedder}[PROVIDER](EMBED_SPEC)
