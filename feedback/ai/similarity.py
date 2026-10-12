"""Text similarity for hybrid issue grouping.

* TfidfSimilarity (default): local word unigram+bigram TF-IDF with cosine similarity. Zero cost, no
  network, deterministic. Good enough to flag "this complaint reads like the ones in issue X".
* GeminiEmbeddingProvider (optional, GROUPING_SIMILARITY=gemini_embedding): Gemini
  `batchEmbedContents`; every call goes through the persisted request budget like extraction.
Both produce vectors whose cosine similarity is compared; the business decision (suggest, never
auto-merge) lives in services/grouping.py.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9']+")
STOPWORDS = set("""a an the and or but if of to in on at for with from by is are was were be been it its this that
these those i me my we our you your he she they them their his her im i'm dont don't cant can't wont won't
not no so just very too also do does did have has had will would can could should please hi hey hello thanks
thank spotify spotifycares user link""".split())


def tokens(text: str) -> list[str]:
    words = [w for w in _TOKEN.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class TfidfSimilarity:
    name = "tfidf-v1"

    def vectors(self, texts: list[str]) -> list[dict[str, float]]:
        docs = [Counter(tokens(t)) for t in texts]
        n = len(docs)
        df = Counter(term for d in docs for term in d)
        idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        out = []
        for d in docs:
            v = {t: (1 + math.log(c)) * idf[t] for t, c in d.items()}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            out.append({t: x / norm for t, x in v.items()})
        return out

    @staticmethod
    def cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(x * b.get(t, 0.0) for t, x in a.items())


def dense_cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class GeminiEmbeddingProvider:
    """Gemini embeddings via REST. Requires a budget guard (each batch call = one request)."""

    name = "gemini"
    is_mock = False
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, model: str, timeout: float, guard, transport=None):
        from .client import LLMConfigError
        if not api_key:
            raise LLMConfigError("GEMINI_API_KEY is not set.")
        if guard is None:
            raise LLMConfigError("Embedding requests require a budget guard.")
        import httpx
        self.model, self.guard, self._httpx = model, guard, httpx
        self._http = httpx.Client(base_url=self.BASE, timeout=timeout, headers={"x-goog-api-key": api_key},
                                  transport=transport)

    def embed(self, texts: list[str]) -> list[list[float]]:
        from .client import LLMError, ProviderReply
        import time
        body = {"requests": [{"model": f"models/{self.model}", "content": {"parts": [{"text": t}]}}
                             for t in texts]}
        ticket = self.guard.acquire(self.name, self.model, "embedding",
                                    estimated_input_tokens=sum(len(t) for t in texts) // 3 + 1)
        t0 = time.perf_counter()
        try:
            resp = self._http.post(f"/models/{self.model}:batchEmbedContents", json=body)
            if resp.status_code != 200:
                from .client import GeminiProvider
                raise GeminiProvider._error_from_response(resp)
            vectors = [e["values"] for e in resp.json().get("embeddings", [])]
            if len(vectors) != len(texts):
                raise LLMError("invalid_json", "Embedding count does not match input count.")
        except LLMError as exc:
            self.guard.finish(ticket, ok=False, latency_ms=int((time.perf_counter() - t0) * 1000), error=exc)
            raise
        except self._httpx.HTTPError as exc:
            err = LLMError("connection_error", type(exc).__name__)
            self.guard.finish(ticket, ok=False, latency_ms=int((time.perf_counter() - t0) * 1000), error=err)
            raise err from None
        self.guard.finish(ticket, ok=True, latency_ms=int((time.perf_counter() - t0) * 1000),
                          reply=ProviderReply("", sum(len(t) for t in texts) // 4, 0, None))
        return vectors
