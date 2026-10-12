"""Text similarity for hybrid issue grouping.

TfidfSimilarity: local word unigram+bigram TF-IDF with cosine similarity. Zero cost, no network,
deterministic. Good enough to flag "this complaint reads like the ones in issue X". The business
decision (suggest, never auto-merge) lives in services/grouping.py.
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
