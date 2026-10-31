from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from ..models import Source

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class LocalBM25Tool:
    """In-process BM25 search over a JSONL corpus of ``{"id", "title", "text"}`` lines.

    No server, no embeddings, no extra dependency: a fixed, reproducible
    retriever for benchmarks. Both the baseline and the agent use exactly this
    tool, so any difference in results comes from what they do with retrieval,
    not from the retriever. Okapi BM25 with the usual k1=1.5, b=0.75; the title
    is indexed with the text.
    """

    name = "local_bm25"

    def __init__(self, corpus_path: str | Path, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: list[dict] = []
        for line in Path(corpus_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                self.docs.append(json.loads(line))
        if not self.docs:
            raise ValueError(f"Corpus is empty: {corpus_path}")
        self._tf: list[Counter] = []
        self._len: list[int] = []
        df: Counter = Counter()
        for d in self.docs:
            toks = _tokenize(f"{d.get('title', '')} {d.get('text', '')}")
            tf = Counter(toks)
            self._tf.append(tf)
            self._len.append(len(toks))
            df.update(tf.keys())
        n = len(self.docs)
        self._avg = sum(self._len) / n
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = [t for t in _tokenize(query) if t in self._idf]
        out = []
        for tf, dl in zip(self._tf, self._len):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * dl / self._avg)
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self._idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out

    async def search(self, query: str, *, top_k: int = 5) -> list[Source]:
        sc = self.scores(query)
        ranked = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)[:top_k]
        return [
            Source(title=self.docs[i].get("title"), locator=str(self.docs[i].get("id")), snippet=self.docs[i].get("text"))
            for i in ranked
            if sc[i] > 0
        ]
