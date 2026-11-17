"""In-memory search engine that behaves like Azure AI Search hybrid search.

- keyword: BM25 (same family of ranking Azure uses for full-text search)
- vector: cosine similarity
- hybrid: Reciprocal Rank Fusion (RRF, k=60) of both lists - the same fusion
  method Azure AI Search documents for hybrid queries.
- security trimming: documents are filtered by `allowed_principals` BEFORE
  ranking, equivalent to an OData filter with vectorFilterMode=preFilter.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from securerag.models import Chunk, SearchHit
from securerag.search.base import SearchMode

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its my of on or our "
    "should the their there this to was we what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class LocalSearchBackend:
    RRF_K = 60

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._chunks: list[Chunk] = []
        self._tfs: list[Counter] = []
        self._lens: list[int] = []
        self._df: Counter = Counter()
        self._vectors: np.ndarray | None = None

    # ---------------------------------------------------------------- indexing
    def index(self, chunks: list[Chunk]) -> None:
        existing = {c.chunk_id for c in self._chunks}
        for c in chunks:
            if c.chunk_id in existing:
                continue
            tokens = tokenize(c.content)
            tf = Counter(tokens)
            self._chunks.append(c)
            self._tfs.append(tf)
            self._lens.append(len(tokens))
            self._df.update(tf.keys())
        vecs = [c.vector for c in self._chunks]
        self._vectors = np.array(vecs, dtype=float) if all(v is not None for v in vecs) and vecs else None

    def __len__(self) -> int:
        return len(self._chunks)

    def browse(self, principals: list[str], doc_id: str | None = None) -> list[Chunk]:
        """All chunks the principals may read (optionally of one document), in
        document order. Same permission rule as search: fail closed."""
        if not principals:
            return []
        return [self._chunks[i] for i in self._allowed(principals)
                if doc_id is None or self._chunks[i].doc_id == doc_id]

    # --------------------------------------------------------------- searching
    def _allowed(self, principals: list[str]) -> list[int]:
        p = set(principals)
        return [i for i, c in enumerate(self._chunks) if p.intersection(c.allowed_principals)]

    def _bm25(self, query: str, candidates: list[int]) -> list[tuple[int, float]]:
        terms = tokenize(query)
        if not terms or not candidates:
            return []
        n = len(self._chunks)
        avgdl = (sum(self._lens) / n) if n else 0.0
        scored = []
        for i in candidates:
            tf, dl, s = self._tfs[i], self._lens[i], 0.0
            for t in terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            if s > 0:
                scored.append((i, s))
        return sorted(scored, key=lambda x: -x[1])

    def _cosine(self, vector: list[float] | None, candidates: list[int]) -> list[tuple[int, float]]:
        if vector is None or self._vectors is None or not candidates:
            return []
        q = np.asarray(vector, dtype=float)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self._vectors[candidates] @ q
        return sorted(zip(candidates, sims.tolist()), key=lambda x: -x[1])

    def search(self, query: str, query_vector: list[float] | None, principals: list[str],
               top_k: int = 5, mode: SearchMode = "hybrid") -> list[SearchHit]:
        if not principals:
            return []
        candidates = self._allowed(principals)  # pre-filter: security trimming
        keyword = self._bm25(query, candidates) if mode in ("keyword", "hybrid") else []
        vector = self._cosine(query_vector, candidates) if mode in ("vector", "hybrid") else []
        cos = dict(vector)

        if mode == "keyword":
            ranked = keyword
        elif mode == "vector":
            ranked = vector
        else:
            fused: dict[int, float] = {}
            # Azure retrieves k=50 per leg before fusion; we do the same.
            for results in (keyword[:50], vector[:50]):
                for rank, (i, _) in enumerate(results, start=1):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (self.RRF_K + rank)
            ranked = sorted(fused.items(), key=lambda x: -x[1])

        return [
            SearchHit(chunk=self._chunks[i], score=float(s),
                      relevance=float(max(0.0, cos[i])) if i in cos else None)
            for i, s in ranked[:top_k]
        ]
