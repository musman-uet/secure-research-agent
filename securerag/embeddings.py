"""Embedding providers.

- AzureOpenAIEmbedder: production (e.g. text-embedding-3-small, 1536 dims).
- HashingEmbedder: deterministic, dependency-light, offline. Used for local
  runs and tests so the pipeline needs no network or API key. It captures
  word and character n-gram overlap, not true semantics, so local scores are
  a lower bound of what real embeddings give.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dimensions: int = 512):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dimensions = dimensions
        self._word = HashingVectorizer(
            n_features=dimensions, ngram_range=(1, 2), stop_words="english",
            alternate_sign=False, norm=None,
        )
        self._char = HashingVectorizer(
            n_features=dimensions, analyzer="char_wb", ngram_range=(3, 5),
            alternate_sign=False, norm=None,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        m = self._word.transform(texts).toarray() + 0.5 * self._char.transform(texts).toarray()
        m = np.log1p(m)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (m / norms).tolist()


class AzureOpenAIEmbedder:
    def __init__(self, endpoint: str, api_key: str, deployment: str,
                 api_version: str = "2024-10-21", dimensions: int = 1536, batch_size: int = 64):
        from openai import AzureOpenAI

        self._client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        self._deployment = deployment
        self.dimensions = dimensions
        self._batch = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            resp = self._client.embeddings.create(model=self._deployment, input=texts[i:i + self._batch])
            out.extend(d.embedding for d in resp.data)
        return out
