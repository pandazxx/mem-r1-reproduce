"""Similarity-based retrieval over a memory bank.

The paper follows Mem0-style RAG (~60 candidates per question) but never
names an embedding model; we default to OpenAI text-embedding-3-small,
Mem0's default. Embedder is a protocol so tests can use a fake.
"""

from __future__ import annotations

import math
from typing import Protocol

from memory_r1.memory_bank import MemoryBank, MemoryEntry

DEFAULT_TOP_K = 60


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if norm == 0 else dot / norm


class Retriever:
    """Top-k cosine retrieval with an embedding cache keyed by entry text."""

    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._cache: dict[str, list[float]] = {}

    def _embed_cached(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for text, vector in zip(missing, self._embedder.embed(missing), strict=True):
                self._cache[text] = vector
        return [self._cache[t] for t in texts]

    def retrieve(self, bank: MemoryBank, query: str, k: int = DEFAULT_TOP_K) -> list[MemoryEntry]:
        entries = bank.entries
        if not entries:
            return []
        query_vec = self._embedder.embed([query])[0]
        entry_vecs = self._embed_cached([e.text for e in entries])
        scored = sorted(
            zip(entries, entry_vecs, strict=True),
            key=lambda pair: cosine_similarity(query_vec, pair[1]),
            reverse=True,
        )
        return [entry for entry, _ in scored[:k]]
