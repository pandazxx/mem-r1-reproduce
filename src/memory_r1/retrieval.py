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
    def embed(self, texts: list[str], *, kind: str = "passage") -> list[list[float]]: ...


class OpenAIEmbedder:
    """OpenAI-compatible embeddings.

    NVIDIA's retrieval NIMs (e.g. nv-embedqa-e5-v5) are asymmetric and
    require an ``input_type`` of "query" or "passage"; set
    ``input_type_param=True`` for those. OpenAI models ignore the kind.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        client=None,
        input_type_param: bool = False,
        batch_size: int = 64,
        throttle=None,
        retry=None,
    ):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client
        self._model = model
        self._input_type_param = input_type_param
        self._batch_size = batch_size
        self._throttle = throttle
        self._retry = retry

    def embed(self, texts: list[str], *, kind: str = "passage") -> list[list[float]]:
        kwargs = {}
        if self._input_type_param:
            kwargs["extra_body"] = {"input_type": kind, "truncate": "END"}
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            if self._throttle:
                self._throttle()

            def call(batch=batch):
                return self._client.embeddings.create(model=self._model, input=batch, **kwargs)

            response = self._retry(call) if self._retry else call()
            vectors.extend(item.embedding for item in response.data)
        return vectors


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
            vectors = self._embedder.embed(missing, kind="passage")
            for text, vector in zip(missing, vectors, strict=True):
                self._cache[text] = vector
        return [self._cache[t] for t in texts]

    def retrieve(self, bank: MemoryBank, query: str, k: int = DEFAULT_TOP_K) -> list[MemoryEntry]:
        entries = bank.entries
        if not entries:
            return []
        query_vec = self._embedder.embed([query], kind="query")[0]
        entry_vecs = self._embed_cached([e.text for e in entries])
        scored = sorted(
            zip(entries, entry_vecs, strict=True),
            key=lambda pair: cosine_similarity(query_vec, pair[1]),
            reverse=True,
        )
        return [entry for entry, _ in scored[:k]]
