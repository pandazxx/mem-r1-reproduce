"""LLM/embedding provider configuration.

Default provider is NVIDIA NIM (build.nvidia.com free tier, OpenAI-compatible,
~40 RPM): $0 for bootstrap, judging, embeddings, and frozen-baseline inference.
OpenAI (GPT-4o-mini, the paper's choice) stays available as a fallback for the
final paper-comparable judge run.

Select via MEMR1_PROVIDER env var or an explicit ``get_provider("openai")``.
Model defaults can be overridden with MEMR1_CHAT_MODEL / MEMR1_EMBEDDING_MODEL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from memory_r1.bootstrap import LLMFn
from memory_r1.retrieval import OpenAIEmbedder


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str | None
    api_key_env: str
    chat_model: str
    embedding_model: str
    embedding_input_type: bool = False


NIM = Provider(
    name="nim",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key_env="NVIDIA_API_KEY",
    chat_model="meta/llama-3.1-8b-instruct",
    # QA-retrieval-tuned; asymmetric, so requires input_type query/passage.
    # (baai/bge-m3 is listed on the endpoint but 500s as of 2026-07.)
    embedding_model="nvidia/nv-embedqa-e5-v5",
    embedding_input_type=True,
)

OPENAI = Provider(
    name="openai",
    base_url=None,
    api_key_env="OPENAI_API_KEY",
    chat_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
)

PROVIDERS = {p.name: p for p in (NIM, OPENAI)}
DEFAULT_PROVIDER = NIM.name


def get_provider(name: str | None = None) -> Provider:
    name = name or os.environ.get("MEMR1_PROVIDER", DEFAULT_PROVIDER)
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; choose from {sorted(PROVIDERS)}")
    return PROVIDERS[name]


def make_client(provider: Provider):
    from openai import OpenAI

    api_key = os.environ.get(provider.api_key_env)
    if not api_key:
        raise RuntimeError(f"provider {provider.name!r} requires {provider.api_key_env} to be set")
    return OpenAI(base_url=provider.base_url, api_key=api_key)


def make_llm(provider: Provider | None = None, model: str | None = None, client=None) -> LLMFn:
    provider = provider or get_provider()
    model = model or os.environ.get("MEMR1_CHAT_MODEL") or provider.chat_model
    if client is None:
        client = make_client(provider)

    def complete(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    return complete


def make_embedder(
    provider: Provider | None = None, model: str | None = None, client=None
) -> OpenAIEmbedder:
    provider = provider or get_provider()
    model = model or os.environ.get("MEMR1_EMBEDDING_MODEL") or provider.embedding_model
    if client is None:
        client = make_client(provider)
    return OpenAIEmbedder(
        model=model, client=client, input_type_param=provider.embedding_input_type
    )
