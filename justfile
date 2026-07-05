# Project tasks. Install just: https://github.com/casey/just (or `uv tool install rust-just`)

default:
    @just --list

# Install/sync the Python environment
setup:
    uv sync

# Run the fast, offline test suite
test:
    uv run pytest -q

# Lint (with autofix) and format
lint:
    uv run ruff check --fix .
    uv run ruff format .

# Lint + tests — run before every commit
check: lint test

# Download LoCoMo benchmark data to data/
data:
    uv run python scripts/download_locomo.py

# Build memory banks for all conversations via the API provider (resumable)
banks: data
    uv run python scripts/build_memory_banks.py

# Live smoke test of the configured provider (chat + embeddings + retrieval)
smoke:
    uv run python -c "\
    from memory_r1.providers import get_provider, make_llm, make_embedder; \
    from memory_r1.retrieval import Retriever; \
    from memory_r1.memory_bank import MemoryBank; \
    p = get_provider(); print('provider:', p.name); \
    print('chat:', make_llm(p)('Reply with exactly the word: pong')); \
    b = MemoryBank(); b.add('John has a dog named Rex'); b.add('John lives in Berlin'); \
    print('retrieval:', [e.text for e in Retriever(make_embedder(p)).retrieve(b, 'What pet does John have?', k=1)])"
