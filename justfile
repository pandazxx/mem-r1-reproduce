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

# Precompute top-60 retrieval contexts for train/val QA (resumable)
contexts: data
    uv run python scripts/build_train_contexts.py

# GRPO-train the Answer Agent — GPU box only (uv sync --extra train first)
train-answer *ARGS:
    uv run python scripts/train_grpo_answer_agent.py {{ARGS}}

# Frozen-baseline eval on the validation split (81 QA, ~25 min on NIM)
eval-val *ARGS: data
    uv run python scripts/run_eval.py --split val {{ARGS}}

# Frozen-baseline eval on the full test split (1307 QA, ~3 h on NIM)
eval-test *ARGS: data
    uv run python scripts/run_eval.py --split test {{ARGS}}

# Offline val eval of a local model (GPU box or Apple Silicon; uv sync --extra train)
# e.g. just eval-local --model Qwen/Qwen2.5-3B-Instruct --adapter outputs/grpo-answer-qwen3b
eval-local *ARGS: data
    uv run python scripts/run_eval.py --split val --contexts artifacts/contexts/val.jsonl {{ARGS}}

# Merge the trained LoRA into the base model; --push-repo <user/repo> uploads to HF Hub
export-adapter *ARGS:
    uv run python scripts/export_answer_adapter.py {{ARGS}}

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
