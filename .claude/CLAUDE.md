# mem-r1-reproduce — Project Instructions

Reproduction of **Memory-R1** (arXiv:2508.19828): RL-trained Memory Manager + Answer Agent for LLM memory, evaluated on LoCoMo. No official code exists — this is a from-scratch reproduction. Hobby/portfolio project: prefer cheap, simple, budget-cloud-friendly choices over enterprise polish.

## Git Workflow

- `main` is the stable branch. Never commit directly to `main`.
- Work on topic branches named `feat/<slug>`, `fix/<slug>`, or `docs/<slug>`.
- Commit early and often with imperative-mood messages ("Add LoCoMo loader"), body explains *why*.
- Always push the branch and open a PR against `main` for the user to review — the user cannot inspect the workspace directly.
- Keep PRs focused: one milestone or concern per PR.

## Knowledge Persistence

- Use the `notes` MCP tools (workspace/topic notes tagged `memory`) for cross-session context: decisions made, experiment outcomes, blockers, credentials locations. Never create local memory files.
- Durable *project* knowledge (design decisions, experiment results, gotchas) goes in `docs/` and is committed — notes are for session continuity, docs are the record.
- After any training/eval run, record: config used, cost, wall-clock time, and metrics in `docs/experiments.md` (create on first run).

## Project Layout

```
.claude/           Claude agent config (this file)
docs/              Design docs, paper notes, plans, experiment logs
src/memory_r1/    Python package (agents, memory bank, retrieval, rewards, eval)
scripts/           Runnable entry points (data prep, train, eval) — thin wrappers over src/
configs/           Training/eval configs (YAML), one file per experiment
tests/             pytest tests; fast, no GPU, no network
data/              Datasets and memory banks (gitignored; scripts download/build them)
outputs/           Checkpoints, logs, eval results (gitignored)
```

## Document Layout

- `docs/paper-notes.md` — distilled method summary of the paper; the source of truth for what we're reproducing.
- `docs/reproduction-plan.md` — milestones, stack decisions, budget estimates. Update when decisions change.
- `docs/experiments.md` — append-only experiment log (config, cost, metrics).
- Design docs for new components go in `docs/` as `<component>.md` before major implementation work.

## Development Workflow

- Python 3.11+, managed with `uv` (`uv sync`, `uv run ...`). Dependencies live in `pyproject.toml`; never `pip install` ad hoc.
- Lint/format with `ruff` (`uv run ruff check --fix . && uv run ruff format .`), test with `uv run pytest`. Run both before every commit.
- Everything GPU-heavy runs on rented cloud GPUs (RunPod first, but keep scripts provider-agnostic: plain Docker + bash). Local workspace has **no GPU** — never attempt training or vLLM inference here; write code + fast tests locally, run training remotely.
- Separate "logic" from "scale": memory-bank ops, reward functions, and eval metrics must be pure-Python testable without models.

## Stack Decisions (defaults, revisit as needed)

- RL: prototype with **TRL GRPOTrainer + LoRA**, migrate to **verl** (what the paper used) for faithful runs.
- Primary backbone: **Qwen2.5-3B-Instruct** (paper's smallest); scale to 7B/8B only after the pipeline works end-to-end.
- Judge / memory-bank bootstrapping: **GPT-4o-mini** (matches paper). API keys via env vars, never committed.
- Benchmark: **LoCoMo** (152 train / 81 val / 1307 test QA pairs).
