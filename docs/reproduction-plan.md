# Reproduction Plan

Budget-first reproduction of Memory-R1 (see [paper-notes.md](paper-notes.md)). Goal: end-to-end pipeline (data → training → benchmark) that reproduces the paper's *trends* on a small model first, then scales toward the headline numbers.

## Stack decisions

| Decision | Choice | Rationale |
|---|---|---|
| RL framework | TRL GRPOTrainer first, verl for faithful runs | TRL+LoRA validates the pipeline on 1 GPU cheaply; the paper used verl, so faithful runs migrate there |
| Primary backbone | Qwen2.5-3B-Instruct | Paper's smallest backbone; single-GPU friendly |
| Scale-up backbone | LLaMA-3.1-8B-Instruct | Paper's headline results table |
| API LLM provider | NVIDIA NIM free tier default; OpenAI GPT-4o-mini fallback | NIM is $0 (~40 RPM) for bootstrap/judge/baseline inference; GPT-4o-mini reserved for the final paper-comparable judge run |
| Retrieval | API embeddings + cosine similarity (Mem0-style): `nvidia/nv-embedqa-e5-v5` on NIM, `text-embedding-3-small` on OpenAI | Paper names no embedding model; QA-retrieval-tuned NIM model (`baai/bge-m3` is broken on the hosted endpoint as of 2026-07) |
| Cloud | RunPod first; scripts provider-agnostic (Docker + bash) | Cheapest reliable A6000/A100 spot capacity |
| Tooling | Python 3.11+, uv, ruff, pytest | Simple, fast |

## Milestones

### M0 — Bootstrap (this PR)
Repo scaffolding, Claude agent config, paper notes, plan.

### M1 — Data & memory bank
- LoCoMo download + loader; reproduce 152/81/1307 splits.
- Memory bank data structure (entries with IDs + temporal info) with ADD/UPDATE/DELETE/NOOP ops — pure Python, fully unit-tested.
- GPT-4o-mini bootstrap of initial memory banks per dialogue.
- Embedding retrieval over the bank (top-k, ~60 candidates for answering).

### M2 — Frozen baseline (no RL) — done (2026-07-06)
- Prompted (untrained) Answer Agent served free via NIM — no GPU rental.
- Eval harness: F1, BLEU-1, LLM-as-a-Judge on LoCoMo test.
- Result: test F1 .352 / BLEU-1 .291 / Judge .423 — above the paper's Mem0 row, below its RL row (see docs/experiments.md).

### M3 — RL Answer Agent (Stage 2) — in progress
- GRPO via TRL + LoRA on Qwen2.5-3B, single 24–48 GB pod (see docs/grpo-answer-agent.md).
- Reward: EM vs gold. Memory distillation prompt format, identical to the M2 eval prompt.
- Retrieval contexts precomputed and committed so the GPU box needs no API access.
- Target: measurable lift over M2 baseline.

### M4 — RL Memory Manager (Stage 1)
- The hard part: reward = frozen answerer's EM using the post-op memory bank.
- Custom reward function wrapping bank state + frozen model inference.
- Target: reproduce ablation direction (RL manager > vanilla manager).

### M5 — Faithful runs & write-up
- Port to verl, PPO + GRPO, scale to 7B/8B on multi-GPU pod.
- Compare against paper's Table 1; write up results in README + docs/experiments.md.

## Budget estimate

- M1–M2: API costs only (GPT-4o-mini bootstrap + judge) — a few dollars.
- M3–M4: single A6000-48GB (~$0.40–0.80/hr) or A100-80GB (~$1.30–1.90/hr) on RunPod; 152 training QA pairs means short runs — expect $10–50 per experiment.
- M5: 2–4×A100/H100 runs — the expensive tail; only after M1–M4 are solid.

## Risks

- LoCoMo splits used by the paper may not be published exactly — may need to infer the 152/81/1307 partition.
- Multi-turn reward (M4) doesn't fit TRL's vanilla reward-function API cleanly; may force the verl migration earlier.
- LLM-judge scores depend on judge prompt wording; F1/BLEU are the more comparable metrics.
