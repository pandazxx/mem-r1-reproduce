# Memory-R1 — Paper Notes

**Paper**: [arXiv:2508.19828](https://arxiv.org/abs/2508.19828) ([HTML v5](https://arxiv.org/html/2508.19828v5))
**Status of official code**: [yansikuan/memory-r1](https://github.com/yansikuan/memory-r1) is README-only ("code coming soon" since Sep 2025). No official implementation exists — from-scratch reproduction is justified.

## Method

Two agents, each fine-tuned with outcome-driven RL (no intermediate operation labels):

### Stage 1 — Memory Manager
- Input: new dialogue-turn information + retrieved related memories from an external memory bank (entries have unique IDs and temporal info).
- Output: one memory operation — `ADD` / `UPDATE` / `DELETE` / `NOOP`.
- Reward: downstream answer correctness — Exact Match of a **frozen** answer agent using the resulting memory bank.

### Stage 2 — Answer Agent
- Input: question + ~60 candidate memories retrieved via similarity-based RAG (Mem0-style retrieval).
- Behavior: **memory distillation** — filter candidates down to relevant entries, then reason and answer.
- Reward: EM against gold answers.

### RL setup
- Algorithms: PPO and GRPO (GRPO generally best), implemented in **verl**.
- Hyperparams: actor LR 1e-6, critic LR 1e-5, batch 128, prompt/response max 4096/2048, temperature 1.0 (train), greedy (eval).

## Models & Data

- Backbones: LLaMA-3.1-8B-Instruct; Qwen-2.5-Instruct at 3B / 7B / 14B.
- Benchmark: **LoCoMo** (multi-session dialogues, ~300 turns / ~9k tokens each). Splits: **152 train / 81 val / 1307 test** QA pairs.
- GPT-4o-mini bootstraps the initial temporal memory bank.
- Also evaluated on MSC and LongMemEval.
- LoCoMo data: https://snap-research.github.io/locomo/

## Key Results (LLaMA-3.1-8B)

| Method | F1 | BLEU-1 | LLM-Judge |
|---|---|---|---|
| Mem0 (prior best) | 30.4 | 22.2 | 45.7 |
| Memory-R1 (GRPO) | 45.0 | 37.5 | 62.7 |

(+48% relative F1 over prior best.)

## Ablations worth reproducing

1. Removing RL Memory Manager: F1 41.0 → 34.5 (PPO).
2. Memory distillation adds ~4 F1 (41.0 → 45.0, GRPO).
3. Qualitative: vanilla managers wrongly emit DELETE+ADD where UPDATE would consolidate.

## Compute notes

- Paper used 4×H100-80GB (8× for Qwen-14B).
- Budget path: Qwen2.5-3B + GRPO + LoRA fits a single A6000-48GB / A100-80GB; training data is tiny (152 QA), so runs cost tens of dollars on RunPod/vast.ai.
- The tricky infra piece: the reward function is a *multi-turn environment* (memory-bank state + frozen answerer inside the reward), not a simple scalar scorer.

## Related repos (unofficial)

- [pradyutnair-prosus/memory-r1](https://github.com/pradyutnair-prosus/memory-r1) — partial unofficial reproduction + LoCoMo benchmarking of SimpleMem/Mem0/MemU; actively updated. Worth studying, not copying.
- [vpakspace/graph-memory-r1](https://github.com/vpakspace/graph-memory-r1) — inspired-by extension with Neo4j graph memory.
