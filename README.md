# mem-r1-reproduce

From-scratch reproduction of **Memory-R1** ([arXiv:2508.19828](https://arxiv.org/abs/2508.19828)) — RL-trained agents that *manage* (ADD/UPDATE/DELETE/NOOP) and *use* (retrieve + distill) an external memory bank for long-horizon dialogue QA. End-to-end: data prep → GRPO/PPO training → LoCoMo benchmark.

No official implementation exists (the [authors' repo](https://github.com/yansikuan/memory-r1) is README-only), so this is built from the paper.

Hobby / portfolio project — **budget cloud first**: smallest paper backbone (Qwen2.5-3B) on single rented GPUs, scaling up only once the pipeline works.

## Docs

- [Paper notes](docs/paper-notes.md) — distilled method, results, and ablations to reproduce
- [Reproduction plan](docs/reproduction-plan.md) — milestones, stack decisions, budget

## Stack

Python 3.11+ / uv / TRL-then-verl / Qwen2.5-3B-Instruct / LoCoMo / RunPod-first.

## Development

```bash
uv sync
uv run pytest
uv run ruff check --fix . && uv run ruff format .
```

GPU work (training, vLLM inference) runs on rented cloud GPUs — see the plan for the compute path.
