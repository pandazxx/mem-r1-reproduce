# M3 — GRPO Answer Agent (Stage 2 RL)

Goal: RL-train the Answer Agent with GRPO (TRL + LoRA, Qwen2.5-3B-Instruct)
and show a measurable lift over the M2 frozen baseline (test F1 .352 /
BLEU-1 .291 / Judge .423). The paper trains this stage with verl on the 152
LoCoMo train questions using exact-match reward; we prototype with TRL on a
single rented GPU, then port to verl in M5.

## Data: precomputed retrieval contexts

Retrieval (NIM embeddings, top-60) is precomputed once per QA by
`scripts/build_train_contexts.py` and committed to
`artifacts/contexts/{train,val}.jsonl`:

```json
{"conversation_id": "...", "question": "...", "answer": "...",
 "category": 4, "memories": [{"text": "...", "timestamp": "..."}]}
```

Why precompute:
- the GPU box needs **no API keys and no rate-limited calls** — training is
  pure local compute;
- retrieval is frozen across runs, so reward changes are attributable to the
  policy, not retrieval drift;
- identical prompts to the M2 eval harness (`ANSWER_PROMPT`), so baseline
  vs. trained comparisons are apples-to-apples.

## Training

- **Policy**: Qwen2.5-3B-Instruct + LoRA (r=16 on all attention/MLP
  projections), bf16, gradient checkpointing.
- **Algorithm**: TRL `GRPOTrainer`, 8 completions/prompt.
- **Reward**: exact match of the parsed `Answer:` line vs gold
  (`memory_r1.grpo.answer_reward`, the paper's reward). Token-F1 is available
  behind `reward_metric: f1` as a shaped alternative if EM proves too sparse
  on 152 prompts.
- **Config**: `configs/grpo-answer-qwen3b.yaml`; entry point
  `scripts/train_grpo_answer_agent.py --config ...`. Checkpoints under
  `outputs/` (gitignored) — copy the LoRA adapter off the pod before killing
  it.

Everything reward/prompt-related is pure Python and unit-tested offline;
the heavy deps (torch/trl/peft/...) are an optional extra
(`uv sync --extra train`) so the local no-GPU workspace never installs them.

## Runbook (RunPod or any Docker box)

```bash
# pod: 1× RTX 4090 24GB (prototype) or L40S/A6000 48GB (headroom)
git clone https://github.com/pandazxx/mem-r1-reproduce && cd mem-r1-reproduce
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
uv sync --extra train
uv run python scripts/train_grpo_answer_agent.py --config configs/grpo-answer-qwen3b.yaml
# then: tar the LoRA adapter in outputs/ and scp it off the pod
```

## Cost estimate

152 prompts × 8 generations × 3 epochs ≈ 3.6k completions of ≤256 tokens on
a 3B model: a few GPU-hours. On a ~$0.35–0.70/hr RTX 4090 that is **under
$5 per experiment**; a 48 GB card roughly doubles that. Budget $20–30 for
several reward/hyperparameter iterations.

## Eval plan (follow-up within M3)

After training, rerun the M2 harness on the test split with the tuned model
serving answers locally on the pod (transformers/vLLM `LLMFn` instead of the
NIM API), judge stays on the API. Target: F1 above the frozen .352 with the
same banks, retrieval, and prompts.

## Risks

- EM reward on 152 prompts is sparse; if reward stays flat, switch
  `reward_metric: f1` or raise generations per prompt.
- 24 GB may OOM at max_prompt_length 3072 with 8 generations — drop
  batch size or rent 48 GB.
- TRL GRPO ≠ verl GRPO in implementation details (advantage normalization,
  KL handling); trends here, faithful numbers in M5.
