# Experiment Log

Append-only. For each run: config, cost, wall-clock, results.

## 2026-07-05 — Memory-bank bootstrap (LoCoMo, full)

- **Config**: `scripts/build_memory_banks.py`, provider `nim`, chat model `meta/llama-3.1-8b-instruct`, temperature 0, per-turn fact extraction (`FACT_EXTRACTION_PROMPT`), 30 RPM client-side throttle.
- **Input**: all 10 LoCoMo conversations, 5,882 dialogue turns.
- **Output**: 16,094 memories → `artifacts/memory_banks/*.json` (2.2 MB, committed).
- **Cost**: $0 (NIM free tier). Wall-clock ≈ 3.5 h (rate-limit bound).
- **Notes**:
  - First attempt without throttling died on 429s ~1 min in; SDK-level retries are not enough at NIM's ~40 RPM — fixed with `RateLimiter` (30 RPM) + minutes-scale 429 backoff.
  - Extraction quality is usable but noisy: some low-value facts ("Melanie has guts", "Caroline spoke to Mel at ..."). Consider prompt tuning or a relevance filter before M4; the paper's RL Memory Manager is itself the mechanism that should clean this up.
  - Paper uses GPT-4o-mini for this step; rerun with `MEMR1_PROVIDER=openai` later if bank quality becomes a suspect in eval gaps.

## 2026-07-06 — Frozen no-RL baseline (LoCoMo, val split)

- **Config**: `scripts/run_eval.py --split val`, provider `nim`, answer + judge model `meta/llama-3.1-8b-instruct`, embeddings `nvidia/nv-embedqa-e5-v5`, top-k 60, committed banks from 2026-07-05 bootstrap.
- **Input**: 81 val QA (seed-42 split), all 10 banks embedded (16,094 memories).
- **Output**: `artifacts/eval/nim-val/` (results.jsonl + summary.json, committed).
- **Cost**: $0 (NIM free tier). Wall-clock ≈ 40 min (rate-limit bound).
- **Results** (val, n=81):

  | Metric | Overall | multi-hop (16) | temporal (13) | open-domain (3) | single-hop (49) |
  | --- | --- | --- | --- | --- | --- |
  | EM | 0.160 | 0.125 | 0.000 | 0.000 | 0.224 |
  | F1 | 0.377 | 0.359 | 0.467 | 0.000 | 0.382 |
  | BLEU-1 | 0.315 | 0.290 | 0.391 | 0.000 | 0.322 |
  | Judge | 0.481 | 0.438 | 0.231 | 0.667 | 0.551 |

- **Notes**:
  - Plausible band: above the paper's Mem0 baseline (test: F1 30.4 / BLEU-1 22.2 / Judge 45.7), below its RL result (45.0 / 37.5 / 62.7) — as a frozen baseline should be. Val ≠ test, so only roughly comparable.
  - llama-8b judge is lenient (accepted "Two or more." vs gold "Three" in smoke); paper-comparable numbers need the GPT-4o-mini judge (`MEMR1_PROVIDER=openai`).
  - open-domain n=3 is too small to read anything into; temporal EM=0 mostly date-format mismatches ("1993" vs "in 1993" scores F1 but not EM).

## 2026-07-06 — Frozen no-RL baseline (LoCoMo, test split)

- **Config**: `scripts/run_eval.py --split test` — same as the val run above (NIM, llama-3.1-8b answer + judge, nv-embedqa-e5-v5, top-k 60, committed banks).
- **Input**: 1307 test QA (seed-42 split).
- **Output**: `artifacts/eval/nim-test/` (results.jsonl + summary.json, committed).
- **Cost**: $0 (NIM free tier). Wall-clock ≈ 3 h (rate-limit bound).
- **Results** (test, n=1307), paper rows for reference:

  | Metric | Ours overall | multi-hop (238) | temporal (279) | open-domain (85) | single-hop (705) | Paper Mem0 | Paper RL (GRPO, llama-8b) |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | EM | 0.153 | 0.101 | 0.061 | 0.059 | 0.218 | — | — |
  | F1 | 0.352 | 0.355 | 0.324 | 0.152 | 0.387 | 0.304 | 0.450 |
  | BLEU-1 | 0.291 | 0.286 | 0.263 | 0.097 | 0.327 | 0.222 | 0.375 |
  | Judge | 0.423 | 0.378 | 0.312 | 0.341 | 0.492 | 0.457 | 0.627 |

- **Notes**:
  - Where we want the frozen baseline: F1/BLEU-1 above the paper's Mem0 row, well below its RL row — the gap M3/M4 training is supposed to close.
  - Judge (.423) lands slightly *below* Mem0's (.457), but the judges differ (llama-8b here vs GPT-4o-mini in the paper) so this comparison is soft; rescore predictions with `MEMR1_PROVIDER=openai` for the paper-comparable number (the JSONL keeps every prediction, so rescoring doesn't need to rerun answers).
  - open-domain is the weakest category (F1 .152) — these questions need broad synthesis across sessions, which top-60 retrieval + noisy banks handles poorly.
  - Our seed-42 test split ≠ the paper's unpublished partition; aggregate comparison only.

## 2026-07-11 — GRPO Answer Agent training (M3, first full run)

- **Config**: `configs/grpo-answer-qwen3b.yaml` @ `8f66fb6` — Qwen/Qwen2.5-3B-Instruct + LoRA (r16/α32, all proj), TRL 1.8.0 GRPOTrainer, EM reward, 152 train prompts (precomputed top-60 contexts, chat-template format), 3 epochs, effective batch 16, 8 generations/prompt, temperature 0.9, β=0.04, bf16 + gradient checkpointing.
- **Hardware**: RunPod RTX PRO 4500 Blackwell 32 GB ($0.74/hr), torch 2.13+cu130.
- **Output**: LoRA adapter (120 MB) → `outputs/grpo-answer-qwen3b/` on the pod's network volume (not committed; exceeds GitHub's 100 MB file limit).
- **Cost**: ≈ $2.10 (2.8 h pod time; train_runtime 9,031 s = 2 h 30 m for 228 steps, ~40 s/step). Plus ≈ $1.75 burned on a failed 24 GB RTX 4090 attempt (see notes).
- **Training signal** (mean EM reward per epoch, online sampling @ T=0.9):
  epoch 1 = 0.121, epoch 2 = 0.150, epoch 3 = 0.152 (+26% e1→e3, plateauing).
  151/228 steps had all-zero rewards (sparse EM ⇒ no gradient for those groups); final KL ≈ 0.003.
- **Notes**:
  - Two OOM lessons on 24 GB (RTX 4090): batch 8×accum 2 dies at step 0; batch 2×accum 8 dies mid-run (~step 62) on generation peaks of 16 sequences. Fixes: `generation_batch_size: 8`, checkpoint every 25 steps, auto-resume — after which the 32 GB run was clean end-to-end.
  - TRL resolved to 1.8.0 (config written for 0.14-era API): `max_prompt_length` removed; prompts must be message-format for chat-template application; `trl>=1.8` now pinned.
  - EM reward is sparse — 66% of steps carried no learning signal. If val eval shows weak lift, retry with `reward_metric: f1` (shaped) per the config's fallback note.
  - Reward plateau by epoch 3 suggests more epochs won't help at this scale; more train prompts or shaped reward are likelier levers.
  - Next: local-inference eval of the adapter on the val split (needs a pod-side LLMFn wrapper), then test split if promising.
