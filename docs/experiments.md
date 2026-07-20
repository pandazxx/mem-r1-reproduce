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

## 2026-07-11 — GRPO-trained vs frozen Qwen2.5-3B, val split (offline, MBP)

- **Config**: `scripts/run_eval.py --split val --contexts artifacts/contexts/val.jsonl` — fully offline: precomputed top-60 retrieval, local transformers inference (`memory_r1.local_llm`, greedy decoding, chat template), no judge. Run on an 18 GB M-series MacBook (MPS, fp16). Trained model = base + LoRA adapter from the 2026-07-11 GRPO run (HF: `pandazxx/mem-r1-answer-qwen3b`).
- **Input**: 81 val QA (seed-42 split), identical contexts/prompts for both models.
- **Output**: `artifacts/eval/mbp-qwen3b-grpo-val/` and `artifacts/eval/mbp-qwen3b-frozen-val/` (summary.json, committed).
- **Cost**: $0 (local). Wall-clock ≈ 45 min per run.
- **Results** (val, n=81, F1):

  | Model | EM | F1 | BLEU-1 | multi-hop F1 (16) | temporal F1 (13) | open-domain F1 (3) | single-hop F1 (49) |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | Frozen Qwen2.5-3B | 0.198 | 0.392 | 0.352 | 0.295 | 0.429 | 0.083 | 0.433 |
  | GRPO-trained | 0.198 | 0.379 | 0.341 | 0.239 | 0.429 | 0.000 | 0.435 |

- **Verdict: no lift.** The first GRPO run is a null result — overall F1 slightly *down* (−.013), EM identical, single-hop a wash, multi-hop and open-domain worse.
- **Notes**:
  - Temporal scores are byte-identical between the two models — under greedy decoding the policy produces the same outputs on those 13 questions. Consistent with final KL ≈ 0.003: the policy barely moved during training.
  - Root-cause hypothesis: sparse EM reward (66% of steps had all-zero rewards ⇒ no gradient) + conservative lr/LoRA meant almost no policy update, so eval parity is expected, not surprising.
  - Note the frozen Qwen-3B (local, greedy) already beats the M2 frozen llama-8b-via-NIM val baseline on F1 (.392 vs .377) — model/back-end differences matter as much as RL here; keep comparisons within the same inference path.
  - Next lever: rerun with `reward_metric: f1` (shaped reward, ~every step carries gradient) as a new config (`configs/grpo-answer-qwen3b-f1.yaml`), ~$2–3 on the same 32 GB pod (network volume kept alive). If still flat, raise lr or LoRA r before questioning the approach.

## 2026-07-12 — GRPO rerun with F1-shaped reward: first real lift (M3)

- **Config**: `configs/grpo-answer-qwen3b-f1.yaml` — single-variable change vs the EM run: `reward_metric: f1`. Everything else identical (Qwen2.5-3B + LoRA r16, TRL 1.8, 3 epochs, 228 steps, effective batch 16, 8 generations, T=0.9, β=0.04).
- **Hardware**: RunPod RTX PRO 4500 32 GB, run self-service by the user via `just pod-train` (cost guard auto-stopped the pod + Telegram ping — first real-pod exercise of both).
- **Cost**: train_runtime 8,519 s = 2 h 22 m ≈ $1.75. Eval (both models, same pod) ≈ $0.50.
- **Training signal**: shaping worked as designed — `frac_reward_zero_std` mostly 0 (vs 66% zero-signal steps under EM); per-step KL 0.006–0.43 (vs 0.003 final KL under EM). The policy actually moved this time.
- **Output**: adapter in `outputs/grpo-answer-qwen3b-f1/` on the network volume (export to HF Hub pending).
- **Eval** (val n=81, both models on the same pod/CUDA/bf16/greedy path; `artifacts/eval/pod-qwen3b-{f1,frozen}-val/`):

  | Model | EM | F1 | BLEU-1 | multi-hop F1 (16) | temporal F1 (13) | open-domain F1 (3) | single-hop F1 (49) |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | Frozen Qwen2.5-3B | 0.198 | 0.386 | 0.347 | 0.305 | 0.407 | 0.000 | 0.431 |
  | GRPO F1-reward | 0.198 | **0.410** | **0.364** | 0.303 | 0.462 | 0.000 | 0.456 |

- **Verdict: +2.4 F1 / +1.7 BLEU-1 overall (+6% relative)** — the first positive RL result of the reproduction. Gains concentrated in temporal (+.054) and single-hop (+.026); multi-hop flat; EM unchanged (the reward shaped answers *toward* gold tokens without producing more exact matches).
- **Notes**:
  - Confirms the null-result diagnosis: reward sparsity, not lr/LoRA capacity, was the bottleneck. One variable changed, KL rose ~100×, F1 followed.
  - Backend variance check: frozen F1 is .386 on CUDA/bf16 vs .392 on MPS/fp16 — ~.006 of greedy-decoding drift between backends, which is why both models must share an inference path. The +.024 lift is ~4× that noise floor.
  - EM-trained adapter (val F1 .379) < frozen < F1-trained (.410): reward choice alone swings ~3 F1 points on this budget.
  - Still short of the paper's relative lift (LLaMA-8B, verl, full test split) — but the mechanism now demonstrably works at 3B/TRL/$2 scale.
  - Next: export adapter to HF Hub, judge rescore (gpt-4o-mini) of both prediction sets, then decide between test-split eval of this adapter vs. moving to M4 (Memory Manager RL).

## 2026-07-16 — GRPO Memory Manager training (M4, first full run)

- **Config**: `configs/grpo-manager-qwen3b.yaml` — Qwen2.5-3B + LoRA r16, TRL GRPOTrainer, 230 evidence-anchored episodes (`artifacts/episodes/train.jsonl`), spliced-context reward: frozen Answer Agent (same base weights, adapter disabled, batched generation) token-F1 on the linked QA. 3 epochs = 345 steps.
- **Hardware**: RunPod RTX PRO 4500 32 GB, run self-service via `just pod-train` (after the PR #12 dispatch fix; the earlier crash cost ~1 min of pod time and proved the cost guard).
- **Cost**: train_runtime 13,830 s = 3 h 50 m ≈ $2.85 (~40 s/step; the reward's 8 answer generations run as one batched forward pass — sequential was ~90 s/step, PR #11).
- **Output**: adapter in `outputs/grpo-manager-qwen3b/` on the network volume.
- **Training signal** (from the log tail; full history in the checkpoint's `trainer_state.json`, not yet pulled):
  - Late-training **entropy collapse**: final steps show entropy 0.01–0.05 and `reward_std = 0` / `frac_reward_zero_std = 1` — at T=0.9 all 8 samples per prompt were identical, so epoch-3 steps carried no gradient. The policy settled on a deterministic op pattern well before the end.
  - Whether that pattern is *useful* (learned edits) or *degenerate* (e.g. always the same op) is not decidable from training telemetry — the proxy eval decides.
- **Next**: `scripts/eval_manager_episodes.py` on the 127 val episodes (manager ops vs NOOP baseline, same GPU, ~15 min) as go/no-go; if go, `scripts/rebuild_banks_with_manager.py` + context rebuild + banks A/B (docs/memory-manager-rl.md eval plan).

## 2026-07-16 — Memory Manager proxy eval: beats NOOP baseline on val episodes (M4)

- **Config**: `scripts/eval_manager_episodes.py`, 127 val episodes, manager adapter (greedy) vs NOOP baseline, frozen answerer = same base with adapter disabled. Pod GPU, 7 min, ~$0.10.
- **Output**: `artifacts/eval/manager-proxy-val/summary.json` (full rows in `outputs/manager-eval-val.json` on the pod volume).
- **Results**: mean reward **0.368 vs 0.329 NOOP** (+3.9 F1 points, +12% relative); 26 wins / 24 losses / 77 ties vs NOOP per episode.
- **Op distribution**: ADD 401 (92%), UPDATE 26, DELETE 3, NOOP 5, INVALID 1 — ~3.2 ADDs/episode ≈ one per extracted fact.
- **Reading**:
  - The entropy collapse seen in training settled on a *simple but reward-aligned* policy: re-ADD the turn's facts as memories. Under the splice semantics that pushes evidence directly into the QA context — precisely the mechanism that attacks the 27%-coverage gap. Not degenerate (UPDATEs/DELETEs still appear), but not sophisticated consolidation either.
  - Near-balanced wins/losses (26/24) with a positive mean: wins are larger than losses; 61% of episodes tie (edits didn't change the answer).
  - **Transfer caveat for the bank rebuild**: the proxy injects ADDs straight into the context; in the real A/B the re-ADDed facts must *win top-60 retrieval* to matter (they're near-duplicates of M1 entries, so ranking is plausible but not guaranteed). The rebuild A/B measures exactly this.
- **Verdict**: go for the bank rebuild (~$2). If the A/B shows the ADD-bias washes out, next lever is an entropy bonus / higher temperature to escape the early collapse and reach UPDATE/consolidation behavior (the paper's qualitative claim).

## 2026-07-19 — Banks A/B: RL-managed banks vs M1 banks (M4 final eval)

- **Config**: full pipeline on one pod session — `rebuild_banks_with_manager.py` (all 5,882 turns through the trained manager, batched greedy ops, applied in turn order to copies of the M1 banks), `build_train_contexts.py --banks artifacts/memory_banks_rl --out artifacts/contexts_rl --splits val` (NIM), then `run_eval.py` for both answerers on the RL contexts (CUDA/bf16/greedy — same path as the 2026-07-12 M1-bank numbers).
- **Output**: `artifacts/memory_banks_rl/` (committed), `artifacts/contexts_rl/val.jsonl`, `artifacts/eval/qwen3b-{frozen,f1}-rlbanks-val/`.
- **Cost**: ≈ $2.60 pod time total (including $0.75 for a first attempt that finished the rebuild but died on a missing `NVIDIA_API_KEY` at the context step; everything resumable, nothing lost).
- **Bank stats**: 16,094 → **29,755 entries (1.85×)** — the ADD-heavy policy nearly doubled the banks (per-conversation ops ≈ 92% ADD, few UPDATEs, ~0 DELETEs, matching the proxy eval distribution).
- **Results** — the completed A/B grid (val n=81, overall F1, same CUDA path):

  | Answerer \ Banks | M1 banks | RL banks |
  | --- | --- | --- |
  | Frozen Qwen2.5-3B | 0.386 | 0.367 |
  | GRPO F1-reward agent | **0.410** | 0.387 |

  Per-category (F1-agent): single-hop **.456 → .493 (+.037)**, multi-hop **.239 → .132 (−.107)**, temporal .462 → .390, open-domain 0 → 0. The frozen answerer shows the same pattern (single-hop .431 → .473, multi-hop .305 → .094).

- **Verdict: the RL-managed banks *hurt* overall (−2 F1 for both answerers).** The proxy-eval gain did not survive real retrieval. Mechanism, cleanly split by category:
  - **Coverage barely moved**: QA with <50% gold-token coverage in the top-60 went 22 → 20 of 81. The ADDs mostly *duplicated* facts already in the bank instead of filling gaps.
  - **Dilution is real**: 1.85× bank size means duplicates crowd the top-60. Single-hop questions *benefit* (their fact appears more often → +3.7 F1); multi-hop questions are crushed (they need many *distinct* facts in context → −10.7 F1).
  - The M3 Answer-Agent lift is robust across bank types (+2.4 on M1, +2.0 on RL banks) — the two stages are independent, as designed.
- **Interpretation**: this *reproduces the paper's qualitative ablation in reverse* — the paper claims a good manager consolidates (UPDATE) where naive managers duplicate (ADD), and our entropy-collapsed ADD-policy demonstrates exactly the failure mode that claim implies. The mechanism (op → bank → retrieval → answer → reward) works end-to-end; the *policy* is what needs improvement.
- **Next levers** (in order): (1) entropy bonus / higher sampling temperature to escape the early collapse; (2) penalize bank growth in the reward (e.g. small per-ADD cost) to push toward UPDATE/NOOP; (3) the paper's 50-turn windowed pre-op bank so UPDATE targets are less diluted. Each is a ~$3 rerun on the existing infra.
- **M4 total spend**: ≈ $6.20 (training $2.85 + proxy $0.35 + A/B $2.60 + failed-launch pennies).

## 2026-07-20 — M4 eval-semantics correction + v2 plan (issue #17)

- **Correction to the 2026-07-19 entry**: that A/B is a **postprocessor eval** — the rebuild applied manager ops *on top of the completed M1 banks*, so `NOOP` could not keep an M1 fact out of memory and `ADD` stacked duplicates onto M1 extraction. It measures *M1-plus-manager-edits*, which is favorable to the M1 baseline and not a true manager-constructed bank. The −2 F1 verdict stands for that semantics, but M4 claims should use the **true-manager eval** (bank starts empty; the manager decides storage) — now supported via `rebuild_banks_with_manager.py --mode construct`.
- **Reward fixes** (training/proxy no longer over-credit ADD):
  - `context_cap: 60` — spliced reward contexts are capped; an ADD displaces the weakest retrieved memory instead of getting a free extra slot real top-60 retrieval would not grant.
  - `add_penalty: 0.01` — per-ADD bank-growth cost.
- **New tooling**: raw manager ops persisted to `artifacts/manager_ops/` (one GPU pass → both rebuild modes replay offline); `scripts/manager_diagnostics.py` (op distribution, growth ratio, exact/near-duplicate rates, top-60 gold coverage, distinct-in-top-60, per-category F1).
- **v2 rerun plan**: `configs/grpo-manager-qwen3b-v2.yaml` — cap + penalty + **1 epoch** (v1 entropy-collapsed early; extra epochs bought nothing). ~115 steps ≈ 1.3 h ≈ $1 on the 32 GB pod. Same model/LoRA — no scale-up until reward semantics are validated.
