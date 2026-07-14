# M4 — GRPO Memory Manager (Stage 1 RL)

Goal: RL-train the Memory Manager — the model that decides `ADD` / `UPDATE` /
`DELETE` / `NOOP` as new dialogue facts arrive — and show the paper's ablation
direction: an RL-trained manager builds banks that answer questions better
than the vanilla (frozen-extraction) banks from M1. Paper reference: removing
the RL manager costs 6.5 F1 (41.0 → 34.5, PPO ablation).

Motivation from M3: 22/81 val questions (27%) have <50% of their gold-answer
tokens anywhere in the top-60 context (mean F1 .136 vs .598 when covered).
That ceiling belongs to memory construction — no Answer Agent training can
fix it.

## The core problem: the reward is an environment

The paper's reward for one memory operation is *downstream QA correctness of
a frozen Answer Agent using the resulting bank*. Faithfully, that means every
GRPO rollout replays a ~300-turn conversation into a bank and answers
questions — far beyond a single-GPU budget. M4 makes three approximations,
each stated so M5 (verl) can undo them:

1. **Single-op episodes anchored to QA evidence.** LoCoMo annotates each QA
   with its evidence turns (`QAPair.evidence` dia_ids). An episode is: the
   manager processes one evidence turn's extracted facts against the frozen
   M1 bank, and is rewarded by how well the QA linked to that evidence is
   answered afterwards. No multi-turn credit assignment (paper trains this
   way too — ops are rewarded per-turn by outcome, not per-conversation).
2. **Frozen-bank splicing instead of live retrieval.** The QA's top-60
   context is precomputed from the M1 bank (`artifacts/contexts/train.jsonl`,
   already committed). A candidate op edits that context directly:
   - `ADD` → append the new memory to the context
   - `UPDATE id` → replace that entry's text where it appears in the context
     (append if it wasn't retrieved)
   - `DELETE id` → drop the entry from the context
   - `NOOP` → context unchanged
   No embedding calls at training time; the GPU box stays API-free. This
   approximates "the op changes what retrieval later sees" and is exact
   whenever the affected memory is in (or would enter) the top-60.
3. **Token-F1 reward, not EM.** Direct lesson from M3: EM left 66% of steps
   with zero gradient; the F1-shaped rerun fixed it (+2.4 val F1). We start
   the manager with F1 reward (`reward_metric: f1`), EM available for a
   faithfulness ablation.

## Episode construction (offline, once)

`scripts/build_manager_episodes.py` emits `artifacts/episodes/train.jsonl`,
one record per (train QA × evidence turn):

```json
{"conversation_id": "...", "question": "...", "answer": "...", "category": 4,
 "turn": {"speaker": "...", "text": "...", "date_time": "..."},
 "facts": ["<extracted fact 1>", ...],
 "related": [{"id": "12", "text": "...", "timestamp": "..."}, ...],
 "context": [{"id": "3", "text": "...", "timestamp": "..."}, ...]}
```

- `facts`: NIM re-extraction of the evidence turn (same
  `FACT_EXTRACTION_PROMPT` as M1, so the manager sees M1-style input).
- `related`: top-6 memories most similar to the facts (NIM embeddings) —
  the candidates the manager may `UPDATE`/`DELETE`, with real bank IDs.
- `context`: the QA's top-60 from `artifacts/contexts/train.jsonl`, now
  carrying entry IDs so splicing can match `UPDATE`/`DELETE` targets.

152 train QA × ~2 evidence turns ≈ ~300 episodes; NIM cost $0 (rate-limit
bound, ~30 min). Val episodes built the same way for later analysis.

## Policy and prompt

Qwen2.5-3B-Instruct + LoRA (same r16/α32 recipe as M3 — infra reuse, and the
paper trains both stages on the same backbone). The manager prompt shows the
new facts + the `related` memories (with IDs and timestamps) and asks for a
JSON operation list; `memory_bank.parse_operations` already parses exactly
this (fences, aliases, single-object and list forms). Unparseable output or
an op referencing an unknown ID → reward 0 for that completion (the paper's
outcome-driven spirit: no format shaping beyond validity).

## Reward (the new plumbing)

```
completion → parse_operations → apply to context copy (splice rules above)
          → frozen Answer Agent answers the linked QA on the spliced context
          → reward = token-F1(answer, gold)
```

The frozen Answer Agent is the *same base model* the policy trains on
(Qwen2.5-3B): inside the TRL reward function we generate with the policy's
base weights under `disable_adapter()` — no second model in GPU memory,
greedy decoding, same `ANSWER_PROMPT` as M2/M3. A group of 8 candidate ops
whose spliced contexts differ produces 8 different answers → dense relative
advantage, exactly the mechanism that worked in the F1 rerun.

Baseline subtlety: `NOOP` reproduces the frozen-bank answer, so the manager
is rewarded only for ops that *beat* what the M1 bank already achieves —
the ablation comparison is built into the group baseline.

## Training config

`configs/grpo-manager-qwen3b.yaml`, same shape as M3: 3 epochs, effective
batch 16, 8 generations/prompt, T=0.9, β=0.04, bf16 + gradient checkpointing,
checkpoint every 25 steps + auto-resume. New keys: `episodes:` path,
`answer_max_new_tokens`. Ops are short (~50 tokens) but each reward call adds
8 greedy answer generations (~256 tokens): estimated step time ~60–90 s →
~300 episodes × 3 epochs / 16 ≈ 56 optimizer steps... **≈ 2–4 h ≈ $2–3 on
the 32 GB pod** (`just pod-train configs/grpo-manager-qwen3b.yaml 5h`).

## Eval plan

1. **Cheap proxy (during/after training)**: mean reward vs the NOOP baseline
   on held-out val episodes — did the manager learn ops that beat the frozen
   bank?
2. **Real eval (the M4 result)**: rebuild all 10 banks by replaying every
   conversation turn through the trained manager on the pod (~5.9k turns ×
   1 short generation ≈ 2–3 h ≈ $2), rebuild val contexts (`just contexts`,
   NIM, $0), then the standard A/B on the same CUDA path:
   frozen Answer Agent × {M1 banks, RL-manager banks} — and if positive,
   F1-adapter × RL-manager banks for the full two-stage number.
3. **Coverage stat**: the 27%-unanswerable fraction from M3's analysis should
   drop — that's the mechanism being bought, and it's checkable without any
   model (gold-token coverage of the new contexts).

## Risks

- **Splice approximation**: an ADD that would *not* be retrieved into the
  top-60 at eval time gets training credit it won't cash in. Mitigation: the
  episode's facts come from the QA's own evidence turn, so relevance (and
  hence retrieval) is likely; the bank-rebuild eval measures the true effect.
- **Reward-inside-trainer memory**: policy training + frozen generation on
  one 32 GB card is untested; `disable_adapter()` shares weights so the
  overhead is activation memory only. Fallback: answer-generation batch of 4,
  or a CPU-offloaded answerer (slow), or rent 48 GB.
- **Evidence annotation quality**: some LoCoMo evidence ids may not resolve
  to turns (format drift); episodes with no resolvable evidence are dropped
  and counted in the build log.
- **~56 optimizer steps is few**: if reward is flat, first lever is more
  episodes per QA (all sessions' turns near the evidence, not just evidence
  turns), second is more epochs — both cheap.
