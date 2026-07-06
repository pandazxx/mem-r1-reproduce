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
