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
