# M2 — Frozen (no-RL) Baseline

Goal: reproduce the paper's "vanilla" LoCoMo baseline row — a prompted, untrained
Answer Agent reading the bootstrapped memory banks — and build the eval harness
that every later milestone (M3 GRPO Answer Agent, M4 Memory Manager) reuses.
Everything runs on API models (NIM free tier by default): no GPU, $0.

## Pipeline per question

1. Load the committed bank for the question's conversation
   (`artifacts/memory_banks/<sample_id>.json`).
2. Embed all bank entries once per conversation (cached in-process), embed the
   question, retrieve top-60 memories by cosine similarity (Mem0-style RAG,
   as in the paper).
3. Frozen Answer Agent: single prompt with the retrieved memories
   (timestamped), instructed to first pick the relevant ones ("memory
   distillation", the same structure the RL policy is later trained on) and
   finish with a final `Answer:` line. We parse that line as the prediction.
4. Score against gold.

## Metrics

- **Exact match / token F1** — SQuAD-style normalization (lowercase, strip
  punctuation and articles, whitespace tokenize).
- **BLEU-1** — clipped unigram precision × brevity penalty (matches
  nltk `sentence_bleu` with weights `(1,0,0,0)`).
- **LLM-as-a-Judge** — yes/no semantic-correctness verdict given question,
  gold, and prediction. Default judge is the NIM chat model; the final
  paper-comparable run should use `MEMR1_PROVIDER=openai` (GPT-4o-mini judge,
  as in the paper).

All metrics are pure Python (`src/memory_r1/metrics.py`) and offline-testable;
the judge is just another `LLMFn`.

## Reporting

`scripts/run_eval.py --split {val,test,train} [--limit N] [--no-judge]`
writes per-question records (JSONL, resumable — reruns skip already-scored
questions) plus a summary JSON with overall and per-category means to
`artifacts/eval/<run-name>/`. Summaries also get logged in
`docs/experiments.md`.

## Cost / wall-clock (NIM free tier, 30 RPM)

| Run | LLM calls | Embedding calls | Wall-clock |
| --- | --- | --- | --- |
| val (81 Q, judge on) | ~162 | ~250 bank batches + 81 queries | ~25 min |
| test (1307 Q, judge on) | ~2614 | ~250 + 1307 | ~2.5–3 h |

Chat and embeddings are separately throttled at the provider's RPM; the run
is single-threaded so combined throughput stays under NIM's ~40 RPM per
endpoint. 429s get minutes-scale backoff on top.

## Known caveats

- Bank quality is bottlenecked by the llama-3.1-8b extraction (see
  `docs/experiments.md` 2026-07-05 entry) — if baseline numbers land far
  below the paper's Mem0/vanilla rows, rerun the bootstrap with
  `MEMR1_PROVIDER=openai` before blaming the harness.
- Our train/val/test split is a seeded reconstruction (sizes match the paper,
  exact partition unpublished), so test-set numbers are comparable in
  aggregate, not per-question.
