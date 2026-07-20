"""Proxy-eval the trained Memory Manager on val episodes (GPU box).

Go/no-go check before the full bank rebuild: for each val episode the
manager (adapter, greedy) proposes ops, the frozen Answer Agent (same base
weights, adapter disabled) answers from the op-spliced context, and the F1
is compared against the NOOP baseline (untouched M1 context) — i.e. did the
manager's edits beat the frozen bank on the same questions?

Fully offline: precomputed episodes, local inference.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
NOOP_COMPLETION = '{"operations": [{"operation": "NOOP"}]}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", default="artifacts/episodes/val.jsonl")
    parser.add_argument("--adapter", default="outputs/grpo-manager-qwen3b")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out", default="outputs/manager-eval-val.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only the first N episodes (smoke)")
    parser.add_argument(
        "--cap",
        type=int,
        default=60,
        help="context size cap: ADDs displace the weakest retrieved memories "
        "instead of extending the context (0 = uncapped, the pre-issue-#17 semantics)",
    )
    args = parser.parse_args()
    cap = args.cap or None

    from memory_r1.answer_agent import parse_answer
    from memory_r1.local_llm import make_local_batch_llm
    from memory_r1.manager import (
        _spliced_answer_prompt,
        build_manager_prompt,
        load_episodes,
    )
    from memory_r1.memory_bank import OperationError, parse_operations
    from memory_r1.metrics import f1_score

    episodes = load_episodes(ROOT / args.episodes)[: args.limit]
    generate = make_local_batch_llm(args.model, adapter=str(ROOT / args.adapter))
    print(f"{len(episodes)} episodes, adapter {args.adapter}")

    rows = []
    op_counts: Counter = Counter()
    for start in range(0, len(episodes), args.batch_size):
        batch = episodes[start : start + args.batch_size]
        completions = generate([build_manager_prompt(ep) for ep in batch], use_adapter=True)
        answer_prompts, slots = [], []
        for i, (ep, completion) in enumerate(zip(batch, completions, strict=True)):
            try:
                op_counts.update(op.op for op in parse_operations(completion))
            except OperationError:
                op_counts["INVALID"] += 1
            for kind, source in (("managed", completion), ("noop", NOOP_COMPLETION)):
                result = _spliced_answer_prompt(source, ep, cap=cap)
                if result is not None:
                    answer_prompts.append(result[0])
                    slots.append((i, kind))
        answers = generate(answer_prompts, use_adapter=False)
        scores: dict[tuple[int, str], float] = {}
        for (i, kind), raw in zip(slots, answers, strict=True):
            scores[(i, kind)] = f1_score(parse_answer(raw), batch[i]["answer"])
        for i, (ep, completion) in enumerate(zip(batch, completions, strict=True)):
            rows.append(
                {
                    "conversation_id": ep["conversation_id"],
                    "question": ep["question"],
                    "evidence": ep["evidence"],
                    "category": ep["category"],
                    "completion": completion,
                    "reward": scores.get((i, "managed"), 0.0),
                    "noop_reward": scores[(i, "noop")],
                }
            )
        done = len(rows)
        print(f"[{done}/{len(episodes)}]", flush=True)

    n = len(rows)
    mean = sum(r["reward"] for r in rows) / n
    noop_mean = sum(r["noop_reward"] for r in rows) / n
    wins = sum(r["reward"] > r["noop_reward"] for r in rows)
    losses = sum(r["reward"] < r["noop_reward"] for r in rows)
    invalid = sum(1 for r in rows if r["reward"] == 0.0 and r["noop_reward"] > 0.0)
    summary = {
        "n": n,
        "mean_reward": mean,
        "mean_noop_reward": noop_mean,
        "wins_vs_noop": wins,
        "losses_vs_noop": losses,
        "ties": n - wins - losses,
        "op_counts": dict(op_counts),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    print(f"(invalid-or-worse marker: {invalid} episodes scored 0 where NOOP scored >0)")


if __name__ == "__main__":
    main()
