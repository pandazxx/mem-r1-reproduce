"""Diagnostics for a memory-bank setup (issue #17). Pure offline.

For each named setup (--setup NAME BANKS_DIR CONTEXTS_JSONL [EVAL_DIR]):
bank size, exact/near-duplicate rates, top-60 gold-token coverage, distinct
memories per context, and per-category F1 from the eval results. Run with
two setups for an A/B table, e.g.:

    manager_diagnostics.py \
        --setup m1 artifacts/memory_banks artifacts/contexts/val.jsonl artifacts/eval/pod-qwen3b-frozen-val \
        --setup rl artifacts/memory_banks_rl artifacts/contexts_rl/val.jsonl artifacts/eval/qwen3b-frozen-rlbanks-val
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_r1.locomo import CATEGORY_NAMES
from memory_r1.metrics import _tokens, normalize_answer

ROOT = Path(__file__).resolve().parents[1]


def bank_stats(banks_dir: Path) -> dict:
    entries = exact_dupes = near_dupes = 0
    for path in sorted(banks_dir.glob("*.json")):
        texts = [e["text"] for e in json.loads(path.read_text())["entries"]]
        entries += len(texts)
        exact = Counter(normalize_answer(t) for t in texts)
        exact_dupes += sum(n - 1 for n in exact.values() if n > 1)
        near = Counter(" ".join(sorted(set(_tokens(t)))) for t in texts)
        near_dupes += sum(n - 1 for n in near.values() if n > 1)
    return {
        "entries": entries,
        "exact_dup_rate": round(exact_dupes / entries, 4),
        "near_dup_rate": round(near_dupes / entries, 4),
    }


def context_stats(contexts_path: Path) -> dict:
    low_coverage = 0
    distinct: list[int] = []
    records = [json.loads(line) for line in contexts_path.read_text().splitlines()]
    for record in records:
        memory_texts = [m["text"] for m in record["memories"]]
        distinct.append(len({normalize_answer(t) for t in memory_texts}))
        gold = set(_tokens(record["answer"]))
        context_tokens = set(_tokens(" ".join(memory_texts)))
        coverage = sum(t in context_tokens for t in gold) / max(len(gold), 1)
        low_coverage += coverage < 0.5
    return {
        "n": len(records),
        "low_gold_coverage": low_coverage,
        "mean_distinct_in_topk": round(sum(distinct) / len(distinct), 1),
    }


def eval_stats(eval_dir: Path) -> dict:
    rows = [json.loads(line) for line in (eval_dir / "results.jsonl").read_text().splitlines()]
    by_category = defaultdict(list)
    for row in rows:
        by_category[CATEGORY_NAMES[row["category"]]].append(row["f1"])
    stats = {"overall_f1": round(sum(r["f1"] for r in rows) / len(rows), 4)}
    stats.update(
        {f"{name}_f1": round(sum(scores) / len(scores), 4) for name, scores in by_category.items()}
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setup",
        nargs="+",
        action="append",
        required=True,
        metavar="NAME BANKS CONTEXTS [EVAL_DIR]",
    )
    args = parser.parse_args()

    report = {}
    for setup in args.setup:
        name, banks, contexts = setup[0], ROOT / setup[1], ROOT / setup[2]
        entry = {"banks": bank_stats(banks), "contexts": context_stats(contexts)}
        if len(setup) > 3:
            entry["eval"] = eval_stats(ROOT / setup[3])
        report[name] = entry
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
