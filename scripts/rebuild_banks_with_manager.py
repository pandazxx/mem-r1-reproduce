"""Rebuild the memory banks with the trained Memory Manager (GPU box).

Replays every conversation turn through the manager: the precomputed turn
inputs (facts + related memories from the session-so-far M1 bank, see
scripts/build_turn_inputs.py) become manager prompts; the proposed ops are
applied in turn order to a copy of the conversation's M1 bank. The result is
the "RL-managed" bank: M1 extraction edited by the manager's ADD / UPDATE /
DELETE / NOOP decisions — the A/B counterpart to the vanilla M1 banks.

Op generation is batched (inputs are precomputed, so order only matters when
applying). Fully offline; resumable per conversation. Copy the output banks
off the pod (or commit them) for context rebuilding.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turn-inputs", default="artifacts/turn_inputs")
    parser.add_argument("--adapter", default="outputs/grpo-manager-qwen3b")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out", default="artifacts/memory_banks_rl")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    from memory_r1.local_llm import make_local_batch_llm
    from memory_r1.manager import apply_operations, build_manager_prompt, load_episodes
    from memory_r1.memory_bank import MemoryBank, OperationError, parse_operations

    generate = make_local_batch_llm(args.model, adapter=str(ROOT / args.adapter))
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted((ROOT / args.turn_inputs).glob("*.jsonl")):
        dest = out_dir / f"{src.stem}.json"
        if dest.exists():
            print(f"{src.stem}: exists, skipping")
            continue
        records = [r for r in load_episodes(src) if r["facts"]]
        bank = MemoryBank.load(ROOT / "artifacts" / "memory_banks" / f"{src.stem}.json")
        before = len(bank)
        print(f"{src.stem}: {len(records)} turns with facts, bank {before} entries", flush=True)

        completions: list[str] = []
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            completions.extend(generate([build_manager_prompt(r) for r in batch]))
            print(
                f"{src.stem} [{min(start + args.batch_size, len(records))}/{len(records)}]",
                flush=True,
            )

        stats: Counter = Counter()
        for record, completion in zip(records, completions, strict=True):
            try:
                operations = parse_operations(completion)
            except OperationError:
                stats["invalid"] += 1
                continue
            stats.update(op.op for op in operations)
            applied, skipped = apply_operations(
                bank, operations, default_timestamp=record["turn"]["date_time"]
            )
            stats["skipped"] += skipped
        bank.save(dest)
        print(f"{src.stem}: {before} -> {len(bank)} entries, ops {dict(stats)} -> {dest}")


if __name__ == "__main__":
    main()
