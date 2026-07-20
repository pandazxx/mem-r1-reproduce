"""Rebuild the memory banks with the trained Memory Manager (GPU box).

Replays every conversation turn through the manager: the precomputed turn
inputs (facts + related memories from the session-so-far M1 bank, see
scripts/build_turn_inputs.py) become manager prompts; the proposed ops are
applied in turn order. Two rebuild semantics (issue #17):

- ``--mode edit`` (postprocessor): ops edit a copy of the completed M1 bank.
  NOOP cannot keep an M1 fact out; ADDs stack on top of M1 extraction.
- ``--mode construct`` (true manager): the bank starts empty and the manager
  is the source of truth — NOOP means the fact is never stored; UPDATE and
  DELETE resolve their M1-id targets to the constructed entry with the same
  text (unresolved when that fact was never stored).

Raw manager ops are saved per conversation (``--ops-dir``) so either rebuild
can be replayed offline without re-running the model. Fully offline;
resumable per conversation.
"""

import argparse
import json
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
    parser.add_argument("--mode", choices=("edit", "construct"), default="edit")
    parser.add_argument(
        "--ops-dir",
        default="artifacts/manager_ops",
        help="save/reuse raw manager completions per conversation (jsonl); "
        "existing files skip generation entirely",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    from memory_r1.manager import (
        apply_operations,
        build_manager_prompt,
        construct_bank,
        load_episodes,
    )
    from memory_r1.memory_bank import MemoryBank, OperationError, parse_operations

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ops_dir = ROOT / args.ops_dir
    ops_dir.mkdir(parents=True, exist_ok=True)
    generate = None  # model loads lazily: replaying saved ops needs no GPU

    for src in sorted((ROOT / args.turn_inputs).glob("*.jsonl")):
        dest = out_dir / f"{src.stem}.json"
        if dest.exists():
            print(f"{src.stem}: exists, skipping")
            continue
        records = [r for r in load_episodes(src) if r["facts"]]

        ops_path = ops_dir / f"{src.stem}.jsonl"
        if ops_path.exists():
            saved = {r["dia_id"]: r["completion"] for r in load_episodes(ops_path)}
            completions = [saved[r["dia_id"]] for r in records]
            print(f"{src.stem}: replaying {len(completions)} saved ops", flush=True)
        else:
            if generate is None:
                from memory_r1.local_llm import make_local_batch_llm

                generate = make_local_batch_llm(args.model, adapter=str(ROOT / args.adapter))
            completions = []
            for start in range(0, len(records), args.batch_size):
                batch = records[start : start + args.batch_size]
                completions.extend(generate([build_manager_prompt(r) for r in batch]))
                print(
                    f"{src.stem} [{min(start + args.batch_size, len(records))}/{len(records)}]",
                    flush=True,
                )
            with ops_path.open("w") as sink:
                for record, completion in zip(records, completions, strict=True):
                    sink.write(
                        json.dumps({"dia_id": record["dia_id"], "completion": completion}) + "\n"
                    )

        parsed = []
        stats: Counter = Counter()
        for record, completion in zip(records, completions, strict=True):
            try:
                parsed.append((record, parse_operations(completion)))
            except OperationError:
                stats["invalid"] += 1

        if args.mode == "construct":
            m1_bank = MemoryBank.load(ROOT / "artifacts" / "memory_banks" / f"{src.stem}.json")
            bank, op_stats = construct_bank(
                m1_bank, ((r["turn"]["date_time"], ops) for r, ops in parsed)
            )
            stats.update(op_stats)
            before = len(m1_bank)
        else:
            bank = MemoryBank.load(ROOT / "artifacts" / "memory_banks" / f"{src.stem}.json")
            before = len(bank)
            for record, operations in parsed:
                stats.update(op.op for op in operations)
                _, skipped = apply_operations(
                    bank, operations, default_timestamp=record["turn"]["date_time"]
                )
                stats["skipped"] += skipped
        bank.save(dest)
        print(f"{src.stem} [{args.mode}]: M1 {before} -> {len(bank)} entries, ops {dict(stats)}")


if __name__ == "__main__":
    main()
