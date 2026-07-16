"""Precompute top-60 retrieval contexts for the train and val QA splits.

Writes artifacts/contexts/{train,val}.jsonl (committed): one record per QA
with the gold answer and the retrieved memories. GRPO training (M3) then
runs on a GPU box with no API keys or rate limits.

Resumable per question: existing records are kept and skipped on rerun.

For the M4 banks A/B, point --banks at the RL-managed banks and --out at a
separate directory, e.g.:
    build_train_contexts.py --banks artifacts/memory_banks_rl \
        --out artifacts/contexts_rl --splits val
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_r1.evaluation import result_key
from memory_r1.locomo import load_locomo, make_splits
from memory_r1.memory_bank import MemoryBank
from memory_r1.providers import get_provider, make_embedder
from memory_r1.retrieval import DEFAULT_TOP_K, Retriever

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", default="artifacts/memory_banks")
    parser.add_argument("--out", default="artifacts/contexts")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    provider = get_provider()
    print(f"provider: {provider.name} ({provider.embedding_model})")
    retriever = Retriever(make_embedder(provider))
    banks = {p.stem: MemoryBank.load(p) for p in sorted((ROOT / args.banks).glob("*.json"))}
    splits = make_splits(load_locomo(ROOT / "data" / "locomo10.json"))
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        qa_pairs = getattr(splits, split)
        dest = out_dir / f"{split}.jsonl"
        done = set()
        if dest.exists():
            for line in dest.read_text().splitlines():
                record = json.loads(line)
                done.add(result_key(record["conversation_id"], record["question"]))
        print(f"{split}: {len(qa_pairs)} QA, {len(done)} already built", flush=True)
        with dest.open("a") as sink:
            for i, qa in enumerate(qa_pairs, 1):
                if result_key(qa.conversation_id, qa.question) in done:
                    continue
                memories = retriever.retrieve(
                    banks[qa.conversation_id], qa.question, k=DEFAULT_TOP_K
                )
                record = {
                    "conversation_id": qa.conversation_id,
                    "question": qa.question,
                    "answer": qa.answer,
                    "category": qa.category,
                    "memories": [{"text": m.text, "timestamp": m.timestamp} for m in memories],
                }
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                print(f"{split} [{i}/{len(qa_pairs)}] {qa.conversation_id}", flush=True)
        print(f"{split}: wrote {dest}")


if __name__ == "__main__":
    main()
