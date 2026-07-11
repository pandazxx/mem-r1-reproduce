"""Run the LoCoMo evaluation over the committed memory banks.

Resumable: per-question results stream to <out>/results.jsonl and already-
scored questions are skipped on rerun. A summary.json (overall and
per-category means) is rewritten at the end of every run.

Two modes:
- API (default): answer + judge via the provider (NIM free tier unless
  MEMR1_PROVIDER=openai), retrieval via provider embeddings.
- Local (--model, optionally --adapter): answer with a local transformers
  model (GPU box or Apple Silicon; needs `uv sync --extra train`). Judge is
  skipped (rescore via API later). Combine with --contexts to reuse
  precomputed retrievals and run fully offline.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_r1.evaluation import QAResult, evaluate_answer, evaluate_question, result_key, summarize
from memory_r1.locomo import load_locomo, make_splits
from memory_r1.memory_bank import MemoryBank, MemoryEntry
from memory_r1.providers import get_provider, make_embedder, make_llm
from memory_r1.retrieval import DEFAULT_TOP_K, Retriever

ROOT = Path(__file__).resolve().parents[1]
BANKS = ROOT / "artifacts" / "memory_banks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N QA")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-judge calls")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--name", default=None, help="run name (default: <provider>-<split>)")
    parser.add_argument("--model", default=None, help="local HF model id/path instead of the API")
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir for --model")
    parser.add_argument(
        "--contexts",
        default=None,
        help="precomputed contexts JSONL (artifacts/contexts/<split>.jsonl): skip retrieval",
    )
    return parser.parse_args()


def load_context_memories(path: Path) -> dict[str, list[MemoryEntry]]:
    from memory_r1.grpo import load_contexts

    return {
        result_key(c["conversation_id"], c["question"]): [
            MemoryEntry(id=str(i), text=m["text"], timestamp=m["timestamp"])
            for i, m in enumerate(c["memories"])
        ]
        for c in load_contexts(path)
    }


def main() -> None:
    args = parse_args()
    if args.model:
        from memory_r1.local_llm import make_local_llm

        default_name = f"{Path(args.adapter or args.model).name}-{args.split}"
        llm = make_local_llm(args.model, adapter=args.adapter)
        judge_llm = None  # never self-judge; rescore predictions via API later
        print(f"local model: {args.model}" + (f" + {args.adapter}" if args.adapter else ""))
    else:
        provider = get_provider()
        default_name = f"{provider.name}-{args.split}"
        llm = make_llm(provider)  # shared by answer agent and judge: one rate-limit budget
        judge_llm = None if args.no_judge else llm
        print(f"provider: {provider.name} ({provider.chat_model} / {provider.embedding_model})")

    name = args.name or default_name
    out_dir = ROOT / "artifacts" / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    conversations = load_locomo(ROOT / "data" / "locomo10.json")
    qa_pairs = list(getattr(make_splits(conversations), args.split))
    if args.limit is not None:
        qa_pairs = qa_pairs[: args.limit]

    done: dict[str, QAResult] = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            record = json.loads(line)
            done[result_key(record["conversation_id"], record["question"])] = QAResult(**record)

    print(f"run: {name}, split: {args.split}, {len(qa_pairs)} QA, {len(done)} already scored")

    if args.contexts:
        context_memories = load_context_memories(ROOT / args.contexts)
    else:
        context_memories = None
        retriever = Retriever(make_embedder(get_provider()))
        banks = {p.stem: MemoryBank.load(p) for p in sorted(BANKS.glob("*.json"))}

    results = []
    with results_path.open("a") as sink:
        for i, qa in enumerate(qa_pairs, 1):
            key = result_key(qa.conversation_id, qa.question)
            if key in done:
                results.append(done[key])
                continue
            if context_memories is not None:
                result = evaluate_answer(qa, context_memories[key], llm, judge_llm=judge_llm)
            else:
                result = evaluate_question(
                    qa,
                    banks[qa.conversation_id],
                    retriever,
                    llm,
                    judge_llm=judge_llm,
                    top_k=args.top_k,
                )
            sink.write(json.dumps(result.to_dict()) + "\n")
            sink.flush()
            results.append(result)
            print(
                f"[{i}/{len(qa_pairs)}] {qa.conversation_id} {qa.category_name}"
                f" f1={result.f1:.2f} | {qa.question[:60]}",
                flush=True,
            )

    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {results_path} and {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
