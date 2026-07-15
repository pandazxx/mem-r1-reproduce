"""Build Memory Manager training episodes (M4).

One episode per (QA x evidence turn): the facts NIM re-extracts from the
evidence turn, the top-6 related bank memories (the manager's UPDATE/DELETE
candidates), and the QA's precomputed top-60 context with bank ids recovered
so ops can be spliced in at reward time (see docs/memory-manager-rl.md).

Writes artifacts/episodes/{train,val}.jsonl (committed). Needs the default
provider (NIM free tier, NVIDIA_API_KEY) plus the committed banks and
contexts. Resumable per (QA, evidence turn); evidence ids that don't resolve
to a turn are logged and skipped.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_r1.bootstrap import extract_facts
from memory_r1.locomo import Conversation, Turn, load_locomo, make_splits
from memory_r1.manager import RELATED_K
from memory_r1.memory_bank import MemoryBank
from memory_r1.providers import get_provider, make_embedder, make_llm
from memory_r1.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "episodes"
SPLITS = ("train", "val")


def turn_index(conversation: Conversation) -> dict[str, tuple[Turn, str]]:
    return {
        turn.dia_id: (turn, session.date_time)
        for session in conversation.sessions
        for turn in session.turns
    }


def context_with_ids(memories: list[dict], bank: MemoryBank) -> list[dict]:
    """Recover bank entry ids for a precomputed context (stored as text+timestamp)."""
    by_text: dict[tuple[str, str | None], str] = {}
    for entry in bank.entries:
        by_text.setdefault((entry.text, entry.timestamp), entry.id)
    return [
        {
            "id": by_text.get((m["text"], m["timestamp"])),
            "text": m["text"],
            "timestamp": m["timestamp"],
        }
        for m in memories
    ]


def episode_key(record: dict) -> tuple[str, str, str]:
    return (record["conversation_id"], record["question"], record["evidence"])


def main() -> None:
    provider = get_provider()
    print(f"provider: {provider.name} ({provider.chat_model} / {provider.embedding_model})")
    llm = make_llm(provider)
    retriever = Retriever(make_embedder(provider))

    conversations = {c.sample_id: c for c in load_locomo(ROOT / "data" / "locomo10.json")}
    turns = {cid: turn_index(c) for cid, c in conversations.items()}
    banks = {
        p.stem: MemoryBank.load(p)
        for p in sorted((ROOT / "artifacts" / "memory_banks").glob("*.json"))
    }
    splits = make_splits(list(conversations.values()))
    OUT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        contexts = {
            (r["conversation_id"], r["question"]): r
            for r in map(
                json.loads,
                (ROOT / "artifacts" / "contexts" / f"{split}.jsonl").read_text().splitlines(),
            )
        }
        dest = OUT / f"{split}.jsonl"
        done = set()
        if dest.exists():
            done = {episode_key(json.loads(line)) for line in dest.read_text().splitlines()}
        qa_pairs = getattr(splits, split)
        print(f"{split}: {len(qa_pairs)} QA, {len(done)} episodes already built", flush=True)
        skipped_evidence = skipped_facts = 0
        with dest.open("a") as sink:
            for i, qa in enumerate(qa_pairs, 1):
                bank = banks[qa.conversation_id]
                context = context_with_ids(
                    contexts[(qa.conversation_id, qa.question)]["memories"], bank
                )
                for dia_id in qa.evidence:
                    if (qa.conversation_id, qa.question, dia_id) in done:
                        continue
                    resolved = turns[qa.conversation_id].get(dia_id)
                    if resolved is None:
                        skipped_evidence += 1
                        continue
                    turn, date_time = resolved
                    facts = extract_facts(llm, turn, date_time)
                    if not facts:
                        skipped_facts += 1
                        continue
                    related = retriever.retrieve(bank, " ".join(facts), k=RELATED_K)
                    record = {
                        "conversation_id": qa.conversation_id,
                        "question": qa.question,
                        "answer": qa.answer,
                        "category": qa.category,
                        "evidence": dia_id,
                        "turn": {
                            "speaker": turn.speaker,
                            "text": turn.text,
                            "date_time": date_time,
                        },
                        "facts": facts,
                        "related": [
                            {"id": m.id, "text": m.text, "timestamp": m.timestamp} for m in related
                        ],
                        "context": context,
                    }
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
                print(f"{split} [{i}/{len(qa_pairs)}] {qa.conversation_id}", flush=True)
        total = len(dest.read_text().splitlines())
        print(
            f"{split}: wrote {dest} ({total} episodes; skipped {skipped_evidence} unresolved "
            f"evidence ids, {skipped_facts} turns with no extractable facts)"
        )


if __name__ == "__main__":
    main()
