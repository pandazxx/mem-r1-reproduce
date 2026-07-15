"""Precompute Memory Manager inputs for every dialogue turn (M4 bank rebuild).

For each of the ~5.9k LoCoMo turns: re-extract facts (same prompt as the M1
bootstrap) and retrieve the top-6 related memories from the M1 bank
*restricted to sessions up to that turn* (entries carry their session
timestamp, so the "bank so far" is recoverable per session). The GPU rebuild
(scripts/rebuild_banks_with_manager.py) then needs no API access.

Writes artifacts/turn_inputs/<sample_id>.jsonl (committed). Needs the default
provider (NIM free tier). Resumable per turn; turns with no extractable facts
are recorded with empty facts so the rebuild can skip them without recounting.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_r1.bootstrap import extract_facts
from memory_r1.locomo import load_locomo
from memory_r1.manager import RELATED_K
from memory_r1.memory_bank import MemoryBank
from memory_r1.providers import get_provider, make_embedder, make_llm
from memory_r1.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "turn_inputs"


def bank_through_session(bank: MemoryBank, session_dates: list[str]) -> MemoryBank:
    """The M1 bank restricted to entries from the given sessions (in order)."""
    allowed = set(session_dates)
    view = MemoryBank()
    for entry in bank.entries:
        if entry.timestamp in allowed:
            view._entries[entry.id] = entry  # keep original ids for UPDATE/DELETE targets
    return view


def main() -> None:
    provider = get_provider()
    print(f"provider: {provider.name} ({provider.chat_model} / {provider.embedding_model})")
    llm = make_llm(provider)
    retriever = Retriever(make_embedder(provider))
    conversations = load_locomo(ROOT / "data" / "locomo10.json")
    OUT.mkdir(parents=True, exist_ok=True)

    for conv in conversations:
        bank = MemoryBank.load(ROOT / "artifacts" / "memory_banks" / f"{conv.sample_id}.json")
        dest = OUT / f"{conv.sample_id}.jsonl"
        done = set()
        if dest.exists():
            done = {json.loads(line)["dia_id"] for line in dest.read_text().splitlines()}
        n_turns = sum(len(s.turns) for s in conv.sessions)
        print(f"{conv.sample_id}: {n_turns} turns, {len(done)} already built", flush=True)
        session_dates: list[str] = []
        with dest.open("a") as sink:
            for session in conv.sessions:
                session_dates.append(session.date_time)
                so_far = bank_through_session(bank, session_dates)
                for turn in session.turns:
                    if turn.dia_id in done:
                        continue
                    facts = extract_facts(llm, turn, session.date_time)
                    related = (
                        retriever.retrieve(so_far, " ".join(facts), k=RELATED_K) if facts else []
                    )
                    record = {
                        "conversation_id": conv.sample_id,
                        "dia_id": turn.dia_id,
                        "session_index": session.index,
                        "turn": {
                            "speaker": turn.speaker,
                            "text": turn.text,
                            "date_time": session.date_time,
                        },
                        "facts": facts,
                        "related": [
                            {"id": m.id, "text": m.text, "timestamp": m.timestamp} for m in related
                        ],
                    }
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
                print(f"{conv.sample_id} session {session.index} done", flush=True)
        print(f"{conv.sample_id}: wrote {dest} ({len(dest.read_text().splitlines())} turns)")


if __name__ == "__main__":
    main()
