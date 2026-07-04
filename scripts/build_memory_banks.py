"""Build initial memory banks for all LoCoMo conversations with GPT-4o-mini.

Requires OPENAI_API_KEY. Writes one bank per conversation to
data/memory_banks/<sample_id>.json; existing banks are skipped so the
script is resumable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_r1.bootstrap import bootstrap_memory_bank, openai_llm
from memory_r1.locomo import load_locomo

DATA = Path(__file__).resolve().parents[1] / "data"
BANKS = DATA / "memory_banks"


def main() -> None:
    conversations = load_locomo(DATA / "locomo10.json")
    BANKS.mkdir(parents=True, exist_ok=True)
    llm = openai_llm()
    for conv in conversations:
        dest = BANKS / f"{conv.sample_id}.json"
        if dest.exists():
            print(f"{conv.sample_id}: exists, skipping")
            continue
        n_turns = sum(len(s.turns) for s in conv.sessions)
        print(f"{conv.sample_id}: extracting facts from {n_turns} turns...")
        bank = bootstrap_memory_bank(llm, conv)
        bank.save(dest)
        print(f"{conv.sample_id}: saved {len(bank)} memories -> {dest}")


if __name__ == "__main__":
    main()
