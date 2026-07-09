"""Frozen (prompted, no-RL) Answer Agent.

Mirrors the paper's Answer Agent structure: given ~60 retrieved memories,
first distill the relevant ones, then answer. The RL-trained agent (M3) uses
the same prompt shape so baseline vs. trained numbers are comparable.
"""

from __future__ import annotations

from memory_r1.bootstrap import LLMFn
from memory_r1.memory_bank import MemoryEntry

ANSWER_PROMPT = """\
You answer questions about a long-running conversation between two people, \
using only the memories below. Each memory is prefixed with the date/time of \
the conversation session it was learned in.

Memories:
{memories}

Question: {question}

First, briefly identify which memories are relevant to the question. Then give \
your final answer on the last line in exactly this format:

Answer: <short answer of a few words>

Keep the final answer as short as possible (a name, date, phrase, or list). \
If the memories do not contain the answer, write: Answer: No information available
"""


def format_memories(memories: list[MemoryEntry]) -> str:
    if not memories:
        return "(no memories)"
    return "\n".join(f"- [{m.timestamp}] {m.text}" for m in memories)


def parse_answer(raw: str) -> str:
    for line in reversed(raw.strip().splitlines()):
        stripped = line.strip()
        if stripped.lower().startswith("answer:"):
            return stripped[len("answer:") :].strip()
    return raw.strip()


def answer_question(llm: LLMFn, memories: list[MemoryEntry], question: str) -> str:
    prompt = ANSWER_PROMPT.format(memories=format_memories(memories), question=question)
    return parse_answer(llm(prompt))
