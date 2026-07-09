"""Bootstrap initial temporal memory banks from LoCoMo dialogues.

Per the paper (Appendix B.2), GPT-4o-mini extracts facts per dialogue turn
and each memory entry carries the session timestamp. The LLM is injected as
a plain ``Callable[[str], str]`` so the pipeline is testable without network.
"""

from __future__ import annotations

import json
from typing import Callable

from memory_r1.locomo import Conversation, Turn
from memory_r1.memory_bank import MemoryBank

LLMFn = Callable[[str], str]
"""Prompt-in, completion-out. Build one with providers.make_llm()."""

FACT_EXTRACTION_PROMPT = """\
You extract personal facts from one turn of a dialogue for a long-term memory system.

Extract the salient, self-contained facts about the speaker (or people they mention)
from the turn below. Write each fact in third person, mentioning the speaker by name.
Ignore greetings, filler, and questions that carry no information. If there is nothing
worth remembering, return an empty list.

Dialogue turn (spoken by {speaker} at {date_time}):
{text}

Respond with only JSON in this exact format:
{{"facts": ["<fact 1>", "<fact 2>", ...]}}"""


def extract_facts(llm: LLMFn, turn: Turn, date_time: str) -> list[str]:
    text = turn.text
    if turn.blip_caption:
        text = f"{text} [shared an image: {turn.blip_caption}]"
    if not text.strip():
        return []
    prompt = FACT_EXTRACTION_PROMPT.format(speaker=turn.speaker, date_time=date_time, text=text)
    raw = llm(prompt)
    try:
        payload = json.loads(raw)
        facts = payload["facts"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
    return [str(f) for f in facts if str(f).strip()]


def bootstrap_memory_bank(llm: LLMFn, conversation: Conversation) -> MemoryBank:
    bank = MemoryBank()
    for session in conversation.sessions:
        for turn in session.turns:
            for fact in extract_facts(llm, turn, session.date_time):
                bank.add(fact, timestamp=session.date_time)
    return bank
