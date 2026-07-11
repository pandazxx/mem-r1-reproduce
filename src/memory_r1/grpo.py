"""Reward and dataset plumbing for GRPO Answer Agent training (M3).

Pure Python (no torch/trl imports) so rewards stay unit-testable on the
no-GPU workspace. Prompts are built with the same template as the M2 eval
harness, so frozen-baseline and trained-model numbers are comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

from memory_r1.answer_agent import ANSWER_PROMPT, format_memories, parse_answer
from memory_r1.memory_bank import MemoryEntry
from memory_r1.metrics import exact_match, f1_score

REWARD_METRICS = {"em": exact_match, "f1": f1_score}


def build_prompt(memories: list[dict], question: str) -> str:
    entries = [
        MemoryEntry(id=str(i), text=m["text"], timestamp=m["timestamp"])
        for i, m in enumerate(memories)
    ]
    return ANSWER_PROMPT.format(memories=format_memories(entries), question=question)


def load_contexts(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def answer_reward(completion: str, gold: str, *, metric: str = "em") -> float:
    return REWARD_METRICS[metric](parse_answer(completion), gold)


def _completion_text(completion) -> str:
    # conversational datasets: TRL passes [{"role": "assistant", "content": ...}]
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion


def make_trl_reward(metric: str = "em"):
    """TRL reward function: completions + the dataset's ``answer`` column."""

    def reward(completions: list, answer: list[str], **kwargs) -> list[float]:
        return [
            answer_reward(_completion_text(completion), gold, metric=metric)
            for completion, gold in zip(completions, answer, strict=True)
        ]

    reward.__name__ = f"answer_reward_{metric}"
    return reward
