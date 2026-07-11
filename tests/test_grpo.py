import json

import pytest

from memory_r1.answer_agent import ANSWER_PROMPT, format_memories
from memory_r1.grpo import answer_reward, build_prompt, load_contexts, make_trl_reward
from memory_r1.memory_bank import MemoryEntry

MEMORIES = [
    {"text": "Caroline bought a blue bike", "timestamp": "8 May 2023"},
    {"text": "Melanie lives in a big city", "timestamp": "9 May 2023"},
]


def test_build_prompt_matches_eval_harness_format():
    entries = [
        MemoryEntry(id="0", text=MEMORIES[0]["text"], timestamp=MEMORIES[0]["timestamp"]),
        MemoryEntry(id="1", text=MEMORIES[1]["text"], timestamp=MEMORIES[1]["timestamp"]),
    ]
    expected = ANSWER_PROMPT.format(
        memories=format_memories(entries), question="What did Caroline buy?"
    )
    assert build_prompt(MEMORIES, "What did Caroline buy?") == expected


def test_answer_reward_parses_answer_line():
    completion = "Memory 1 is relevant.\nAnswer: a blue bike"
    assert answer_reward(completion, "a blue bike") == 1.0
    assert answer_reward(completion, "red car") == 0.0


def test_answer_reward_f1_metric():
    completion = "Answer: 8 May 2023"
    assert answer_reward(completion, "May 2023", metric="f1") == pytest.approx(0.8)


def test_answer_reward_unknown_metric_raises():
    with pytest.raises(KeyError):
        answer_reward("Answer: x", "x", metric="bleu")


def test_make_trl_reward_batches():
    reward = make_trl_reward("em")
    scores = reward(
        completions=["Answer: dog", "Answer: cat"],
        answer=["dog", "dog"],
        extra_column=["ignored", "ignored"],
    )
    assert scores == [1.0, 0.0]
    assert reward.__name__ == "answer_reward_em"


def test_make_trl_reward_unwraps_conversational_completions():
    reward = make_trl_reward("em")
    scores = reward(
        completions=[[{"role": "assistant", "content": "Answer: dog"}]],
        answer=["dog"],
    )
    assert scores == [1.0]


def test_load_contexts_roundtrip(tmp_path):
    records = [
        {
            "conversation_id": "conv-1",
            "question": "q",
            "answer": "a",
            "category": 4,
            "memories": MEMORIES,
        }
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert load_contexts(path) == records
