import json

import pytest

from memory_r1.manager import (
    build_manager_prompt,
    episode_valid_ids,
    load_episodes,
    make_manager_trl_reward,
    manager_reward,
    splice_context,
)
from memory_r1.memory_bank import MemoryOperation, OperationError

CONTEXT = [
    {"id": "3", "text": "Caroline bought a blue bike", "timestamp": "8 May 2023"},
    {"id": "7", "text": "Melanie lives in a big city", "timestamp": "9 May 2023"},
]

EPISODE = {
    "conversation_id": "conv-1",
    "question": "What did Caroline buy?",
    "answer": "a red bike",
    "category": 4,
    "evidence": "D1:3",
    "turn": {"speaker": "Caroline", "text": "...", "date_time": "10 May 2023"},
    "facts": ["Caroline exchanged her bike for a red one"],
    "related": [{"id": "3", "text": "Caroline bought a blue bike", "timestamp": "8 May 2023"}],
    "context": CONTEXT,
}

VALID_IDS = {"3", "7"}


def test_splice_add_appends_with_default_timestamp():
    ops = [MemoryOperation(op="ADD", text="Caroline rides daily")]
    spliced = splice_context(CONTEXT, ops, valid_ids=VALID_IDS, default_timestamp="10 May 2023")
    assert spliced[-1] == {"id": None, "text": "Caroline rides daily", "timestamp": "10 May 2023"}
    assert CONTEXT[0]["text"] == "Caroline bought a blue bike"  # input untouched


def test_splice_update_replaces_in_place():
    ops = [MemoryOperation(op="UPDATE", id="3", text="Caroline now has a red bike")]
    spliced = splice_context(CONTEXT, ops, valid_ids=VALID_IDS)
    assert spliced[0]["text"] == "Caroline now has a red bike"
    assert spliced[0]["timestamp"] == "8 May 2023"  # kept when the op has none


def test_splice_update_of_unretrieved_valid_id_appends():
    ops = [MemoryOperation(op="UPDATE", id="9", text="new text")]
    spliced = splice_context(CONTEXT, ops, valid_ids=VALID_IDS | {"9"}, default_timestamp="t")
    assert spliced[-1] == {"id": "9", "text": "new text", "timestamp": "t"}


def test_splice_delete_removes_entry():
    ops = [MemoryOperation(op="DELETE", id="7")]
    spliced = splice_context(CONTEXT, ops, valid_ids=VALID_IDS)
    assert [m["id"] for m in spliced] == ["3"]


def test_splice_delete_of_unretrieved_valid_id_is_noop():
    ops = [MemoryOperation(op="DELETE", id="9")]
    spliced = splice_context(CONTEXT, ops, valid_ids=VALID_IDS | {"9"})
    assert spliced == CONTEXT


def test_splice_noop_changes_nothing():
    spliced = splice_context(CONTEXT, [MemoryOperation(op="NOOP")], valid_ids=VALID_IDS)
    assert spliced == CONTEXT


@pytest.mark.parametrize(
    "op",
    [
        MemoryOperation(op="UPDATE", id="99", text="x"),
        MemoryOperation(op="DELETE", id="99"),
        MemoryOperation(op="UPDATE", id=None, text="x"),
        MemoryOperation(op="ADD"),
        MemoryOperation(op="UPDATE", id="3"),
    ],
)
def test_splice_invalid_ops_raise(op):
    with pytest.raises(OperationError):
        splice_context(CONTEXT, [op], valid_ids=VALID_IDS)


def test_episode_valid_ids_unions_related_and_context():
    episode = dict(EPISODE, related=[{"id": "12", "text": "t", "timestamp": None}])
    episode["context"] = CONTEXT + [{"id": None, "text": "unmatched", "timestamp": None}]
    assert episode_valid_ids(episode) == {"3", "7", "12"}


def test_build_manager_prompt_shows_facts_and_ids():
    prompt = build_manager_prompt(EPISODE)
    assert "- Caroline exchanged her bike for a red one" in prompt
    assert "- id=3 [8 May 2023] Caroline bought a blue bike" in prompt
    assert "10 May 2023" in prompt


def test_manager_reward_scores_spliced_answer():
    def answer_llm(prompt: str) -> str:
        assert "Caroline now has a red bike" in prompt  # UPDATE reached the context
        return "Answer: a red bike"

    completion = json.dumps(
        {"operations": [{"operation": "UPDATE", "id": "3", "text": "Caroline now has a red bike"}]}
    )
    assert manager_reward(completion, EPISODE, answer_llm, metric="f1") == 1.0


def test_manager_reward_zero_on_unparseable_completion():
    assert manager_reward("no json here", EPISODE, lambda p: "Answer: a red bike") == 0.0


def test_manager_reward_zero_on_unknown_id():
    completion = json.dumps({"operations": [{"operation": "DELETE", "id": "99"}]})
    assert manager_reward(completion, EPISODE, lambda p: "Answer: a red bike") == 0.0


def test_manager_reward_noop_answers_from_frozen_context():
    def answer_llm(prompt: str) -> str:
        assert "Caroline bought a blue bike" in prompt
        return "Answer: a blue bike"

    completion = json.dumps({"operations": [{"operation": "NOOP"}]})
    reward = manager_reward(completion, EPISODE, answer_llm, metric="f1")
    assert reward == pytest.approx(0.5)  # "blue bike" vs "red bike": 1 of 2 tokens


def test_make_manager_trl_reward_batches_and_unwraps():
    def answer_batch(prompts: list[str]) -> list[str]:
        return ["Answer: a red bike"] * len(prompts)

    reward = make_manager_trl_reward(answer_batch, metric="f1")
    completions = [
        [{"role": "assistant", "content": json.dumps({"operations": [{"operation": "NOOP"}]})}],
        "not json",
    ]
    scores = reward(completions=completions, episode=[EPISODE, EPISODE], extra=[0, 0])
    assert scores == [1.0, 0.0]
    assert reward.__name__ == "manager_reward_f1"


def test_make_manager_trl_reward_only_valid_completions_reach_answerer():
    seen: list[list[str]] = []

    def answer_batch(prompts: list[str]) -> list[str]:
        seen.append(prompts)
        return ["Answer: a red bike"] * len(prompts)

    reward = make_manager_trl_reward(answer_batch, metric="f1")
    noop = json.dumps({"operations": [{"operation": "NOOP"}]})
    bad = json.dumps({"operations": [{"operation": "DELETE", "id": "999"}]})
    scores = reward(completions=["garbage", noop, bad, noop], episode=[EPISODE] * 4)
    assert scores == [0.0, 1.0, 0.0, 1.0]  # slot mapping survives invalid gaps
    assert len(seen) == 1 and len(seen[0]) == 2  # one batched call, invalid never sent


def test_make_manager_trl_reward_all_invalid_skips_answerer():
    def answer_batch(prompts: list[str]) -> list[str]:
        raise AssertionError("should not be called")

    reward = make_manager_trl_reward(answer_batch, metric="f1")
    assert reward(completions=["nope"], episode=[EPISODE]) == [0.0]


def test_load_episodes_roundtrip(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(EPISODE) + "\n")
    assert load_episodes(path) == [EPISODE]


def test_apply_operations_defaults_add_timestamp_and_skips_invalid():
    from memory_r1.manager import apply_operations
    from memory_r1.memory_bank import MemoryBank

    bank = MemoryBank()
    kept = bank.add("Caroline bought a blue bike", timestamp="8 May 2023")
    ops = [
        MemoryOperation(op="ADD", text="Caroline rides daily"),
        MemoryOperation(op="UPDATE", id=kept.id, text="Caroline has a red bike"),
        MemoryOperation(op="DELETE", id="404"),  # unknown -> skipped, not fatal
        MemoryOperation(op="NOOP"),
    ]
    applied, skipped = apply_operations(bank, ops, default_timestamp="10 May 2023")
    assert (applied, skipped) == (3, 1)
    assert bank.get(kept.id).text == "Caroline has a red bike"
    added = [e for e in bank.entries if e.text == "Caroline rides daily"]
    assert added[0].timestamp == "10 May 2023"
