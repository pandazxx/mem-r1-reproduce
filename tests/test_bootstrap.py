import json

from memory_r1.bootstrap import bootstrap_memory_bank, extract_facts
from memory_r1.locomo import Turn, load_locomo


def fake_llm(prompt: str) -> str:
    if "shoes" in prompt:
        return json.dumps({"facts": ["Rex chewed Alice's shoes"]})
    if "Rex" in prompt:
        return json.dumps({"facts": ["Alice adopted a dog named Rex"]})
    if "bicycle" in prompt:
        return json.dumps({"facts": ["Bob shared a photo of a red bicycle"]})
    return json.dumps({"facts": []})


def test_extract_facts_includes_speaker_and_time():
    seen = {}

    def llm(prompt):
        seen["prompt"] = prompt
        return '{"facts": ["Alice adopted a dog"]}'

    turn = Turn(speaker="Alice", dia_id="D1:1", text="I adopted a dog!")
    facts = extract_facts(llm, turn, "8 May, 2023")
    assert facts == ["Alice adopted a dog"]
    assert "Alice" in seen["prompt"]
    assert "8 May, 2023" in seen["prompt"]


def test_extract_facts_appends_image_caption():
    seen = {}

    def llm(prompt):
        seen["prompt"] = prompt
        return '{"facts": []}'

    turn = Turn(speaker="Bob", dia_id="D1:2", text="Look!", blip_caption="a red bicycle")
    extract_facts(llm, turn, "8 May, 2023")
    assert "a red bicycle" in seen["prompt"]


def test_extract_facts_empty_turn_skips_llm():
    def llm(prompt):
        raise AssertionError("should not be called")

    turn = Turn(speaker="Alice", dia_id="D1:1", text="   ")
    assert extract_facts(llm, turn, "8 May, 2023") == []


def test_extract_facts_bad_llm_output_returns_empty():
    turn = Turn(speaker="Alice", dia_id="D1:1", text="I adopted a dog!")
    assert extract_facts(lambda p: "not json", turn, "8 May, 2023") == []
    assert extract_facts(lambda p: '{"wrong": []}', turn, "8 May, 2023") == []


def test_bootstrap_memory_bank(locomo_path):
    conv = load_locomo(locomo_path)[0]
    bank = bootstrap_memory_bank(fake_llm, conv)
    assert [e.text for e in bank.entries] == [
        "Alice adopted a dog named Rex",
        "Bob shared a photo of a red bicycle",
        "Rex chewed Alice's shoes",
    ]
    assert bank.entries[0].timestamp == "1:56 pm on 8 May, 2023"
    assert bank.entries[2].timestamp == "10:00 am on 9 May, 2023"
