from memory_r1.answer_agent import answer_question, format_memories, parse_answer
from memory_r1.judge import judge_answer
from memory_r1.memory_bank import MemoryEntry


def entry(id_: str, text: str, ts: str = "8 May 2023") -> MemoryEntry:
    return MemoryEntry(id=id_, text=text, timestamp=ts)


def test_format_memories_includes_timestamps():
    formatted = format_memories([entry("0", "Melanie has children")])
    assert formatted == "- [8 May 2023] Melanie has children"


def test_format_memories_empty():
    assert format_memories([]) == "(no memories)"


def test_parse_answer_takes_last_answer_line():
    raw = "Relevant: memory 3.\nAnswer: maybe this\nMore thought\nanswer: blue bike"
    assert parse_answer(raw) == "blue bike"


def test_parse_answer_falls_back_to_whole_output():
    assert parse_answer("  blue bike  ") == "blue bike"


def test_answer_question_builds_prompt_and_parses():
    seen = {}

    def llm(prompt: str) -> str:
        seen["prompt"] = prompt
        return "Memory 1 is relevant.\nAnswer: a blue bike"

    memories = [entry("0", "Caroline bought a blue bike")]
    result = answer_question(llm, memories, "What did Caroline buy?")
    assert result == "a blue bike"
    assert "Caroline bought a blue bike" in seen["prompt"]
    assert "What did Caroline buy?" in seen["prompt"]


def test_judge_yes_and_no():
    assert judge_answer(lambda p: "Yes, equivalent.", "q", "dog", "a dog") is True
    assert judge_answer(lambda p: "no", "q", "dog", "cat") is False


def test_judge_prompt_contains_fields():
    seen = {}

    def llm(prompt: str) -> str:
        seen["prompt"] = prompt
        return "yes"

    judge_answer(llm, "What pet?", "dog", "a dog")
    assert "What pet?" in seen["prompt"]
    assert "Gold answer: dog" in seen["prompt"]
    assert "Model answer: a dog" in seen["prompt"]
