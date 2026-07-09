import pytest

from memory_r1.evaluation import QAResult, evaluate_question, result_key, summarize
from memory_r1.locomo import QAPair
from memory_r1.memory_bank import MemoryBank
from memory_r1.retrieval import Retriever


class KeywordEmbedder:
    VOCAB = ["bike", "dog", "city"]

    def embed(self, texts, *, kind="passage"):
        return [[1.0 if word in text.lower() else 0.0 for word in self.VOCAB] for text in texts]


def make_qa(question="What did Caroline buy?", answer="a blue bike", category=4):
    return QAPair(
        conversation_id="conv-1",
        question=question,
        answer=answer,
        evidence=(),
        category=category,
    )


def make_bank():
    bank = MemoryBank()
    bank.add("Caroline bought a blue bike", timestamp="8 May 2023")
    bank.add("Melanie lives in a big city", timestamp="9 May 2023")
    return bank


def test_evaluate_question_end_to_end():
    def llm(prompt):
        assert "Caroline bought a blue bike" in prompt
        return "Relevant: memory about the bike.\nAnswer: a blue bike"

    result = evaluate_question(make_qa(), make_bank(), Retriever(KeywordEmbedder()), llm)
    assert result.prediction == "a blue bike"
    assert result.em == 1.0
    assert result.f1 == 1.0
    assert result.judge is None


def test_evaluate_question_with_judge():
    def llm(prompt):
        return "Answer: blue bicycle"

    def judge_llm(prompt):
        assert "Gold answer: a blue bike" in prompt
        return "yes"

    result = evaluate_question(
        make_qa(), make_bank(), Retriever(KeywordEmbedder()), llm, judge_llm=judge_llm
    )
    assert result.em == 0.0
    assert result.judge is True


def test_evaluate_question_respects_top_k():
    seen = {}

    def llm(prompt):
        seen["prompt"] = prompt
        return "Answer: x"

    evaluate_question(make_qa(), make_bank(), Retriever(KeywordEmbedder()), llm, top_k=1)
    assert "Caroline bought a blue bike" in seen["prompt"]
    assert "Melanie lives in a big city" not in seen["prompt"]


def make_result(category=4, em=1.0, f1=1.0, bleu1=1.0, judge=None):
    return QAResult(
        conversation_id="conv-1",
        question="q",
        gold="g",
        prediction="p",
        category=category,
        em=em,
        f1=f1,
        bleu1=bleu1,
        judge=judge,
    )


def test_summarize_overall_and_per_category():
    results = [
        make_result(category=4, f1=1.0, judge=True),
        make_result(category=4, f1=0.0, judge=False),
        make_result(category=2, f1=0.5, judge=True),
    ]
    summary = summarize(results)
    assert summary["overall"]["n"] == 3
    assert summary["overall"]["f1"] == pytest.approx(0.5)
    assert summary["overall"]["judge"] == pytest.approx(2 / 3)
    assert summary["by_category"]["single-hop"]["n"] == 2
    assert summary["by_category"]["temporal"]["f1"] == pytest.approx(0.5)


def test_summarize_no_judge_and_empty():
    summary = summarize([make_result(judge=None)])
    assert summary["overall"]["judge"] is None
    assert summarize([]) == {"overall": None, "by_category": {}}


def test_result_key_distinguishes_conversations():
    assert result_key("conv-1", "q") != result_key("conv-2", "q")


def test_result_roundtrips_through_dict():
    result = make_result(judge=True)
    assert QAResult(**result.to_dict()) == result
