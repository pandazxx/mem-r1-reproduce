"""LoCoMo evaluation of an answer pipeline over per-conversation memory banks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean

from memory_r1.answer_agent import answer_question
from memory_r1.bootstrap import LLMFn
from memory_r1.judge import judge_answer
from memory_r1.locomo import CATEGORY_NAMES, QAPair
from memory_r1.memory_bank import MemoryBank, MemoryEntry
from memory_r1.metrics import bleu_1, exact_match, f1_score
from memory_r1.retrieval import DEFAULT_TOP_K, Retriever


@dataclass(frozen=True)
class QAResult:
    conversation_id: str
    question: str
    gold: str
    prediction: str
    category: int
    em: float
    f1: float
    bleu1: float
    judge: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def result_key(conversation_id: str, question: str) -> str:
    return f"{conversation_id}\x1f{question}"


def evaluate_answer(
    qa: QAPair,
    memories: list[MemoryEntry],
    llm: LLMFn,
    judge_llm: LLMFn | None = None,
) -> QAResult:
    prediction = answer_question(llm, memories, qa.question)
    verdict = None
    if judge_llm is not None:
        verdict = judge_answer(judge_llm, qa.question, qa.answer, prediction)
    return QAResult(
        conversation_id=qa.conversation_id,
        question=qa.question,
        gold=qa.answer,
        prediction=prediction,
        category=qa.category,
        em=exact_match(prediction, qa.answer),
        f1=f1_score(prediction, qa.answer),
        bleu1=bleu_1(prediction, qa.answer),
        judge=verdict,
    )


def evaluate_question(
    qa: QAPair,
    bank: MemoryBank,
    retriever: Retriever,
    llm: LLMFn,
    judge_llm: LLMFn | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> QAResult:
    memories = retriever.retrieve(bank, qa.question, k=top_k)
    return evaluate_answer(qa, memories, llm, judge_llm=judge_llm)


def _aggregate(results: list[QAResult]) -> dict:
    judged = [r.judge for r in results if r.judge is not None]
    return {
        "n": len(results),
        "em": mean(r.em for r in results),
        "f1": mean(r.f1 for r in results),
        "bleu1": mean(r.bleu1 for r in results),
        "judge": mean(judged) if judged else None,
    }


def summarize(results: list[QAResult]) -> dict:
    if not results:
        return {"overall": None, "by_category": {}}
    by_category = {}
    for category in sorted({r.category for r in results}):
        subset = [r for r in results if r.category == category]
        by_category[CATEGORY_NAMES[category]] = _aggregate(subset)
    return {"overall": _aggregate(results), "by_category": by_category}
