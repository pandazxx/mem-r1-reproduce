"""LLM-as-a-Judge correctness verdict, as in the paper's third metric."""

from __future__ import annotations

from memory_r1.bootstrap import LLMFn

JUDGE_PROMPT = """\
You are grading a question-answering system.

Question: {question}
Gold answer: {gold}
Model answer: {prediction}

Is the model answer correct? It counts as correct if it conveys the same \
meaning as the gold answer, even with different wording, formatting, or extra \
detail. Reply with exactly one word: yes or no.
"""


def judge_answer(llm: LLMFn, question: str, gold: str, prediction: str) -> bool:
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, prediction=prediction)
    verdict = llm(prompt).strip().lower()
    return verdict.startswith("yes")
