"""Pure-Python answer-quality metrics (SQuAD-style F1/EM, BLEU-1).

BLEU-1 matches nltk ``sentence_bleu`` with weights ``(1, 0, 0, 0)``:
clipped unigram precision times the brevity penalty.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu_1(prediction: str, gold: str) -> float:
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    precision = sum(common.values()) / len(pred_tokens)
    if precision == 0:
        return 0.0
    if len(pred_tokens) >= len(gold_tokens):
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - len(gold_tokens) / len(pred_tokens))
    return brevity_penalty * precision
