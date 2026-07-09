import math

import pytest

from memory_r1.metrics import bleu_1, exact_match, f1_score, normalize_answer


def test_normalize_strips_case_punctuation_articles():
    assert normalize_answer("The  Cat, sat!") == "cat sat"


def test_exact_match_after_normalization():
    assert exact_match("A dog.", "dog") == 1.0
    assert exact_match("cat", "dog") == 0.0


def test_f1_perfect_and_zero():
    assert f1_score("blue bike", "blue bike") == 1.0
    assert f1_score("red car", "blue bike") == 0.0


def test_f1_partial_overlap():
    # pred: {8, may, 2023}, gold: {may, 2023} -> p=2/3, r=1 -> f1=0.8
    assert f1_score("8 May 2023", "May 2023") == pytest.approx(0.8)


def test_f1_empty_cases():
    assert f1_score("", "") == 1.0
    assert f1_score("", "dog") == 0.0
    assert f1_score("the", "dog") == 0.0  # normalizes to empty prediction


def test_f1_repeated_tokens_clipped():
    # pred {dog: 2} vs gold {dog: 1}: overlap 1, p=0.5, r=1 -> 2/3
    assert f1_score("dog dog", "dog") == pytest.approx(2 / 3)


def test_bleu1_no_penalty_when_longer():
    # pred longer than gold: precision 2/3, BP 1
    assert bleu_1("blue bike helmet", "blue bike") == pytest.approx(2 / 3)


def test_bleu1_brevity_penalty_when_shorter():
    # pred 1 token vs gold 2: precision 1, BP exp(1-2)
    assert bleu_1("blue", "blue bike") == pytest.approx(math.exp(-1))


def test_bleu1_zero_overlap():
    assert bleu_1("red", "blue bike") == 0.0
