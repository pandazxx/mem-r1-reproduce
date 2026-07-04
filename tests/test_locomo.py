import pytest

from memory_r1.locomo import load_locomo, make_splits


def test_load_conversations(locomo_path):
    conversations = load_locomo(locomo_path)
    assert [c.sample_id for c in conversations] == ["conv-1", "conv-2"]

    conv = conversations[0]
    assert conv.speaker_a == "Alice"
    assert conv.speaker_b == "Bob"
    assert [s.index for s in conv.sessions] == [1, 2]
    assert conv.sessions[0].date_time == "1:56 pm on 8 May, 2023"
    assert conv.sessions[0].turns[0].text == "I adopted a dog named Rex."
    assert conv.sessions[1].turns[0].dia_id == "D2:1"


def test_load_image_turn(locomo_path):
    turn = load_locomo(locomo_path)[0].sessions[0].turns[1]
    assert turn.img_urls == ("http://example.com/x.jpg",)
    assert turn.blip_caption == "a photo of a red bicycle"


def test_load_qa(locomo_path):
    conv = load_locomo(locomo_path)[0]
    assert len(conv.qa) == 3
    qa = conv.qa[0]
    assert qa.answer == "Rex"
    assert qa.category == 4
    assert qa.category_name == "single-hop"
    assert qa.conversation_id == "conv-1"
    # category 5 uses adversarial_answer
    assert conv.qa[2].answer == "Not mentioned"


def test_non_string_answer_is_coerced(locomo_path):
    conv = load_locomo(locomo_path)[1]
    assert conv.qa[1].answer == "1"


def test_splits_exclude_adversarial_and_are_deterministic(locomo_path):
    conversations = load_locomo(locomo_path)
    splits = make_splits(conversations, seed=7, train_size=2, val_size=1)
    all_qa = [*splits.train, *splits.val, *splits.test]
    assert len(splits.train) == 2
    assert len(splits.val) == 1
    assert len(splits.test) == 1
    assert all(q.category != 5 for q in all_qa)
    assert len({q.question for q in all_qa}) == 4

    again = make_splits(conversations, seed=7, train_size=2, val_size=1)
    assert again == splits


def test_splits_too_large_raises(locomo_path):
    conversations = load_locomo(locomo_path)
    with pytest.raises(ValueError):
        make_splits(conversations, train_size=100, val_size=1)
