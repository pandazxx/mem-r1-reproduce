import json

import pytest


def _qa(question, answer, category, evidence=("D1:1",)):
    item = {"question": question, "category": category, "evidence": list(evidence)}
    if category == 5:
        item["adversarial_answer"] = answer
    else:
        item["answer"] = answer
    return item


@pytest.fixture
def locomo_raw():
    return [
        {
            "sample_id": "conv-1",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "1:56 pm on 8 May, 2023",
                "session_1": [
                    {"speaker": "Alice", "dia_id": "D1:1", "text": "I adopted a dog named Rex."},
                    {
                        "speaker": "Bob",
                        "dia_id": "D1:2",
                        "text": "Look at this!",
                        "img_url": ["http://example.com/x.jpg"],
                        "blip_caption": "a photo of a red bicycle",
                    },
                ],
                "session_2_date_time": "10:00 am on 9 May, 2023",
                "session_2": [
                    {"speaker": "Alice", "dia_id": "D2:1", "text": "Rex chewed my shoes today."},
                ],
            },
            "qa": [
                _qa("What is Alice's dog called?", "Rex", 4),
                _qa("When did Alice adopt Rex?", "8 May 2023", 2),
                _qa("Does Alice own a cat?", "Not mentioned", 5),
            ],
        },
        {
            "sample_id": "conv-2",
            "conversation": {
                "speaker_a": "Carol",
                "speaker_b": "Dan",
                "session_1_date_time": "3:00 pm on 1 June, 2023",
                "session_1": [
                    {"speaker": "Carol", "dia_id": "D1:1", "text": "I started a pottery class."},
                ],
            },
            "qa": [
                _qa("What class did Carol start?", "Pottery", 4),
                _qa("How many wheels does a pottery wheel have?", 1, 3),
            ],
        },
    ]


@pytest.fixture
def locomo_path(tmp_path, locomo_raw):
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps(locomo_raw))
    return path
