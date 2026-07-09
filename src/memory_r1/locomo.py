"""LoCoMo dataset loading and Memory-R1 train/val/test splits.

LoCoMo ships as a single ``locomo10.json``: a list of 10 multi-session
conversations, each with QA annotations. Memory-R1 excludes the adversarial
QA subset (category 5) and splits the remaining 1540 questions into
152 train / 81 val / 1307 test.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
ADVERSARIAL_CATEGORY = 5

_SESSION_KEY = re.compile(r"^session_(\d+)$")


@dataclass(frozen=True)
class Turn:
    speaker: str
    dia_id: str
    text: str
    img_urls: tuple[str, ...] = ()
    blip_caption: str | None = None


@dataclass(frozen=True)
class Session:
    index: int
    date_time: str
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class QAPair:
    conversation_id: str
    question: str
    answer: str
    evidence: tuple[str, ...]
    category: int

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES[self.category]


@dataclass(frozen=True)
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: tuple[Session, ...]
    qa: tuple[QAPair, ...]


@dataclass(frozen=True)
class Splits:
    train: tuple[QAPair, ...]
    val: tuple[QAPair, ...]
    test: tuple[QAPair, ...]


def _parse_turn(raw: dict) -> Turn:
    img_urls = raw.get("img_url", ())
    if isinstance(img_urls, str):
        img_urls = (img_urls,)
    return Turn(
        speaker=raw["speaker"],
        dia_id=raw["dia_id"],
        text=raw.get("text", ""),
        img_urls=tuple(img_urls),
        blip_caption=raw.get("blip_caption"),
    )


def _parse_sessions(conversation: dict) -> tuple[Session, ...]:
    sessions = []
    for key, value in conversation.items():
        match = _SESSION_KEY.match(key)
        if not match:
            continue
        index = int(match.group(1))
        date_time = conversation.get(f"session_{index}_date_time", "")
        sessions.append(
            Session(index=index, date_time=date_time, turns=tuple(_parse_turn(t) for t in value))
        )
    return tuple(sorted(sessions, key=lambda s: s.index))


def _parse_qa(raw: dict, conversation_id: str) -> QAPair:
    category = int(raw["category"])
    answer = raw.get("answer", raw.get("adversarial_answer", ""))
    return QAPair(
        conversation_id=conversation_id,
        question=raw["question"],
        answer=str(answer),
        evidence=tuple(raw.get("evidence", ())),
        category=category,
    )


def load_locomo(path: str | Path) -> list[Conversation]:
    raw = json.loads(Path(path).read_text())
    conversations = []
    for sample in raw:
        sample_id = sample["sample_id"]
        conv = sample["conversation"]
        conversations.append(
            Conversation(
                sample_id=sample_id,
                speaker_a=conv["speaker_a"],
                speaker_b=conv["speaker_b"],
                sessions=_parse_sessions(conv),
                qa=tuple(_parse_qa(q, sample_id) for q in sample["qa"]),
            )
        )
    return conversations


def make_splits(
    conversations: list[Conversation],
    *,
    seed: int = 42,
    train_size: int = 152,
    val_size: int = 81,
) -> Splits:
    """Drop adversarial QA and split the rest into train/val/test.

    The paper does not publish its exact partition, only the sizes
    (152/81/1307 of the 1540 non-adversarial questions), so we use a
    seeded shuffle for a deterministic, reproducible split.
    """
    qa = [q for c in conversations for q in c.qa if q.category != ADVERSARIAL_CATEGORY]
    if train_size + val_size > len(qa):
        raise ValueError(f"split sizes {train_size}+{val_size} exceed {len(qa)} available QA")
    rng = random.Random(seed)
    rng.shuffle(qa)
    return Splits(
        train=tuple(qa[:train_size]),
        val=tuple(qa[train_size : train_size + val_size]),
        test=tuple(qa[train_size + val_size :]),
    )
