"""External memory bank with Mem0-style operations (ADD / UPDATE / DELETE / NOOP).

Pure Python and model-free so the op semantics and LLM-output parsing are
unit-testable; the RL environment and bootstrap pipeline build on this.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

OpName = Literal["ADD", "UPDATE", "DELETE", "NOOP"]

# Mem0's update prompt calls the no-op "NONE"; the paper calls it NOOP.
_OP_ALIASES = {"NONE": "NOOP"}
VALID_OPS = ("ADD", "UPDATE", "DELETE", "NOOP")


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    text: str
    timestamp: str | None = None


@dataclass(frozen=True)
class MemoryOperation:
    op: OpName
    id: str | None = None
    text: str | None = None
    timestamp: str | None = None


class OperationError(ValueError):
    """A memory operation could not be applied (bad id, missing field, unknown op)."""


@dataclass
class MemoryBank:
    _entries: dict[str, MemoryEntry] = field(default_factory=dict)
    _next_id: int = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries.values())

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._entries.get(entry_id)

    def add(self, text: str, timestamp: str | None = None) -> MemoryEntry:
        entry = MemoryEntry(id=str(self._next_id), text=text, timestamp=timestamp)
        self._next_id += 1
        self._entries[entry.id] = entry
        return entry

    def update(self, entry_id: str, text: str, timestamp: str | None = None) -> MemoryEntry:
        old = self._entries.get(entry_id)
        if old is None:
            raise OperationError(f"UPDATE: no entry with id {entry_id!r}")
        entry = MemoryEntry(id=entry_id, text=text, timestamp=timestamp or old.timestamp)
        self._entries[entry_id] = entry
        return entry

    def delete(self, entry_id: str) -> None:
        if entry_id not in self._entries:
            raise OperationError(f"DELETE: no entry with id {entry_id!r}")
        del self._entries[entry_id]

    def apply(self, operation: MemoryOperation) -> None:
        if operation.op == "ADD":
            if not operation.text:
                raise OperationError("ADD requires text")
            self.add(operation.text, operation.timestamp)
        elif operation.op == "UPDATE":
            if operation.id is None or not operation.text:
                raise OperationError("UPDATE requires id and text")
            self.update(operation.id, operation.text, operation.timestamp)
        elif operation.op == "DELETE":
            if operation.id is None:
                raise OperationError("DELETE requires id")
            self.delete(operation.id)
        elif operation.op == "NOOP":
            pass
        else:
            raise OperationError(f"unknown operation {operation.op!r}")

    def copy(self) -> MemoryBank:
        return MemoryBank(_entries=dict(self._entries), _next_id=self._next_id)

    def to_dict(self) -> dict:
        return {
            "next_id": self._next_id,
            "entries": [
                {"id": e.id, "text": e.text, "timestamp": e.timestamp} for e in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryBank:
        bank = cls(_next_id=data["next_id"])
        for raw in data["entries"]:
            entry = MemoryEntry(id=raw["id"], text=raw["text"], timestamp=raw.get("timestamp"))
            bank._entries[entry.id] = entry
        return bank

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> MemoryBank:
        return cls.from_dict(json.loads(Path(path).read_text()))


def parse_operations(llm_output: str) -> list[MemoryOperation]:
    """Parse memory operations from LLM output.

    Accepts a JSON array of operations, a single operation object, or an
    object with an ``"operations"`` key — optionally inside a ```json fence
    or surrounding prose. Raises OperationError if no valid JSON is found.
    """
    payload = _extract_json(llm_output)
    if isinstance(payload, dict) and "operations" in payload:
        payload = payload["operations"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise OperationError(f"expected a list of operations, got {type(payload).__name__}")
    return [_parse_one(raw) for raw in payload]


def _parse_one(raw: object) -> MemoryOperation:
    if not isinstance(raw, dict):
        raise OperationError(f"operation must be an object, got {raw!r}")
    op = str(raw.get("operation", raw.get("event", ""))).upper()
    op = _OP_ALIASES.get(op, op)
    if op not in VALID_OPS:
        raise OperationError(f"unknown operation {op!r} in {raw!r}")
    entry_id = raw.get("id")
    return MemoryOperation(
        op=op,  # type: ignore[arg-type]
        id=str(entry_id) if entry_id is not None else None,
        text=raw.get("text"),
        timestamp=raw.get("timestamp"),
    )


def _extract_json(text: str) -> object:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min((i for i in (text.find("["), text.find("{")) if i != -1), default=-1)
    if start == -1:
        raise OperationError(f"no JSON found in output: {text[:200]!r}")
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise OperationError(f"invalid JSON in output: {text[:200]!r}") from exc
    return payload
