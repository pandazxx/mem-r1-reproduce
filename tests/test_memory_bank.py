import pytest

from memory_r1.memory_bank import (
    MemoryBank,
    MemoryOperation,
    OperationError,
    parse_operations,
)


@pytest.fixture
def bank():
    b = MemoryBank()
    b.add("John has a dog named Rex", timestamp="8 May, 2023")
    b.add("John lives in Berlin", timestamp="8 May, 2023")
    return b


def test_add_assigns_unique_ids(bank):
    assert [e.id for e in bank.entries] == ["0", "1"]
    entry = bank.add("John likes coffee")
    assert entry.id == "2"
    assert len(bank) == 3


def test_update(bank):
    updated = bank.update("0", "John has two dogs", timestamp="9 May, 2023")
    assert bank.get("0").text == "John has two dogs"
    assert updated.timestamp == "9 May, 2023"


def test_update_keeps_old_timestamp_if_not_given(bank):
    bank.update("0", "John has two dogs")
    assert bank.get("0").timestamp == "8 May, 2023"


def test_delete(bank):
    bank.delete("0")
    assert bank.get("0") is None
    assert len(bank) == 1


def test_delete_missing_raises(bank):
    with pytest.raises(OperationError):
        bank.delete("99")


def test_update_missing_raises(bank):
    with pytest.raises(OperationError):
        bank.update("99", "text")


def test_deleted_id_is_never_reused(bank):
    bank.delete("1")
    entry = bank.add("new fact")
    assert entry.id == "2"


def test_apply_operations(bank):
    bank.apply(MemoryOperation(op="ADD", text="John plays tennis"))
    bank.apply(MemoryOperation(op="UPDATE", id="1", text="John moved to Munich"))
    bank.apply(MemoryOperation(op="DELETE", id="0"))
    bank.apply(MemoryOperation(op="NOOP"))
    texts = [e.text for e in bank.entries]
    assert texts == ["John moved to Munich", "John plays tennis"]


@pytest.mark.parametrize(
    "op",
    [
        MemoryOperation(op="ADD"),
        MemoryOperation(op="UPDATE", id="0"),
        MemoryOperation(op="UPDATE", text="x"),
        MemoryOperation(op="DELETE"),
    ],
)
def test_apply_invalid_operation_raises(bank, op):
    with pytest.raises(OperationError):
        bank.apply(op)


def test_copy_is_independent(bank):
    clone = bank.copy()
    clone.delete("0")
    clone.add("only in clone")
    assert len(bank) == 2
    assert bank.get("0") is not None


def test_save_load_roundtrip(bank, tmp_path):
    path = tmp_path / "bank.json"
    bank.save(path)
    loaded = MemoryBank.load(path)
    assert loaded.entries == bank.entries
    assert loaded.add("new").id == "2"


def test_parse_operations_list():
    ops = parse_operations(
        '[{"operation": "ADD", "text": "fact"}, {"operation": "DELETE", "id": 3}]'
    )
    assert ops == [
        MemoryOperation(op="ADD", text="fact"),
        MemoryOperation(op="DELETE", id="3"),
    ]


def test_parse_operations_single_object_and_fence():
    ops = parse_operations('```json\n{"operation": "UPDATE", "id": "1", "text": "new"}\n```')
    assert ops == [MemoryOperation(op="UPDATE", id="1", text="new")]


def test_parse_operations_wrapped_and_prose():
    ops = parse_operations(
        'Sure! Here is the operation: {"operations": [{"operation": "noop"}]} Done.'
    )
    assert ops == [MemoryOperation(op="NOOP")]


def test_parse_operations_mem0_aliases():
    ops = parse_operations('[{"event": "NONE"}]')
    assert ops == [MemoryOperation(op="NOOP")]


@pytest.mark.parametrize(
    "bad",
    [
        "no json here",
        '{"operation": "EXPLODE"}',
        '"just a string"',
        "[1, 2]",
    ],
)
def test_parse_operations_invalid_raises(bad):
    with pytest.raises(OperationError):
        parse_operations(bad)
