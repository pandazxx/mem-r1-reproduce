from memory_r1.memory_bank import MemoryBank
from memory_r1.retrieval import Retriever, cosine_similarity


class FakeEmbedder:
    """Embeds by keyword overlap so similarity is predictable in tests."""

    VOCAB = ["dog", "city", "food", "sport"]

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0 if word in text.lower() else 0.0 for word in self.VOCAB] for text in texts]


def make_bank():
    bank = MemoryBank()
    bank.add("John has a dog named Rex")
    bank.add("John lives in a big city")
    bank.add("John loves spicy food")
    return bank


def test_cosine_similarity():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_retrieve_ranks_by_similarity():
    retriever = Retriever(FakeEmbedder())
    results = retriever.retrieve(make_bank(), "what dog does John have?", k=2)
    assert [e.text for e in results] == ["John has a dog named Rex", "John lives in a big city"]


def test_retrieve_k_larger_than_bank():
    retriever = Retriever(FakeEmbedder())
    assert len(retriever.retrieve(make_bank(), "dog", k=60)) == 3


def test_retrieve_empty_bank():
    embedder = FakeEmbedder()
    retriever = Retriever(embedder)
    assert retriever.retrieve(MemoryBank(), "dog") == []
    assert embedder.calls == []


def test_entry_embeddings_are_cached():
    embedder = FakeEmbedder()
    retriever = Retriever(embedder)
    bank = make_bank()
    retriever.retrieve(bank, "dog")
    retriever.retrieve(bank, "food")
    # 2 query calls + 1 batch for the three entries
    assert len(embedder.calls) == 3
    entry_batches = [c for c in embedder.calls if len(c) == 3]
    assert len(entry_batches) == 1
