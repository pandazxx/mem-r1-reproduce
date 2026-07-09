from memory_r1.memory_bank import MemoryBank
from memory_r1.retrieval import OpenAIEmbedder, Retriever, cosine_similarity


class FakeEmbedder:
    """Embeds by keyword overlap so similarity is predictable in tests."""

    VOCAB = ["dog", "city", "food", "sport"]

    def __init__(self):
        self.calls = []

    def embed(self, texts, *, kind="passage"):
        self.calls.append((list(texts), kind))
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
    entry_batches = [c for c in embedder.calls if len(c[0]) == 3]
    assert len(entry_batches) == 1
    assert entry_batches[0][1] == "passage"
    query_kinds = [kind for texts, kind in embedder.calls if len(texts) == 1]
    assert query_kinds == ["query", "query"]


class FakeEmbeddingsClient:
    def __init__(self):
        self.kwargs = None
        self.batches = []
        self.embeddings = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.batches.append(list(kwargs["input"]))

        class Item:
            embedding = [1.0, 0.0]

        class Response:
            data = [Item() for _ in kwargs["input"]]

        return Response()


def test_openai_embedder_input_type_param():
    client = FakeEmbeddingsClient()
    OpenAIEmbedder(model="m", client=client, input_type_param=True).embed(["x"], kind="query")
    assert client.kwargs["extra_body"] == {"input_type": "query", "truncate": "END"}

    client = FakeEmbeddingsClient()
    OpenAIEmbedder(model="m", client=client).embed(["x"], kind="query")
    assert "extra_body" not in client.kwargs


def test_openai_embedder_batches_and_throttles():
    client = FakeEmbeddingsClient()
    throttles = []
    embedder = OpenAIEmbedder(
        model="m", client=client, batch_size=2, throttle=lambda: throttles.append(1)
    )
    vectors = embedder.embed(["a", "b", "c", "d", "e"])
    assert client.batches == [["a", "b"], ["c", "d"], ["e"]]
    assert len(vectors) == 5
    assert len(throttles) == 3


def test_openai_embedder_uses_retry_wrapper():
    client = FakeEmbeddingsClient()
    wrapped = []

    def retry(call):
        wrapped.append(1)
        return call()

    OpenAIEmbedder(model="m", client=client, batch_size=2, retry=retry).embed(["a", "b", "c"])
    assert len(wrapped) == 2


def test_openai_embedder_empty_input_makes_no_calls():
    client = FakeEmbeddingsClient()
    assert OpenAIEmbedder(model="m", client=client).embed([]) == []
    assert client.batches == []
