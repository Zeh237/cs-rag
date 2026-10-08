from app.services.embedding import EmbeddingService


class _FakeVec(list):
    def tolist(self):
        return list(self)


class _FakeModel:
    def encode(self, text: str):
        return _FakeVec([0.5, 0.25, 0.125])


def test_get_embeddings_returns_list():
    service = EmbeddingService()
    service._model = _FakeModel()
    emb = service.get_embeddings("some course note text")
    assert emb == [0.5, 0.25, 0.125]


def test_get_embeddings_shape():
    service = EmbeddingService()
    service._model = _FakeModel()
    emb = service.get_embeddings("vector")
    assert isinstance(emb, list)
    assert all(isinstance(x, float) for x in emb)


class _FakePromptModel:
    def __init__(self):
        self.prompts = {"SearchQuery": "query: ", "Document": "doc: "}
        self.calls = []

    def encode(self, text, prompt_name=None, normalize_embeddings=False):
        self.calls.append((text, prompt_name, normalize_embeddings))
        return _FakeVec([0.5, 0.25, 0.125])


def test_query_uses_search_query_prompt():
    service = EmbeddingService()
    service._model = _FakePromptModel()
    emb = service.get_query_embeddings("what is recursion?")
    assert emb == [0.5, 0.25, 0.125]
    assert service._model.calls[0][1] == "SearchQuery"


def test_document_uses_document_prompt():
    service = EmbeddingService()
    service._model = _FakePromptModel()
    emb = service.get_document_embeddings("recursion solves smaller instances")
    assert emb == [0.5, 0.25, 0.125]
    assert service._model.calls[0][1] == "Document"


def test_plain_model_falls_back_without_prompt():
    service = EmbeddingService()
    service._model = _FakeModel()
    assert service.get_query_embeddings("q") == [0.5, 0.25, 0.125]
    assert service.get_document_embeddings("d") == [0.5, 0.25, 0.125]