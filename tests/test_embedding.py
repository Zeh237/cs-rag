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