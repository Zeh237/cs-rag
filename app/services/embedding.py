from app.core.config import settings


class EmbeddingService:
    def __init__(self) -> None:
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self._model

    def get_embeddings(self, text: str):
        return self.model.encode(text).tolist()


embedding_service = EmbeddingService()
