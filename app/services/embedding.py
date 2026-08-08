class EmbeddingService:
    def __init__(self) -> None:
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return self._model

    def get_embeddings(self, text: str):
        return self.model.encode(text).tolist()


embedding_service = EmbeddingService()
