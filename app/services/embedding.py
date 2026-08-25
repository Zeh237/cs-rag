import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from app.core.config import settings


class EmbeddingService:
    def __init__(self) -> None:
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(
                    settings.EMBEDDING_MODEL,
                    local_files_only=True,
                    trust_remote_code=False,
                )
            except Exception:
                self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self._model

    def get_embeddings(self, text: str):
        return self.model.encode(text).tolist()


embedding_service = EmbeddingService()
