import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

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
            except Exception as local_err:
                if "HF_HUB_OFFLINE" in os.environ:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                try:
                    self._model = SentenceTransformer(
                        settings.EMBEDDING_MODEL,
                        trust_remote_code=False,
                    )
                except Exception as remote_err:
                    raise OSError(
                        f"Could not load '{settings.EMBEDDING_MODEL}' from cache and "
                        f"could not reach huggingface.co ({remote_err}). "
                        f"Check network or pre-download with: "
                        f"python -c \"from sentence_transformers import SentenceTransformer; "
                        f"SentenceTransformer('{settings.EMBEDDING_MODEL}')\""
                    ) from remote_err
        return self._model

    def get_embeddings(self, text: str):
        return self.model.encode(text).tolist()


embedding_service = EmbeddingService()
