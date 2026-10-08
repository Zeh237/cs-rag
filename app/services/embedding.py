import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from app.core.config import settings

_QUERY_PROMPT = "SearchQuery"
_DOCUMENT_PROMPT = "Document"


def _is_embeddinggemma(model_id: str) -> bool:
    return "embeddinggemma" in (model_id or "").lower()


class EmbeddingService:
    """Text embeddings with task prompts for instruction-trained models.

    Follows the EmbeddingGemma 2 developer guide: the text-only encoder
    setup (270M parameters) embeds queries with the ``SearchQuery`` prompt
    and documents with the ``Document`` prompt into a shared 768-dimensional
    space. Models without prompt support fall back to plain encoding.
    """

    def __init__(self) -> None:
        self._model = None

    def _load_kwargs(self) -> dict:
        if _is_embeddinggemma(settings.EMBEDDING_MODEL):
            return {"config_kwargs": {"vision_config": None, "audio_config": None}}
        return {}

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            load_kwargs = self._load_kwargs()
            try:
                self._model = SentenceTransformer(
                    settings.EMBEDDING_MODEL,
                    local_files_only=True,
                    trust_remote_code=False,
                    **load_kwargs,
                )
            except Exception:
                if "HF_HUB_OFFLINE" in os.environ:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                try:
                    self._model = SentenceTransformer(
                        settings.EMBEDDING_MODEL,
                        trust_remote_code=False,
                        **load_kwargs,
                    )
                except Exception as remote_err:
                    raise OSError(
                        f"Could not load '{settings.EMBEDDING_MODEL}' from cache and "
                        f"could not reach huggingface.co ({remote_err}). "
                        f"Check network or pre-download with: "
                        f"python -c \"from sentence_transformers import SentenceTransformer; "
                        f"SentenceTransformer('{settings.EMBEDDING_MODEL}', "
                        f"config_kwargs={{'vision_config': None, 'audio_config': None}})\""
                    ) from remote_err
        return self._model

    def _encode(self, text: str, prompt_name: str | None):
        model = self.model
        normalize = _is_embeddinggemma(settings.EMBEDDING_MODEL)
        prompts = getattr(model, "prompts", None) or {}
        if prompt_name and prompt_name in prompts:
            return model.encode(text, prompt_name=prompt_name, normalize_embeddings=normalize).tolist()
        try:
            return model.encode(
                text,
                **({"prompt_name": prompt_name} if prompt_name else {}),
                normalize_embeddings=normalize,
            ).tolist()
        except TypeError:
            return model.encode(text).tolist()

    def get_embeddings(self, text: str, task: str | None = None):
        prompt = {"query": _QUERY_PROMPT, "document": _DOCUMENT_PROMPT}.get(task or "")
        return self._encode(text, prompt)

    def get_query_embeddings(self, text: str):
        return self.get_embeddings(text, task="query")

    def get_document_embeddings(self, text: str):
        return self.get_embeddings(text, task="document")


embedding_service = EmbeddingService()
