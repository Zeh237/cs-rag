import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class RerankerService:
    def __init__(self) -> None:
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            try:
                self._model = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    local_files_only=True,
                    trust_remote_code=False,
                )
            except Exception:
                self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._model

    def rerank(self, query: str, hits: list[dict], top_k: int | None = None) -> list[dict]:
        if not hits:
            return hits

        texts = [h.get("text") or "" for h in hits]
        scores = self.model.predict([(query, text) for text in texts], show_progress_bar=False)

        ranked = []
        for hit, score in sorted(zip(hits, scores), key=lambda x: x[1], reverse=True):
            entry = dict(hit)
            entry["score"] = float(score)
            ranked.append(entry)

        return ranked[:top_k] if top_k is not None else ranked


reranker_service = RerankerService()
