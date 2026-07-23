import os

# os.environ["HF_HUB_OFFLINE"] = "1" # Commented out to allow downloading the new model

class EmbeddingService:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            # pyrefly: ignore [missing-import]
            from sentence_transformers import SentenceTransformer
            print("Loading embedding model (this will download on the first API call)...")
            self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return self._model

    def get_embeddings(self, text: str):
        return self.model.encode(text).tolist()

embedding_service = EmbeddingService()