import re

from elasticsearch import Elasticsearch

from app.core.config import settings
from app.services.reranker import reranker_service

_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "paper_id": {"type": "keyword"},
            "title": {"type": "text"},
            "authors": {"type": "keyword"},
            "category": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "page_number": {"type": "integer"},
            "text": {"type": "text"},
            "highlight_boxes": {"type": "object", "enabled": False},
            "source_url": {"type": "keyword"},
            "local_pdf": {"type": "boolean"},
            "vector_embedding": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"},
        }
    }
}


class ElasticsearchService:
    def __init__(self) -> None:
        if settings.ES_USER and settings.ES_PASSWORD:
            self.client = Elasticsearch(
                settings.ES_HOST,
                basic_auth=(settings.ES_USER, settings.ES_PASSWORD),
                verify_certs=False,
                ssl_show_warn=False,
            )
        else:
            self.client = Elasticsearch(settings.ES_HOST)

    def verify_connection(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False

    def create_index_if_not_exists(self) -> None:
        if not self.client.indices.exists(index=settings.INDEX_NAME):
            self.client.indices.create(index=settings.INDEX_NAME, body=_INDEX_MAPPING)

    def index_chunk(self, chunk_id: str, document: dict) -> None:
        self.client.index(index=settings.INDEX_NAME, id=chunk_id, document=document)

    def refresh_index(self) -> None:
        self.client.indices.refresh(index=settings.INDEX_NAME)

    def delete_index(self) -> None:
        if self.client.indices.exists(index=settings.INDEX_NAME):
            self.client.indices.delete(index=settings.INDEX_NAME)

    def keyword_search(self, query: str, limit: int = 5) -> list[dict]:
        body = {"query": {"multi_match": {"query": query, "fields": ["text", "title^2"]}}}
        return self._format_results(self.client.search(index=settings.INDEX_NAME, body=body, size=limit))

    def semantic_search_cosine(self, query_vector: list[float], limit: int = 5) -> list[dict]:
        body = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector_embedding') + 1.0",
                        "params": {"query_vector": query_vector},
                    },
                }
            }
        }
        return self._format_results(self.client.search(index=settings.INDEX_NAME, body=body, size=limit))

    def hybrid_search(self, query: str, query_vector: list[float], limit: int = 5, rerank: bool = True) -> list[dict]:
        candidates = max(limit * 4, 10)
        semantic_hits = self.semantic_search_cosine(query_vector, limit=candidates)
        keyword_hits = self.keyword_search(query, limit=candidates)

        fused = {}
        k = 60
        for hits in (semantic_hits, keyword_hits):
            for rank, hit in enumerate(hits):
                entry = fused.setdefault(hit["chunk_id"], dict(hit))
                entry["_rrf_score"] = entry.get("_rrf_score", 0.0) + 1.0 / (k + rank + 1)

        ranked = sorted(fused.values(), key=lambda h: h["_rrf_score"], reverse=True)
        for hit in ranked:
            hit.pop("_rrf_score", None)

        if rerank:
            try:
                ranked = reranker_service.rerank(query, ranked)
            except Exception:
                pass

        return ranked[:limit]

    def get_chunk_window(self, chunk_id: str, window_size: int = 1) -> str:
        prefix, current_idx = self._parse_chunk_id(chunk_id)
        if prefix is None or current_idx is None:
            try:
                return self.client.get(index=settings.INDEX_NAME, id=chunk_id)["_source"]["text"]
            except Exception:
                return ""

        start_idx = max(0, current_idx - window_size)
        end_idx = current_idx + window_size
        target_ids = [f"{prefix}{i}" for i in range(start_idx, end_idx + 1)]

        try:
            response = self.client.mget(index=settings.INDEX_NAME, body={"ids": target_ids})
            chunks = [doc["_source"] for doc in response.get("docs", []) if doc.get("found")]
            chunks.sort(key=lambda c: self._parse_chunk_id(c["chunk_id"])[1])
            return "\n\n".join(chunk["text"] for chunk in chunks)
        except Exception:
            return ""

    def _format_results(self, response: dict) -> list[dict]:
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(
                {
                    "score": hit["_score"],
                    "paper_id": source.get("paper_id"),
                    "title": source.get("title"),
                    "authors": source.get("authors"),
                    "category": source.get("category"),
                    "chunk_id": source.get("chunk_id"),
                    "page_number": source.get("page_number", 1),
                    "text": source.get("text"),
                    "source_url": source.get("source_url") or "",
                    "local_pdf": bool(source.get("local_pdf")),
                }
            )
        return results

    def _parse_chunk_id(self, chunk_id: str) -> tuple[str | None, int | None]:
        match = re.match(r"^(.*_chunk_)(\d+)$", chunk_id)
        if match:
            prefix, idx = match.groups()
            return prefix, int(idx)
        return None, None


es_service = ElasticsearchService()
