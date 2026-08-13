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
            "page_start": {"type": "integer"},
            "page_end": {"type": "integer"},
            "text": {"type": "text"},
            "highlight_boxes": {"type": "object", "enabled": False},
            "source_url": {"type": "keyword"},
            "local_pdf": {"type": "boolean"},
            "vector_embedding": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
            "span_start": {"type": "integer"},
            "span_end": {"type": "integer"},
            "strategy": {"type": "keyword"},
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

    def create_index_if_not_exists(self, index: str | None = None) -> None:
        index = index or settings.INDEX_NAME
        if not self.client.indices.exists(index=index):
            self.client.indices.create(index=index, body=_INDEX_MAPPING)

    def index_chunk(self, chunk_id: str, document: dict, index: str | None = None) -> None:
        self.client.index(index=index or settings.INDEX_NAME, id=chunk_id, document=document)

    def refresh_index(self, index: str | None = None) -> None:
        self.client.indices.refresh(index=index or settings.INDEX_NAME)

    def delete_index(self, index: str | None = None) -> None:
        index = index or settings.INDEX_NAME
        if self.client.indices.exists(index=index):
            self.client.indices.delete(index=index)

    def keyword_search(self, query: str, limit: int = 5, index: str | None = None) -> list[dict]:
        body = {"query": {"multi_match": {"query": query, "fields": ["text", "title^3", "category^1.5"]}}}
        response = self.client.search(index=index or settings.INDEX_NAME, body=body, size=limit)
        return self._format_results(response, index=index)

    def semantic_search_cosine(self, query_vector: list[float], limit: int = 5, index: str | None = None) -> list[dict]:
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
        response = self.client.search(index=index or settings.INDEX_NAME, body=body, size=limit)
        return self._format_results(response, index)

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        limit: int = 5,
        rerank: bool = True,
        index: str | None = None,
    ) -> list[dict]:
        target = index or settings.INDEX_NAME
        candidates = max(limit * 6, 40)
        semantic_hits = self.semantic_search_cosine(query_vector, limit=candidates, index=target)
        keyword_hits = self.keyword_search(query, limit=candidates, index=target)

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
        entries = self.get_chunk_window_entries(chunk_id, window_size)
        return "\n\n".join(entry.get("text", "") for entry in entries)

    def search_pages(self, paper_id: str, query: str, limit: int = 12) -> list[dict]:
        """Page-level keyword search inside a single document.

        Returns one entry per page (sorted by page) with a short text snippet and
        the chunk id, for the vision research tool."""
        body = {
            "query": {
                "bool": {
                    "filter": [{"term": {"paper_id": paper_id}}],
                    "must": [{"multi_match": {"query": query, "fields": ["text", "title^2"]}}],
                }
            }
        }
        response = self.client.search(index=settings.INDEX_NAME, body=body, size=limit)
        by_page: dict[int, dict] = {}
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            page = source.get("page_start") or source.get("page_number") or 1
            if page in by_page:
                continue
            snippet = " ".join((source.get("text") or "").split())[:240]
            by_page[page] = {"page": page, "snippet": snippet, "chunk_id": source.get("chunk_id")}
        return [by_page[p] for p in sorted(by_page)]

    def get_chunk_window_entries(self, chunk_id: str, window_size: int = 1) -> list[dict]:
        """Return the anchor chunk plus up to `window_size` neighbors each side,
        as full source documents (text + page fields), ordered by chunk index."""
        prefix, current_idx = self._parse_chunk_id(chunk_id)
        if prefix is None or current_idx is None:
            try:
                doc = self.client.get(index=settings.INDEX_NAME, id=chunk_id)["_source"]
                return [doc] if doc else []
            except Exception:
                return []

        start_idx = max(0, current_idx - window_size)
        end_idx = current_idx + window_size
        target_ids = [f"{prefix}{i}" for i in range(start_idx, end_idx + 1)]

        try:
            response = self.client.mget(index=settings.INDEX_NAME, body={"ids": target_ids})
            chunks = [doc["_source"] for doc in response.get("docs", []) if doc.get("found")]
            chunks.sort(key=lambda c: self._parse_chunk_id(c.get("chunk_id", ""))[1] or 0)
            return chunks
        except Exception:
            return []

    def _format_results(self, response: dict, index: str | None = None) -> list[dict]:
        from app.services.pdf import pdf_service

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
                    "page_start": source.get("page_start") or source.get("page_number") or 1,
                    "page_end": source.get("page_end") or source.get("page_start") or source.get("page_number") or 1,
                    "text": source.get("text"),
                    "source_url": source.get("source_url") or "",
                    "local_pdf": bool(source.get("local_pdf"))
                    or bool(source.get("paper_id") and pdf_service.has_local_pdf(source["paper_id"])),
                    "span_start": source.get("span_start"),
                    "span_end": source.get("span_end"),
                    "strategy": source.get("strategy", ""),
                    "_index": hit.get("_index") or index,
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
