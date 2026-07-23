from elasticsearch import Elasticsearch
from config import settings
import re

class ElasticsearchService:
    def __init__(self):
        if settings.ES_USER and settings.ES_PASSWORD:
            self.client = Elasticsearch(
                settings.ES_HOST,
                basic_auth=(settings.ES_USER, settings.ES_PASSWORD),
                verify_certs=False,
                ssl_show_warn=False  
            )
        else:
            self.client = Elasticsearch(settings.ES_HOST)

    def verify_connection(self) -> bool:
        try:
            return self.client.ping()
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def create_index_if_not_exists(self):
        """Builds index schema configured for 384-dim bge-small-en-v1.5 embeddings."""
        index_mapping = {
            "mappings": {
                "properties": {
                    "paper_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "authors": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "text": {"type": "text"},
                    "vector_embedding": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        if not self.client.indices.exists(index=settings.INDEX_NAME):
            self.client.indices.create(index=settings.INDEX_NAME, body=index_mapping)
            print(f"Created Elasticsearch index: '{settings.INDEX_NAME}'")

    def index_chunk(self, chunk_id: str, document: dict):
        self.client.index(index=settings.INDEX_NAME, id=chunk_id, document=document)

    def refresh_index(self):
        self.client.indices.refresh(index=settings.INDEX_NAME)

    def keyword_search(self, query: str, limit: int = 5) -> list[dict]:
        """Standard BM25 keyword search on text and title fields."""
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text", "title^2"]  # Slightly boost matches in titles
                }
            }
        }
        response = self.client.search(index=settings.INDEX_NAME, body=body, size=limit)
        return self._format_results(response)

    def semantic_search_cosine(self, query_vector: list[float], limit: int = 5) -> list[dict]:
        """Exact semantic search using Cosine Similarity script score instead of KNN."""
        body = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector_embedding') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }
        response = self.client.search(index=settings.INDEX_NAME, body=body, size=limit)
        return self._format_results(response)

    def _format_results(self, response: dict) -> list[dict]:
        """Helper to parse raw Elasticsearch hits into clean, readable dicts."""
        hits = response["hits"]["hits"]
        results = []
        for hit in hits:
            source = hit["_source"]
            results.append({
                "score": hit["_score"],
                "paper_id": source.get("paper_id"),
                "title": source.get("title"),
                "authors": source.get("authors"),
                "category": source.get("category"),
                "chunk_id": source.get("chunk_id"),
                "page_number": source.get("page_number", 1),
                "text": source.get("text")
            })
        return results
    
    def get_chunk_window(self, chunk_id: str, window_size: int = 1) -> str:
        """
        Retrieves the target chunk along with its surrounding sibling chunks (prev and next)
        to reconstruct a larger context window without running additional vector searches.
        """
        base_prefix, current_idx = self._parse_chunk_id(chunk_id)
        
        # Fallback if the chunk naming pattern doesn't match our sequential assumption
        if base_prefix is None or current_idx is None:
            try:
                response = self.client.get(index=settings.INDEX_NAME, id=chunk_id)
                return response["_source"]["text"]
            except Exception:
                return ""

        # Generate target sibling chunk IDs
        # e.g., if window_size is 1 and current_idx is 3, we request: 2, 3, and 4
        start_idx = max(0, current_idx - window_size)
        end_idx = current_idx + window_size
        target_ids = [f"{base_prefix}{i}" for i in range(start_idx, end_idx + 1)]

        try:
            # Perform a highly optimized O(1) multi-get lookup in ES
            response = self.client.mget(index=settings.INDEX_NAME, body={"ids": target_ids})
            docs = response.get("docs", [])

            # Filter out any documents that don't exist (e.g., if index went below 0 or past document limits)
            valid_chunks = [doc["_source"] for doc in docs if doc.get("found")]

            # Sort chunks chronologically by their parsed chunk index to preserve reading flow
            valid_chunks.sort(key=lambda x: self._parse_chunk_id(x["chunk_id"])[1])

            # Merge the contiguous text blocks
            combined_text = "\n\n".join([chunk["text"] for chunk in valid_chunks])
            return combined_text

        except Exception as e:
            print(f"Error expanding chunk window for {chunk_id}: {e}")
            return ""

    def _parse_chunk_id(self, chunk_id: str) -> tuple[str | None, int | None]:
        """Parses sequential chunk IDs. Example: 'paper_chunk_12' -> ('paper_chunk_', 12)"""
        match = re.match(r"^(.*_chunk_)(\d+)$", chunk_id)
        if match:
            prefix, idx = match.groups()
            return prefix, int(idx)
        return None, None

    

# Singleton instance
es_service = ElasticsearchService()