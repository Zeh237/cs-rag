from fastapi import APIRouter, HTTPException, Query

from app.schemas import SearchResponse
from app.services.elasticsearch import es_service
from app.services.embedding import embedding_service

router = APIRouter(prefix="/search", tags=["Search"])


@router.get("/keyword", response_model=SearchResponse)
async def keyword_search(q: str = Query(...), limit: int = 5):
    try:
        return {"query": q, "results": es_service.keyword_search(query=q, limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Keyword search failed: {e}")


@router.get("/semantic", response_model=SearchResponse)
async def semantic_search(q: str = Query(...), limit: int = 5):
    try:
        results = es_service.semantic_search_cosine(
            query_vector=embedding_service.get_query_embeddings(q), limit=limit
        )
        return {"query": q, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {e}")
