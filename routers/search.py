from fastapi import APIRouter, HTTPException, Query
from services.elasticsearch_service import es_service
from services.embedding import embedding_service

router = APIRouter(prefix="/search", tags=["Search"])

@router.get("/keyword")
async def keyword_search(
    q: str = Query(..., description="The text query to search for"), 
    limit: int = 5
):
    """
    Standard lexical search (BM25) over indexed document chunks.
    """
    try:
        results = es_service.keyword_search(query=q, limit=limit)
        return {"query": q, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Keyword search failed: {str(e)}")


@router.get("/semantic")
async def semantic_search(
    q: str = Query(..., description="The text query to embed and search for"), 
    limit: int = 5
):
    try:
        # Change this line to use get_embeddings
        query_vector = embedding_service.get_embeddings(q)
        
        results = es_service.semantic_search_cosine(query_vector=query_vector, limit=limit)
        return {"query": q, "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")