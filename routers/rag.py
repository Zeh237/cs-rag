import json
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from services.elasticsearch_service import es_service
from services.embedding import embedding_service
from services.llm_service import llm_service

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.get("/ask")
async def ask_rag(
    q: str = Query(..., description="Ask a question about your indexed papers"),
    limit: int = 2,          # Match top 2 most semantically relevant anchor points
    window_size: int = 1     # Pull 1 chunk before and 1 chunk after each matched anchor
):
    """
    RAG pipeline using Context-Window (Parent-Child) expansion.
    1. Matches high-relevance 'child' chunks using exact Cosine vector similarity.
    2. Expands hits to their surrounding 'parent' context windows.
    3. Synthesizes the final answer using your local LLM.
    """
    try:
        # 1. Retrieve the high-relevance semantic hit coordinates
        query_vector = embedding_service.get_embeddings(q)
        semantic_hits = es_service.semantic_search_cosine(query_vector=query_vector, limit=limit)

        if not semantic_hits:
            return {
                "answer": "No relevant documentation found to answer this question.",
                "sources": []
            }

        # 2. Expand context around each semantic hit
        expanded_context_chunks = []
        for hit in semantic_hits:
            expanded_text = es_service.get_chunk_window(
                chunk_id=hit["chunk_id"],
                window_size=window_size
            )
            expanded_context_chunks.append({
                "title": hit["title"],
                "chunk_id": hit["chunk_id"],
                "text": expanded_text
            })

        # 3. Call LLM and get the full answer
        print(expanded_context_chunks)
        answer = await llm_service.generate_response(question=q, context_chunks=expanded_context_chunks)

        # 4. Return answer along with the structural source anchor points
        sources = [
            {
                "paper_id": hit["paper_id"],
                "title": hit["title"],
                "anchor_chunk_id": hit["chunk_id"],
                "page_number": hit.get("page_number", 1),
                "matching_score": hit["score"]
            }
            for hit in semantic_hits
        ]

        return {
            "question": q,
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG execution failed: {str(e)}")


@router.get("/ask/stream")
async def ask_rag_stream(
    q: str = Query(..., description="Ask a question about your indexed papers"),
    limit: int = 2,
    window_size: int = 1
):
    """
    Streaming RAG endpoint. Returns a Server-Sent Events (SSE) stream.

    Event types sent to the client:
      - event: sources  — JSON array of source documents (sent first, before tokens)
      - event: token    — a single text token delta from the LLM
      - event: done     — signals the stream is complete
      - event: error    — an error message string
    """
    try:
        # 1. Retrieve semantic hits
        query_vector = embedding_service.get_embeddings(q)
        semantic_hits = es_service.semantic_search_cosine(query_vector=query_vector, limit=limit)

        if not semantic_hits:
            async def no_results():
                yield "event: sources\ndata: []\n\n"
                yield f"event: token\ndata: {json.dumps('No relevant documentation found to answer this question.')}\n\n"
                yield "event: done\ndata: {}\n\n"
            return StreamingResponse(no_results(), media_type="text/event-stream")

        # 2. Expand context windows
        expanded_context_chunks = []
        for hit in semantic_hits:
            expanded_text = es_service.get_chunk_window(
                chunk_id=hit["chunk_id"],
                window_size=window_size
            )
            expanded_context_chunks.append({
                "title": hit["title"],
                "chunk_id": hit["chunk_id"],
                "text": expanded_text
            })

        sources = [
            {
                "paper_id": hit["paper_id"],
                "title": hit["title"],
                "anchor_chunk_id": hit["chunk_id"],
                "page_number": hit.get("page_number", 1),
                "matching_score": hit["score"]
            }
            for hit in semantic_hits
        ]

        async def event_stream():
            # Send sources metadata first so the frontend can display them immediately
            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

            # Stream LLM tokens
            async for token in llm_service.stream_response(question=q, context_chunks=expanded_context_chunks):
                yield f"event: token\ndata: {json.dumps(token)}\n\n"

            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",    # Disable nginx buffering if behind a proxy
            }
        )

    except Exception as e:
        async def error_stream():
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")