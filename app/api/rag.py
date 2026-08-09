import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User
from app.db.session import get_db
from app.schemas import RagResponse, SourceHit
from app.services.auth import auth_service
from app.services.chat import chat_service
from app.services.elasticsearch import es_service
from app.services.embedding import embedding_service
from app.services.llm import llm_service
from app.services.memory import memory_store

router = APIRouter(prefix="/rag", tags=["RAG"])

_STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _build_context(hits: list[dict], window_size: int) -> list[dict]:
    return [
        {
            "title": hit["title"],
            "chunk_id": hit["chunk_id"],
            "text": es_service.get_chunk_window(hit["chunk_id"], window_size),
        }
        for hit in hits
    ]


def _build_sources(hits: list[dict]) -> list[dict]:
    return [
        {
            "paper_id": hit["paper_id"],
            "title": hit["title"],
            "anchor_chunk_id": hit["chunk_id"],
            "page_number": hit.get("page_number", 1),
            "matching_score": hit["score"],
            "source_url": hit.get("source_url") or "",
            "local_pdf": bool(hit.get("local_pdf")),
        }
        for hit in hits
    ]


def _retrieve(search_query: str, limit: int) -> list[dict]:
    return es_service.hybrid_search(
        query=search_query,
        query_vector=embedding_service.get_embeddings(search_query),
        limit=limit,
    )


@router.get("/ask", response_model=RagResponse)
async def ask_rag(q: str = Query(...), limit: int = 2, window_size: int = 1):
    try:
        search_query = q
        hits = _retrieve(search_query, limit)
        if not hits:
            return RagResponse(question=q, answer="No matching notes found in the CS notes library. Try rephrasing your question.")

        answer = await llm_service.generate_response(q, _build_context(hits, window_size))
        return RagResponse(
            question=q,
            answer=answer,
            search_query=search_query,
            sources=[SourceHit(**source) for source in _build_sources(hits)],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG execution failed: {e}")


@router.get("/ask/stream")
async def ask_rag_stream(
    q: str = Query(...),
    chat_id: int = Query(...),
    limit: int = 2,
    window_size: int = 1,
    current_user: User = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        await chat_service.get_chat(chat_id, current_user.id, db)
        session_id = str(chat_id)
        history = await memory_store.get_history(session_id)
        search_query = q
        hits = _retrieve(search_query, limit)

        if not hits:
            async def no_results():
                yield "event: sources\ndata: []\n\n"
                yield f"event: token\ndata: {json.dumps('No matching notes found in the CS notes library. Try rephrasing your question.')}\n\n"
                yield "event: done\ndata: {}\n\n"

            return StreamingResponse(no_results(), media_type="text/event-stream")

        context = _build_context(hits, window_size)
        sources = _build_sources(hits)

        async def event_stream():
            yield f"event: search_query\ndata: {json.dumps(search_query)}\n\n"
            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

            tokens = []
            async for token in llm_service.stream_response(question=q, history=history, context_chunks=context):
                tokens.append(token)
                yield f"event: token\ndata: {json.dumps(token)}\n\n"

            full_response = "".join(tokens)
            await memory_store.add_message(session_id, "user", q)
            await memory_store.add_message(session_id, "assistant", full_response)
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_STREAM_HEADERS)
    except HTTPException:
        raise
    except Exception as e:
        async def error_stream():
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"

        return StreamingResponse(error_stream(), media_type="text/event-stream", headers=_STREAM_HEADERS)
