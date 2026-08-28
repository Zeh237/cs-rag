import json
import os
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import PROJECT_ROOT, settings
from app.db.models import User
from app.db.session import get_db
from app.schemas import RagResponse, SourceHit
from app.services.auth import auth_service
from app.services.chat import chat_service
from app.services.elasticsearch import es_service
from app.services.embedding import embedding_service
from app.services.llm import llm_service
from app.services.memory import memory_store
from app.services.pdf import pdf_service

router = APIRouter(prefix="/rag", tags=["RAG"])

_STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

_MAX_CONTEXT_WORDS = 4500

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
_PAGE_MARKER_RE = re.compile(r"^\[\[PAGE (\d+)\]\]$")
_PAGE_REF_RE = re.compile(r"\((p{1,2}\.)\s*(\d+(?:\s*[-–]\s*\d+)?)\)")


def _research_enabled() -> bool:
    return bool(settings.VISION_LLM_MODEL and settings.LLM_API_BASE)


def _build_toc(paper_id: str, max_entries: int = 150) -> str:
    """Headings -> page map built from the note's markdown, for the get_toc tool."""
    entry = pdf_service._find_notes_entry(paper_id)
    if not entry:
        return ""
    path = entry.get("local_path") or ""
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(path):
        return ""
    current_page = 1
    lines = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                page_match = _PAGE_MARKER_RE.match(stripped)
                if page_match:
                    current_page = int(page_match.group(1))
                    continue
                heading_match = _HEADING_RE.match(stripped)
                if heading_match:
                    lines.append(f"{'  ' * (len(heading_match.group(1)) - 1)}{heading_match.group(2).strip()} ... p. {current_page}")
                    if len(lines) >= max_entries:
                        break
    except Exception:
        return ""
    return "\n".join(lines)


def _research_callbacks(paper_id: str, evidence_pages: set[int] | None = None) -> dict:
    async def search_doc(query: str) -> str:
        results = es_service.search_pages(paper_id, query, limit=12)
        if evidence_pages is not None:
            evidence_pages.update(result["page"] for result in results if result.get("page") is not None)
        return json.dumps(results, ensure_ascii=False)

    async def toc() -> str:
        return _build_toc(paper_id)

    async def read_pages(pages: list[int]) -> list[dict]:
        return pdf_service.render_page_images(paper_id, pages, dpi=settings.RESEARCH_DPI)

    return {"search_document": search_doc, "get_toc": toc, "read_pages": read_pages}


def _research_sources(paper_id: str, title: str, pages: set[int]) -> list[dict]:
    return [
        {
            "paper_id": paper_id,
            "title": title,
            "page_number": page,
            "page_start": page,
            "page_end": page,
            "local_pdf": bool(pdf_service.has_local_pdf(paper_id)),
        }
        for page in sorted(pages)
    ]


def _research_seed(hit: dict) -> list[dict] | None:
    """Render the top hit's page range (plus neighbors) as images for the vision model."""
    if not pdf_service.has_local_pdf(hit["paper_id"]):
        return None
    page_start, page_end = _page_fields(hit)
    window = settings.RESEARCH_SEED_WINDOW
    seed_pages = list(range(max(1, page_start - window), page_end + window + 1))
    seed_images = pdf_service.render_page_images(hit["paper_id"], seed_pages, dpi=settings.RESEARCH_DPI)
    return seed_images if seed_images else None


def _top_research_hit(search_query: str) -> dict | None:
    candidates = es_service.hybrid_search(
        query=search_query,
        query_vector=embedding_service.get_embeddings(search_query),
        limit=12,
    )
    if not candidates:
        return None
    anchors = _group_by_document(
        candidates,
        max(settings.RESEARCH_TOP_K_DOCS, 1),
        max_papers=max(settings.RESEARCH_TOP_K_DOCS, 1),
        per_paper=1,
    )
    return anchors[0] if anchors else None


def _linkify_page_refs(answer: str, paper_id: str) -> str:
    def repl(match: re.Match) -> str:
        prefix = match.group(1)
        body = match.group(2)
        first = re.search(r"\d+", body)
        if not first:
            return match.group(0)
        url = f"/api/v1/pdf/{paper_id}#page={first.group()}"
        return f'<a href="{url}" target="_blank" class="cite" title="Open page {first.group()}">({prefix} {body})</a>'

    return _PAGE_REF_RE.sub(repl, answer)


def _page_fields(hit: dict) -> tuple[int, int]:
    page_start = hit.get("page_start") or hit.get("page_number") or 1
    page_end = hit.get("page_end") or page_start
    return page_start, page_end


def _group_by_document(candidates: list[dict], limit: int, max_papers: int = 2, per_paper: int = 3) -> list[dict]:
    """Keep the top chunks per document so broad questions get multi-chapter context."""
    papers: dict[str, list[dict]] = {}
    for hit in candidates:
        papers.setdefault(hit["paper_id"], []).append(hit)

    anchors: list[dict] = []
    seen: set[str] = set()
    for _, hits in sorted(papers.items(), key=lambda kv: kv[1][0]["score"], reverse=True)[:max_papers]:
        for hit in hits[:per_paper]:
            if hit["chunk_id"] not in seen:
                anchors.append(hit)
                seen.add(hit["chunk_id"])

    if len(anchors) < limit:
        for hit in candidates:
            if len(anchors) >= limit:
                break
            if hit["chunk_id"] not in seen:
                anchors.append(hit)
                seen.add(hit["chunk_id"])

    return anchors[:limit]


def _build_context(hits: list[dict], window_size: int, max_words: int = _MAX_CONTEXT_WORDS) -> list[dict]:
    entries: list[dict] = []
    seen: set[str] = set()
    total_words = 0
    for hit in hits:
        for doc in es_service.get_chunk_window_entries(hit["chunk_id"], window_size):
            chunk_id = doc.get("chunk_id") or hit["chunk_id"]
            if chunk_id in seen:
                continue
            text = doc.get("text") or ""
            words = len(text.split())
            if entries and total_words + words > max_words:
                continue
            seen.add(chunk_id)
            total_words += words
            page_start = (
                doc.get("page_start") or doc.get("page_number") or hit.get("page_start") or hit.get("page_number") or 1
            )
            page_end = doc.get("page_end") or page_start
            entries.append(
                {
                    "title": doc.get("title") or hit.get("title") or "Course notes",
                    "paper_id": doc.get("paper_id") or hit["paper_id"],
                    "chunk_id": chunk_id,
                    "page_start": page_start,
                    "page_end": page_end,
                    "text": text,
                }
            )
    return entries


def _build_sources(hits: list[dict]) -> list[dict]:
    return [
        {
            "paper_id": hit["paper_id"],
            "title": hit["title"],
            "anchor_chunk_id": hit["chunk_id"],
            "page_number": hit.get("page_number", 1),
            "page_start": hit.get("page_start") or hit.get("page_number", 1),
            "page_end": hit.get("page_end") or hit.get("page_start") or hit.get("page_number", 1),
            "matching_score": hit["score"],
            "source_url": hit.get("source_url") or "",
            "local_pdf": bool(hit.get("local_pdf")),
        }
        for hit in hits
    ]


def _citations(context: list[dict]) -> list[dict]:
    return [
        {"paper_id": e["paper_id"], "chunk_id": e["chunk_id"], "page": e["page_start"], "title": e["title"]}
        for e in context
    ]


def _linkify_citations(answer: str, context: list[dict]) -> str:
    def repl(match: re.Match) -> str:
        idx = int(match.group(1))
        if 1 <= idx <= len(context):
            entry = context[idx - 1]
            url = f"/api/v1/pdf/{entry['paper_id']}?chunk_id={entry['chunk_id']}#page={entry['page_start']}"
            title = entry["title"].replace('"', "")
            return f'<a href="{url}" target="_blank" class="cite" title="{title}">[{idx}]</a>'
        return match.group(0)

    return re.sub(r"\[(\d+)\]", repl, answer)


def _retrieve(search_query: str, limit: int) -> list[dict]:
    candidates = es_service.hybrid_search(
        query=search_query,
        query_vector=embedding_service.get_embeddings(search_query),
        limit=max(limit * 4, 12),
    )
    return _group_by_document(candidates, limit)


@router.get("/ask", response_model=RagResponse)
async def ask_rag(q: str = Query(...), limit: int = 4, window_size: int = 2, mode: str = Query("auto")):
    try:
        search_query = q

        if _research_enabled() and mode != "chunks":
            hit = _top_research_hit(search_query)
            seed_images = _research_seed(hit) if hit else None
            if seed_images:
                evidence_pages: set[int] = set()
                answer = ""
                async for chunk in llm_service.stream_research_response(
                    q,
                    paper_id=hit["paper_id"],
                    title=hit["title"],
                    seed_images=seed_images,
                    callbacks=_research_callbacks(hit["paper_id"], evidence_pages),
                    toc_hint=_build_toc(hit["paper_id"]),
                    max_pages=settings.RESEARCH_MAX_PAGES,
                    evidence_pages=evidence_pages,
                ):
                    answer += chunk
                return RagResponse(
                    question=q,
                    answer=_linkify_page_refs(answer, hit["paper_id"]),
                    search_query=search_query,
                    sources=[SourceHit(**source) for source in _research_sources(hit["paper_id"], hit["title"], evidence_pages)],
                )

        hits = _retrieve(search_query, limit)
        if not hits:
            return RagResponse(question=q, answer="No matching notes found in the CS notes library. Try rephrasing your question.")

        context = _build_context(hits, window_size)
        answer = await llm_service.generate_response(q, context)
        answer = _linkify_citations(answer, context)
        return RagResponse(
            question=q,
            answer=answer,
            search_query=search_query,
            sources=[SourceHit(**source) for source in _build_sources(hits)],
            citations=_citations(context),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG execution failed: {e}")


@router.get("/ask/stream")
async def ask_rag_stream(
    q: str = Query(...),
    chat_id: int = Query(...),
    limit: int = 4,
    window_size: int = 2,
    mode: str = Query("auto"),
    current_user: User = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        await chat_service.get_chat(chat_id, current_user.id, db)
        session_id = str(chat_id)
        history = await memory_store.get_history(session_id)
        search_query = q

        research_hit: dict | None = None
        research_seed: list[dict] | None = None
        research_evidence_pages: set[int] = set()
        if _research_enabled() and mode != "chunks":
            hit = _top_research_hit(search_query)
            seed = _research_seed(hit) if hit else None
            if seed:
                research_hit = hit
                research_seed = seed

        hits = [] if research_hit else _retrieve(search_query, limit)

        if not hits and research_hit is None:
            async def no_results():
                yield "event: sources\ndata: []\n\n"
                yield "event: citations\ndata: []\n\n"
                yield f"event: token\ndata: {json.dumps('No matching notes found in the CS notes library. Try rephrasing your question.')}\n\n"
                yield "event: done\ndata: {}\n\n"

            return StreamingResponse(no_results(), media_type="text/event-stream")

        if research_hit is not None:
            hit = research_hit

            async def research_stream():
                yield f"event: search_query\ndata: {json.dumps(search_query)}\n\n"
                yield f"event: research\ndata: {json.dumps({'paper_id': hit['paper_id'], 'title': hit['title']})}\n\n"

                tokens = []
                async for chunk in llm_service.stream_research_response(
                    q,
                    paper_id=hit["paper_id"],
                    title=hit["title"],
                    seed_images=research_seed,
                    callbacks=_research_callbacks(hit["paper_id"], research_evidence_pages),
                    toc_hint=_build_toc(hit["paper_id"]),
                    history=history,
                    max_pages=settings.RESEARCH_MAX_PAGES,
                    evidence_pages=research_evidence_pages,
                ):
                    tokens.append(chunk)
                    yield f"event: token\ndata: {json.dumps(chunk)}\n\n"

                full_response = "".join(tokens)
                await memory_store.add_message(session_id, "user", q)
                await memory_store.add_message(session_id, "assistant", full_response)
                sources = _research_sources(hit["paper_id"], hit["title"], research_evidence_pages)
                yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
                yield "event: done\ndata: {}\n\n"

            return StreamingResponse(research_stream(), media_type="text/event-stream", headers=_STREAM_HEADERS)

        context = _build_context(hits, window_size)
        sources = _build_sources(hits)
        citations = _citations(context)

        async def event_stream():
            yield f"event: search_query\ndata: {json.dumps(search_query)}\n\n"
            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
            yield f"event: citations\ndata: {json.dumps(citations)}\n\n"

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
