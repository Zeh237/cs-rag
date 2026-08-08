import json
import os
import shutil
import uuid
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import PROJECT_ROOT, settings
from app.services.document import document_processor
from app.services.elasticsearch import es_service
from app.services.embedding import embedding_service
from app.services.ingestion import ingestion_manager
from app.services.pdf import pdf_service

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}


def _read_manifest() -> list[dict]:
    if not os.path.exists(settings.NOTES_MANIFEST):
        return []
    with open(settings.NOTES_MANIFEST, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _process_document(paper_id: str, local_path: str, ext: str, meta: dict) -> dict:
    try:
        if ext == ".pdf":
            chunks = document_processor.chunk_pdf(local_path)
        elif ext == ".md":
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                chunks = document_processor.chunk_markdown(f.read())
        else:
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                chunks = document_processor.chunk_text(f.read())

        if not chunks:
            raise ValueError("No readable text content extracted from the document.")

        es_service.create_index_if_not_exists()
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{paper_id}_chunk_{idx}"
            es_service.index_chunk(
                chunk_id,
                {
                    "paper_id": paper_id,
                    "title": meta["title"],
                    "authors": meta["authors"],
                    "category": meta["primary_category"],
                    "chunk_id": chunk_id,
                    "page_number": chunk.get("page", 1),
                    "text": chunk["text"],
                    "highlight_boxes": chunk.get("highlight_boxes", {}),
                    "source_url": meta["url"],
                    "local_pdf": ext == ".pdf",
                    "vector_embedding": embedding_service.get_embeddings(chunk["text"]),
                },
            )
        es_service.refresh_index()

        relative_path = os.path.relpath(local_path, PROJECT_ROOT)
        entry = {
            "id": paper_id,
            "title": meta["title"],
            "authors": meta["authors"],
            "category": meta["primary_category"],
            "local_path": relative_path,
            "url": meta["url"],
            "uploaded_at": datetime.now().isoformat(),
        }
        if ext == ".pdf":
            entry["pdf_path"] = relative_path

        manifest = _read_manifest()
        if not any(item.get("id") == paper_id for item in manifest):
            manifest.append(entry)
            os.makedirs(settings.NOTES_DIR, exist_ok=True)
            with open(settings.NOTES_MANIFEST, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=4, ensure_ascii=False)
    except Exception:
        if os.path.exists(local_path):
            os.remove(local_path)
        raise

    return {
        "paper_id": paper_id,
        "chunks_indexed": len(chunks),
        "message": f"Successfully ingested document: '{meta['title']}'",
    }


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    authors: str = Form(...),
    primary_category: str = Form(...),
    all_categories: str = Form(...),
    url: str | None = Form(None),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .pdf, .md, or .txt documents are allowed.")

    os.makedirs(settings.NOTES_DIR, exist_ok=True)
    paper_id = uuid.uuid4().hex[:8]
    local_path = os.path.join(settings.NOTES_DIR, f"{paper_id}_{file.filename}")

    try:
        with open(local_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save file.")

    meta = {
        "title": title,
        "authors": [a.strip() for a in authors.split(",") if a.strip()],
        "primary_category": primary_category,
        "url": url or "",
    }
    job_id = ingestion_manager.submit(_process_document, paper_id, local_path, ext, meta)

    return {
        "job_id": job_id,
        "paper_id": paper_id,
        "status": "queued",
        "message": "Upload complete. Processing in the background.",
    }


@router.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    job = ingestion_manager.get_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    return job


@router.get("/pdf/{paper_id}")
def serve_pdf(paper_id: str, chunk_id: str | None = None):
    try:
        return pdf_service.serve(paper_id, chunk_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/library")
async def get_library():
    if not os.path.exists(settings.MANIFEST_PATH):
        return []
    with open(settings.MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)