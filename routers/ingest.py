import os
import uuid
import shutil
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

from config import settings
from services.document import chunk_pdf, append_to_manifest
from services.embedding import embedding_service
from services.elasticsearch_service import es_service

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])

@router.post("/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    authors: str = Form(...),  # Comma separated authors
    primary_category: str = Form(...),
    all_categories: str = Form(...),  # Comma separated categories
    url: Optional[str] = Form(None)
):
    # 1. Validation
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are allowed.")

    # Create destination directory
    os.makedirs(settings.LIBRARY_DIR, exist_ok=True)
    
    paper_id = str(uuid.uuid4())[:8]
    local_path = os.path.join(settings.LIBRARY_DIR, f"{paper_id}_{file.filename}")

    # 2. Persist PDF to local library directory
    try:
        with open(local_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Parse metadata lists
    author_list = [a.strip() for a in authors.split(",")]
    category_list = [c.strip() for c in all_categories.split(",")]

    # 3. Processing and Indexing Workflow
    try:
        # Generate chunks using upgraded PyMuPDF4LLM logic
        chunks = chunk_pdf(local_path)
        if not chunks:
            raise ValueError("No readable text content extracted from the document.")
        
        # Ensure target index exists
        es_service.create_index_if_not_exists()

        # Vectorize and Index chunks
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{paper_id}_chunk_{idx}"
            chunk_text = chunk["text"]
            chunk_page = chunk.get("page", 1)

            # FIXED: Calls actual method get_embeddings
            embedding = embedding_service.get_embeddings(chunk_text)

            es_document = {
                "paper_id": paper_id,
                "title": title,
                "authors": author_list,
                "category": primary_category,
                "chunk_id": chunk_id,
                "page_number": chunk_page,
                "text": chunk_text,
                "vector_embedding": embedding
            }
            
            # Indexes chunk to ES
            es_service.index_chunk(chunk_id, es_document)

        # Sync Elasticsearch 
        es_service.refresh_index()

        # 4. Update JSON Manifest
        manifest_data = {
            "id": paper_id,
            "title": title,
            "authors": author_list,
            "published_year": 2026,
            "primary_category": primary_category,
            "all_categories": category_list,
            "local_path": local_path,
            "url": url if url else ""
        }
        append_to_manifest(manifest_data)

    except Exception as e:
        # Cleanup local file if processing fails midway
        if os.path.exists(local_path):
            os.remove(local_path)
        raise HTTPException(status_code=500, detail=f"Ingestion processing failed: {str(e)}")

    return {
        "status": "success",
        "message": f"Successfully ingested paper: '{title}'",
        "paper_id": paper_id,
        "chunks_indexed": len(chunks)
    }


@router.get("/pdf/{paper_id}")
async def serve_pdf(paper_id: str, chunk_id: str = None):
    """
    Serves the original PDF file for a given paper_id.
    If chunk_id is provided, adds yellow highlight annotations on the
    relevant page before serving so the user can see the source region.
    """
    import json
    import io
    import fitz

    manifest_path = settings.MANIFEST_PATH
    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail="Library manifest not found.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    entry = next(
        (
            item for item in manifest
            if item.get("id") == paper_id
            or os.path.splitext(os.path.basename(item.get("local_path", "")))[0] == paper_id
        ),
        None
    )
    if not entry:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found in manifest.")

    local_path = entry.get("local_path", "")
    if not local_path:
        raise HTTPException(status_code=404, detail="PDF file not found on disk.")

    # Resolve relative paths against the project root (directory of this file's parent)
    if not os.path.isabs(local_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_path = os.path.join(project_root, local_path)

    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="PDF file not found on disk.")

    # ── If no chunk_id, serve the plain PDF ──────────────────────────────
    if not chunk_id:
        return FileResponse(
            local_path,
            media_type="application/pdf",
            headers={"Content-Disposition": "inline"}
        )

    # ── Highlight the chunk's text region ────────────────────────────────
    try:
        # Fetch chunk text + page from Elasticsearch
        chunk_doc = es_service.client.get(index=settings.INDEX_NAME, id=chunk_id)
        chunk_text = chunk_doc["_source"]["text"]
        page_num = chunk_doc["_source"].get("page_number", 1)  # 1-indexed

        doc = fitz.open(local_path)

        # Generate overlapping phrases to cover the ENTIRE chunk text
        words = chunk_text.split()
        phrases = []
        phrase_len = 12
        step = 8  # overlap by 4 words to handle line breaks and page transitions smoothly
        for start in range(0, len(words), step):
            phrase = " ".join(words[start:start + phrase_len])
            if len(phrase) > 5:
                phrases.append(phrase)

        # Search on the target page and the one after (chunks can span pages)
        target_indices = range(
            max(0, page_num - 1),             # fitz is 0-indexed
            min(len(doc), page_num + 1)
        )
        for pg_idx in target_indices:
            page = doc[pg_idx]
            for phrase in phrases:
                rects = page.search_for(phrase)
                for rect in rects:
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=(1, 0.8, 0))   # yellow highlight
                    annot.update()

        # Write annotated PDF to memory buffer (original file untouched)
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        buf.seek(0)

        from starlette.responses import StreamingResponse
        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={"Content-Disposition": "inline"}
        )

    except Exception as e:
        # If highlighting fails, fall back to serving the plain PDF
        print(f"Warning: highlight failed for chunk {chunk_id}: {e}")
        return FileResponse(
            local_path,
            media_type="application/pdf",
            headers={"Content-Disposition": "inline"}
        )



@router.get("/library")
async def get_library():
    """Returns the full library manifest (list of all ingested papers)."""
    import json
    manifest_path = settings.MANIFEST_PATH
    if not os.path.exists(manifest_path):
        return JSONResponse(content=[])
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return JSONResponse(content=manifest)