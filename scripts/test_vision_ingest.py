import os
import sys
import time
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.services.document import document_processor
from app.services.elasticsearch import es_service
from app.services.embedding import embedding_service

TEST_INDEX = "csc_notes_vision_test"
TEST_PDF = "ml_a_course_in_machine_learning_ch01.pdf"

async def test_vision_ingest():
    print("Starting Vision-Based PDF Ingestion Test...")
    print(f"Target index: {TEST_INDEX}")

    if not es_service.verify_connection():
        print("Error: Elasticsearch is not running.")
        sys.exit(1)

    es_service.delete_index(TEST_INDEX)
    es_service.create_index_if_not_exists(TEST_INDEX)

    notes_dir = Path(settings.NOTES_DIR)
    pdfs_dir = notes_dir / "pdfs"
    pdf_path = pdfs_dir / TEST_PDF

    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found. Make sure fetch_notes.py was run.")
        sys.exit(1)

    total_chunks = 0
    start_time = time.time()

    print(f"\nProcessing {TEST_PDF} via Vision LLM...")
    doc_start = time.time()

    try:
        chunks = await document_processor.chunk_pdf_vision(str(pdf_path))
        print(f"  - Generated {len(chunks)} chunks in {time.time() - doc_start:.2f}s")

        paper_id = f"vision_{pdf_path.stem}"
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{paper_id}_chunk_{idx}"
            es_service.client.index(
                index=TEST_INDEX,
                id=chunk_id,
                document={
                    "paper_id": paper_id,
                    "title": f"Vision - {pdf_path.stem}",
                    "authors": ["Vision OCR"],
                    "category": "Machine Learning",
                    "chunk_id": chunk_id,
                    "page_number": chunk.get("page", 1),
                    "text": chunk["text"],
                    "highlight_boxes": chunk.get("highlight_boxes", {}),
                    "source_url": "",
                    "local_pdf": True,
                    "vector_embedding": embedding_service.get_embeddings(chunk["text"]),
                }
            )
        total_chunks += len(chunks)
        print(f"  - Successfully indexed {len(chunks)} chunks for {TEST_PDF}")
    except Exception as e:
        print(f"  - Error processing {TEST_PDF}: {e}")

    es_service.refresh_index(TEST_INDEX)
    duration = time.time() - start_time
    print(f"\nCompleted vision ingestion test in {duration:.2f}s!")
    print(f"Total chunks indexed: {total_chunks}")

    res = es_service.client.count(index=TEST_INDEX)
    count = res["count"]
    print(f"Elasticsearch document count in '{TEST_INDEX}': {count}")
    assert count == total_chunks, f"Expected {total_chunks} documents, got {count} in ES."
    print("SUCCESS: Vision-based PDF ingestion verified!")

if __name__ == "__main__":
    asyncio.run(test_vision_ingest())
