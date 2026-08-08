import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elasticsearch.helpers import bulk

from app.core.config import settings
from app.services.document import document_processor
from app.services.elasticsearch import es_service
from app.services.embedding import embedding_service

LIBRARY_DIR = Path(settings.NOTES_DIR)
METADATA_FILE = Path(settings.NOTES_MANIFEST)
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_metadata() -> dict:
    if not METADATA_FILE.exists():
        return {}
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.values()
    return {
        os.path.basename(item.get("local_path") or item.get("file_name") or item.get("file_path") or item.get("id"))
        : item
        for item in items
        if item.get("local_path") or item.get("file_name") or item.get("file_path") or item.get("id")
    }


def build_actions(metadata_map: dict) -> list[dict]:
    es_service.create_index_if_not_exists()
    document_files = [f for f in LIBRARY_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    actions = []

    for file_path in document_files:
        meta = metadata_map.get(file_path.name, {})
        paper_id = meta.get("id") or meta.get("paper_id") or file_path.stem
        if file_path.suffix.lower() == ".pdf":
            chunks = document_processor.chunk_pdf(str(file_path))
        elif file_path.suffix.lower() == ".md":
            chunks = document_processor.chunk_markdown(file_path.read_text(encoding="utf-8", errors="ignore"))
        else:
            chunks = document_processor.chunk_text(file_path.read_text(encoding="utf-8", errors="ignore"))

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{paper_id}_chunk_{idx}"
            actions.append(
                {
                    "_index": settings.INDEX_NAME,
                    "_id": chunk_id,
                    "_source": {
                        "paper_id": paper_id,
                        "title": meta.get("title") or file_path.stem,
                        "authors": meta.get("authors") or [],
                        "category": meta.get("category") or "Uncategorized",
                        "chunk_id": chunk_id,
                        "page_number": chunk.get("page", 1),
                        "text": chunk["text"],
                        "highlight_boxes": chunk.get("highlight_boxes", {}),
                        "source_url": meta.get("url") or "",
                        "local_pdf": bool(meta.get("pdf_path")),
                        "vector_embedding": embedding_service.get_embeddings(chunk["text"]),
                    },
                }
            )
    return actions


def main() -> None:
    if not es_service.verify_connection():
        print("Elasticsearch connection failed.")
        return

    metadata_map = load_metadata()
    actions = build_actions(metadata_map)
    if not actions:
        print("No chunks prepared for indexing.")
        return

    print(f"Bulk indexing {len(actions)} chunks...")
    try:
        success, failed = bulk(
            es_service.client.options(request_timeout=120, retry_on_timeout=True, max_retries=3),
            actions,
            chunk_size=30,
            stats_only=True,
        )
        es_service.refresh_index()
        print(f"Indexed {success} chunks, {failed} failed.")
    except Exception as e:
        print(f"Indexing failed: {e}")


if __name__ == "__main__":
    main()
