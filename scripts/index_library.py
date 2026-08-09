import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elasticsearch.helpers import bulk

from app.core.config import settings
from app.services.chunking import CHUNK_STRATEGIES, chunk_document, parse_document
from app.services.document import document_processor
from app.services.elasticsearch import es_service
from app.services.embedding import embedding_service
from app.services.pdf import pdf_service

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_metadata(notes_manifest: str) -> dict:
    if not Path(notes_manifest).exists():
        return {}
    with open(notes_manifest, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.values()
    return {
        os.path.basename(item.get("local_path") or item.get("file_name") or item.get("file_path") or item.get("id"))
        : item
        for item in items
        if item.get("local_path") or item.get("file_name") or item.get("file_path") or item.get("id")
    }


def prepare_chunks(file_path: Path, strategy: str | None, chunk_size: int, overlap: int) -> list[dict]:
    suffix = file_path.suffix.lower()
    if strategy in CHUNK_STRATEGIES and suffix == ".md":
        doc = parse_document(file_path.read_text(encoding="utf-8", errors="ignore"))
        return [c.__dict__ for c in chunk_document(doc, strategy, target=chunk_size, overlap=overlap)]

    if suffix == ".pdf":
        return document_processor.chunk_pdf(str(file_path), chunk_size=chunk_size, overlap=overlap)
    if suffix == ".md":
        return document_processor.chunk_markdown(file_path.read_text(encoding="utf-8", errors="ignore"))
    return document_processor.chunk_text(file_path.read_text(encoding="utf-8", errors="ignore"), chunk_size, overlap)


def build_actions(
    metadata_map: dict,
    notes_dir: Path,
    index_name: str,
    strategy: str | None,
    chunk_size: int,
    overlap: int,
    only_md: bool,
    require_pdf: bool = True,
) -> list[dict]:
    document_files = [f for f in notes_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if only_md:
        document_files = [f for f in document_files if f.suffix.lower() == ".md"]

    actions = []
    for file_path in sorted(document_files):
        meta = metadata_map.get(file_path.name, {})
        paper_id = meta.get("id") or meta.get("paper_id") or file_path.stem
        if require_pdf and not pdf_service.has_local_pdf(paper_id):
            print(f"  - skipping {file_path.name}: no local PDF available")
            continue
        chunks = prepare_chunks(file_path, strategy, chunk_size, overlap)

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{paper_id}_chunk_{idx}"
            source = {
                "paper_id": paper_id,
                "title": meta.get("title") or file_path.stem,
                "authors": meta.get("authors") or [],
                "category": meta.get("category") or "Uncategorized",
                "chunk_id": chunk_id,
                "page_number": chunk.get("page", 1),
                "text": chunk["text"],
                "highlight_boxes": chunk.get("highlight_boxes", {}),
                "source_url": meta.get("url") or "",
                "local_pdf": bool(pdf_service.has_local_pdf(paper_id)) or bool(meta.get("pdf_path")),
                "vector_embedding": embedding_service.get_embeddings(chunk["text"]),
            }
            if strategy in CHUNK_STRATEGIES:
                source["span_start"] = chunk["span_start"]
                source["span_end"] = chunk["span_end"]
                source["strategy"] = strategy
            actions.append({"_index": index_name, "_id": chunk_id, "_source": source})
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the notes (and optional library PDFs) into Elasticsearch.")
    parser.add_argument("--index", default=settings.INDEX_NAME)
    parser.add_argument("--notes-dir", default=settings.NOTES_DIR)
    parser.add_argument("--strategy", choices=CHUNK_STRATEGIES, default=None)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=45)
    parser.add_argument("--only-md", action="store_true", default=False)
    parser.add_argument("--no-require-pdf", action="store_true", default=False,
                        help="Include notes even when no local PDF is available (default: only PDF-backed notes).")
    parser.add_argument("--delete-first", action="store_true", default=False)
    args = parser.parse_args()

    if not es_service.verify_connection():
        print("Elasticsearch connection failed.")
        return

    if args.delete_first:
        es_service.delete_index(index=args.index)

    es_service.create_index_if_not_exists(index=args.index)

    metadata_map = load_metadata(str(Path(args.notes_dir) / "library_metadata.json"))
    actions = build_actions(
        metadata_map,
        Path(args.notes_dir),
        args.index,
        args.strategy,
        args.chunk_size,
        args.overlap,
        args.only_md,
        require_pdf=not args.no_require_pdf,
    )
    if not actions:
        print("No chunks prepared for indexing.")
        return

    print(f"Bulk indexing {len(actions)} chunks into '{args.index}'...")
    try:
        success, failed = bulk(
            es_service.client.options(request_timeout=300, retry_on_timeout=True, max_retries=3),
            actions,
            chunk_size=30,
            stats_only=True,
            raise_on_error=False,
        )
        es_service.refresh_index(index=args.index)
        print(f"Indexed {success} chunks, {failed} failed.")
    except Exception as e:
        print(f"Indexing failed: {e}")


if __name__ == "__main__":
    main()