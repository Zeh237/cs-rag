import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

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
from app.services.page_cleaner import clean_page_texts

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

EXTRACTION_METHODS = ("vlm", "fitz-raw", "fitz-clean")

_EXTRACTION_ALIASES = {
    "fitz_raw": "fitz-raw",
    "fitz_clean": "fitz-clean",
    "fitz+clean": "fitz-clean",
    "fitz+raw": "fitz-raw",
}

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

def _normalise_extraction(method: str | None) -> str | None:
    if method is None:
        return None
    m = method.strip().lower().replace("_", "-")
    return _EXTRACTION_ALIASES.get(m, m)

def _load_vlm_markdown(pdf_path: Path) -> str | None:
    """Return cached VLM markdown for *pdf_path* if it exists.

    The cache lives in ``<NOTES_DIR>/vision_markdown/<stem>_<hash>.md``
    (see ``app/services/document.py:DocumentProcessor._vision_cache_path``).
    We also check ``csc_notes/vision_markdown`` directly as used by
    ``eval/compare_pdf_methods.py``.
    """

    try:
        cached = document_processor._load_vision_markdown_cache(str(pdf_path))
        if cached is not None:
            return "\n\n".join(f"[[PAGE {p}]]\n{t}" for p, t in cached if t and t.strip() != "[SKIP]")
    except Exception:
        pass

    for candidate_dir in [Path(settings.NOTES_DIR) / "vision_markdown", Path("csc_notes/vision_markdown")]:
        if candidate_dir.is_dir():
            for f in candidate_dir.glob("*.md"):

                if f.stem.startswith(pdf_path.stem):
                    try:
                        return f.read_text(encoding="utf-8")
                    except Exception:
                        continue
    return None

def _extract_pdf_markdown(pdf_path: Path, method: str) -> str | None:
    """Extract *pdf_path* to markdown string with ``[[PAGE N]]`` markers."""
    if method == "vlm":
        md = _load_vlm_markdown(pdf_path)
        if md is not None:
            print(f"  + VLM cache hit for {pdf_path.name}")
            return md
        print(f"  * VLM cache miss for {pdf_path.name}; calling LM Studio ...")
        try:
            import asyncio

            page_texts = asyncio.run(document_processor._transcribe_pdf_pages(str(pdf_path)))
            document_processor._save_vision_markdown_cache(str(pdf_path), page_texts)
            md_live = "\n\n".join(f"[[PAGE {p}]]\n{t}" for p, t in page_texts if t and t.strip() != "[SKIP]")
            if md_live.strip():
                print(f"  + VLM transcribed {len(page_texts)} pages for {pdf_path.name}")
                return md_live
        except Exception as e:
            print(f"  ! VLM transcription failed for {pdf_path.name}: {e}; falling back to fitz-clean")
        method = "fitz-clean"

    if method == "fitz-raw":
        import fitz

        doc = fitz.open(str(pdf_path))
        parts = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            parts.append(f"[[PAGE {page_num}]]\n{text}")
        doc.close()
        return "\n\n".join(parts)

    if method == "fitz-clean":
        import fitz

        doc = fitz.open(str(pdf_path))
        page_texts = [(page_num, page.get_text("text")) for page_num, page in enumerate(doc, start=1)]
        doc.close()
        cleaned = clean_page_texts(page_texts)
        parts = [f"[[PAGE {page_num}]]\n{text}" for page_num, text in cleaned if text.strip()]
        return "\n\n".join(parts)

    return None

def prepare_chunks(
    file_path: Path,
    strategy: str | None,
    chunk_size: int,
    overlap: int,
    extraction_method: str | None = None,
) -> list[dict]:
    suffix = file_path.suffix.lower()
    extraction_method = _normalise_extraction(extraction_method)

    if suffix == ".pdf" and extraction_method in EXTRACTION_METHODS:
        md_text = _extract_pdf_markdown(file_path, extraction_method)
        if md_text is None:

            return document_processor.chunk_pdf(str(file_path), chunk_size=chunk_size, overlap=overlap)
        if strategy in CHUNK_STRATEGIES:
            doc = parse_document(md_text)
            return [c.__dict__ for c in chunk_document(doc, strategy, target=chunk_size, overlap=overlap)]

        return document_processor.chunk_markdown(md_text)

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
    extraction_method: str | None = None,
    num_docs: int | None = None,
) -> list[dict]:
    document_files = [f for f in notes_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if only_md:
        document_files = [f for f in document_files if f.suffix.lower() == ".md"]

    document_files = sorted(document_files)

    if num_docs is not None:
        try:
            n = int(num_docs)
            if n > 0:
                document_files = document_files[: min(n, len(document_files))]
            elif n == 0:
                document_files = []
        except (TypeError, ValueError):
            pass

    extraction_method = _normalise_extraction(extraction_method)

    actions = []
    for file_path in document_files:
        meta = metadata_map.get(file_path.name, {})
        paper_id = meta.get("id") or meta.get("paper_id") or file_path.stem
        if require_pdf and not pdf_service.has_local_pdf(paper_id):
            print(f"  - skipping {file_path.name}: no local PDF available")
            continue
        chunks = prepare_chunks(file_path, strategy, chunk_size, overlap, extraction_method)

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{paper_id}_chunk_{idx}"
            page_start = chunk.get("page_start") or chunk.get("page", 1) or 1
            page_end = chunk.get("page_end") or page_start
            source = {
                "paper_id": paper_id,
                "title": meta.get("title") or file_path.stem,
                "authors": meta.get("authors") or [],
                "category": meta.get("category") or "Uncategorized",
                "chunk_id": chunk_id,
                "page_number": page_start,
                "page_start": page_start,
                "page_end": page_end,
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
            if extraction_method:
                source["extraction_method"] = extraction_method
            actions.append({"_index": index_name, "_id": chunk_id, "_source": source})
    return actions

def main() -> None:
    parser = argparse.ArgumentParser(description="Index the notes into Elasticsearch.")
    parser.add_argument("--index", default=settings.INDEX_NAME)
    parser.add_argument("--notes-dir", default=settings.NOTES_DIR)
    parser.add_argument("--strategy", choices=CHUNK_STRATEGIES, default=None)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=45)
    parser.add_argument(
        "--extraction-method",
        choices=["vlm", "fitz-raw", "fitz-clean", "fitz_raw", "fitz_clean"],
        default=None,
        help=(
            "PDF text-extraction method.  Only applies to .pdf inputs. "
            "Choices: vlm (cached vision markdown, falls back to fitz-clean), "
            "fitz-raw (raw fitz.get_text), fitz-clean (fitz + clean_page_texts). "
            "When omitted the default pdf chunking (fitz + cleaning + word-window) is used."
        ),
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=None,
        help=(
            "Maximum number of documents to index.  Files are sorted "
            "deterministically and the first N are used.  If omitted all "
            "documents are used; if N exceeds the corpus size all documents "
            "are used."
        ),
    )
    parser.add_argument("--only-md", action="store_true", default=False)
    parser.add_argument("--no-require-pdf", action="store_true", default=False,
                        help="Include notes even when no local PDF is available (default: only PDF-backed notes).")
    parser.add_argument("--delete-first", action="store_true", default=False)
    args = parser.parse_args()

    args.extraction_method = _normalise_extraction(args.extraction_method)

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
        extraction_method=args.extraction_method,
        num_docs=args.num_docs,
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