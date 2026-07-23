import os
import json
import re
from pathlib import Path
from elasticsearch.helpers import bulk
import fitz  # PyMuPDF
# pyrefly: ignore [missing-import]
import pymupdf4llm

# Import your configured services
from services.elasticsearch_service import es_service
from services.embedding import embedding_service
from config import settings

# Configurations
LIBRARY_DIR = Path("./csc_library")
METADATA_FILE = LIBRARY_DIR / "library_metadata.json"
CACHE_FILE = LIBRARY_DIR / "indexing_cache.json"  # JSON staging cache path

CHUNK_SIZE = 1000  # Target characters per chunk
CHUNK_OVERLAP = 200  # Overlap to preserve context


def load_metadata():
    """Loads the metadata JSON file."""
    if not METADATA_FILE.exists():
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return {}
    
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    metadata_map = {}
    if isinstance(data, list):
        for item in data:
            # Prefer local_path (actual filename on disk) as the lookup key,
            # then fall back to file_name / file_path / id fields.
            file_key = (
                item.get("local_path")
                or item.get("file_name")
                or item.get("file_path")
                or item.get("id")
            )
            if file_key:
                metadata_map[os.path.basename(file_key)] = item
    elif isinstance(data, dict):
        for key, val in data.items():
            metadata_map[os.path.basename(key)] = val
            
    return metadata_map


def strip_bibliography(markdown_text: str) -> str:
    """
    Finds common academic bibliography headers and truncates the text
    to drop everything after them, avoiding the bibliography index trap.
    """
    bibliography_patterns = [
        r"(?i)(?:\n|^)#+\s+References\b",
        r"(?i)(?:\n|^)#+\s+Bibliography\b",
        r"(?i)(?:\n|^)#+\s+Works Cited\b"
    ]
    
    for pattern in bibliography_patterns:
        match = re.search(pattern, markdown_text)
        if match:
            print(f"  [Post-Process] Truncated bibliography starting at: '{match.group().strip()}'")
            return markdown_text[:match.start()].strip()
            
    return markdown_text


def extract_and_clean_pdf_with_pymupdf(file_path: Path) -> list[tuple[int, str]]:
    """
    Parses a PDF page-by-page using fitz, returning a list of (page_number, text) tuples.
    Handles bibliography stripping on the combined text.
    """
    try:
        print(f"  -> Extracting text page-by-page via PyMuPDF...")
        doc = fitz.open(str(file_path))
        page_texts = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                page_texts.append((page_num, text))
        doc.close()

        if not page_texts:
            return []

        # Build word list with page annotations for bibliography stripping
        word_pages = []
        for page_num, text in page_texts:
            for word in text.split():
                word_pages.append((word, page_num))

        combined_text = " ".join(w for w, _ in word_pages)
        bib_patterns = [
            r"(?i)(?:\n|^|\s)References\b",
            r"(?i)(?:\n|^|\s)Bibliography\b",
            r"(?i)(?:\n|^|\s)Works Cited\b"
        ]
        cutoff = len(word_pages)
        for pattern in bib_patterns:
            match = re.search(pattern, combined_text)
            if match:
                char_pos = match.start()
                word_idx = len(combined_text[:char_pos].split())
                if word_idx < cutoff:
                    print(f"  [Post-Process] Truncated bibliography at word index {word_idx}")
                    cutoff = word_idx

        return word_pages[:cutoff]

    except Exception as e:
        print(f"  Error parsing PDF {file_path.name} with PyMuPDF: {e}")
        return []


def chunk_word_pages(word_pages: list[tuple[str, int]], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Splits a list of (word, page_number) tuples into overlapping chunks.
    Each chunk is a dict: {'text': str, 'page': int}.
    """
    chunks = []
    i = 0
    while i < len(word_pages):
        chunk_slice = word_pages[i : i + chunk_size]
        chunk_text = " ".join(w for w, _ in chunk_slice)
        chunk_page = chunk_slice[0][1]  # page of first word in chunk
        if chunk_text.strip():
            chunks.append({"text": chunk_text.strip(), "page": chunk_page})
        if i + chunk_size >= len(word_pages):
            break
        i += (chunk_size - overlap)
    return chunks


def build_bulk_actions(metadata_map):
    """Processes document files, formats them, and prepares index commands."""
    actions = []
    
    es_service.create_index_if_not_exists()
    
    supported_extensions = {".pdf", ".txt", ".md"}
    document_files = [f for f in LIBRARY_DIR.iterdir() if f.suffix.lower() in supported_extensions]
    
    if not document_files:
        print(f"Warning: No matching files found in {LIBRARY_DIR}")
        return actions

    print(f"Found {len(document_files)} documents to process.")

    for file_path in document_files:
        filename = file_path.name
        print(f"\nProcessing: {filename}...")
        
        meta = metadata_map.get(filename, {})
        paper_id = meta.get("id") or meta.get("paper_id") or file_path.stem
        title = meta.get("title") or file_path.stem
        authors = meta.get("authors") or []
        category = meta.get("category") or "Uncategorized"
        
        if file_path.suffix.lower() == ".pdf":
            word_pages = extract_and_clean_pdf_with_pymupdf(file_path)
            if not word_pages:
                print(f"Warning: Skipping {filename}: No content extracted.")
                continue
            chunks = chunk_word_pages(word_pages)
        else:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                # For non-PDF files, no page info — assign page 1
                words = raw_text.split()
                word_pages_fallback = [(w, 1) for w in words]
                chunks = chunk_word_pages(word_pages_fallback)
            except Exception as e:
                print(f"Warning: Failed to read text file {filename}: {e}")
                continue

        if not chunks:
            print(f"Warning: Skipping {filename}: No content extracted.")
            continue

        print(f"  -> Split into {len(chunks)} structural chunks. Generating embeddings...")

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{paper_id}_chunk_{idx}"
            chunk_text_val = chunk["text"]
            chunk_page = chunk["page"]
            embedding = embedding_service.get_embeddings(chunk_text_val)

            action = {
                "_index": settings.INDEX_NAME,
                "_id": chunk_id,
                "_source": {
                    "paper_id": paper_id,
                    "title": title,
                    "authors": authors,
                    "category": category,
                    "chunk_id": chunk_id,
                    "page_number": chunk_page,
                    "text": chunk_text_val,
                    "vector_embedding": embedding
                }
            }
            actions.append(action)
            
    return actions


def load_cached_actions() -> list | None:
    """Attempts to load pre-calculated actions from the JSON cache file."""
    if CACHE_FILE.exists():
        try:
            print(f"\n[CACHE DETECTED] Loading cached chunks & embeddings from {CACHE_FILE}...")
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                actions = json.load(f)
            print(f"-> Successfully loaded {len(actions)} actions from cache. Skipping visual/embedding pipelines.")
            return actions
        except Exception as e:
            print(f"Warning: Failed to read cache file: {e}. Re-processing documents.")
    return None


def save_actions_to_cache(actions: list):
    """Saves the fully-processed Elasticsearch actions to a JSON file."""
    try:
        print(f"\n[CACHE] Saving {len(actions)} processed actions to {CACHE_FILE}...")
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # compact format to minimize disk I/O write time
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(actions, f, ensure_ascii=False)
        print("[CACHE] Cache write complete!")
    except Exception as e:
        print(f"Warning: Failed to write cache file: {e}")


def clear_cache():
    """Removes the JSON cache file upon successful ingestion."""
    if CACHE_FILE.exists():
        try:
            CACHE_FILE.unlink()
            print("[CACHE] Indexing completed successfully. Cache cleared.")
        except Exception as e:
            print(f"Warning: Failed to delete cache file: {e}")


def main():
    print("Starting local ultra-high-speed document ingestion pipeline...")
    
    if not es_service.verify_connection():
        print("Error: Elasticsearch connection failed.")
        return
        
    # Step 1: Check if we have a valid cache we can restore from
    actions = load_cached_actions()
    
    # Step 2: Cache Miss - Run the pipeline and generate the cache
    if not actions:
        metadata_map = load_metadata()
        actions = build_bulk_actions(metadata_map)
        
        if not actions:
            print("Error: No chunks prepared for indexing.")
            return
            
        save_actions_to_cache(actions)
        
    # Step 3: Index the data into Elasticsearch
    print(f"\nBulk indexing {len(actions)} total chunks into Elasticsearch...")
    
    try:
        client_with_options = es_service.client.options(
            request_timeout=120,
            retry_on_timeout=True,
            max_retries=3
        )
        
        success, failed = bulk(
            client_with_options, 
            actions, 
            chunk_size=30,
            stats_only=True,
            request_timeout=120
        )
        es_service.refresh_index()
        
        print(f"\nIngestion complete!")
        print(f"Successfully indexed: {success} chunks.")
        print(f"Failed chunks: {failed}")
        
        # Ingestion succeeded, clean up the cache
        clear_cache()
        
    except Exception as e:
        print(f"\nError during indexing: {e}")
        print(f"[RECOVERY INFO] Your parsed data is safe inside: {CACHE_FILE}")
        print("Fix the Elasticsearch connection/service issues and run the script again to resume instantly.")


if __name__ == "__main__":
    main()