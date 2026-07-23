import os
import re
import json
from pathlib import Path
from pypdf import PdfReader
from elasticsearch.helpers import bulk

# Import the services we configured earlier
from services.elasticsearch_service import es_service
from services.embedding import embedding_service
from config import settings

# Configurations
LIBRARY_DIR = Path("./csc_library")
METADATA_FILE = LIBRARY_DIR / "library_metadata.json"
CHUNK_SIZE = 1000  # Target characters per chunk (~200-250 words)
CHUNK_OVERLAP = 200  # Overlap to preserve context across chunks


def load_metadata():
    """Loads the metadata JSON file."""
    if not METADATA_FILE.exists():
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return {}
    
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Standardize format: We want a dict where keys are filenames
    metadata_map = {}
    if isinstance(data, list):
        for item in data:
            # Try to grab the identifier or filename
            file_key = item.get("file_name") or item.get("file_path") or item.get("id")
            if file_key:
                metadata_map[os.path.basename(file_key)] = item
    elif isinstance(data, dict):
        for key, val in data.items():
            metadata_map[os.path.basename(key)] = val
            
    return metadata_map


def extract_text(file_path: Path) -> str:
    """Extracts raw text from PDF or TXT files."""
    if file_path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            print(f"Warning: Failed to parse PDF {file_path.name}: {e}")
            return ""
    elif file_path.suffix.lower() in [".txt", ".md"]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Failed to read text file {file_path.name}: {e}")
            return ""
    return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Splits a document text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    # We reconstruct chunks using approximate character lengths
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # count the space
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # Keep the overlap: back up by sliding window
            overlap_words = current_chunk[-max(1, int(overlap / 5)):] # Approx 5 chars per word
            current_chunk = overlap_words
            current_length = sum(len(w) + 1 for w in current_chunk)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return [c.strip() for c in chunks if c.strip()]


def is_bibliography_chunk(text: str) -> bool:
    """
    Heuristically checks if a chunk is primarily academic bibliography or references.
    This prevents Reference sections from getting indexed and matching keyword searches falsely.
    """
    lower_text = text.lower().strip()
    
    # 1. Check if the text explicitly starts with a header indicating references
    if lower_text.startswith("references") or lower_text.startswith("bibliography"):
        return True
        
    # 2. Count citation matches like [1], [24], etc.
    citation_bracket_count = len(re.findall(r"\[\d+\]", text))
    
    # 3. Check for high density of bibliography-specific keywords
    bib_keywords = ["proceedings of", "journal of", "ieee", "acm", "pp.", "vol.", "no.", "press", "arxiv:"]
    keyword_matches = sum(1 for kw in bib_keywords if kw in lower_text)
    
    # Heuristic trigger: heavy bracket density OR a mix of brackets and publishing metadata
    if citation_bracket_count >= 4:
        return True
    if citation_bracket_count >= 2 and keyword_matches >= 3:
        return True
        
    return False


def build_bulk_actions(metadata_map):
    """Iterates through papers, chunks them, filters out bibliography, embeds them, and prepares bulk actions."""
    actions = []
    
    # Ensure Elasticsearch index is set up
    es_service.create_index_if_not_exists()
    
    # Scan library directory for documents
    supported_extensions = {".pdf", ".txt", ".md"}
    document_files = [f for f in LIBRARY_DIR.iterdir() if f.suffix.lower() in supported_extensions]
    
    if not document_files:
        print(f"Warning: No matching files found in {LIBRARY_DIR}")
        return actions

    print(f"Found {len(document_files)} documents to index.")

    for file_path in document_files:
        filename = file_path.name
        print(f"\nProcessing: {filename}...")
        
        # Get metadata values or fallback to defaults
        meta = metadata_map.get(filename, {})
        paper_id = meta.get("id") or meta.get("paper_id") or file_path.stem
        title = meta.get("title") or file_path.stem
        authors = meta.get("authors") or []
        category = meta.get("category") or "Uncategorized"
        
        # Extract full document text
        full_text = extract_text(file_path)
        if not full_text.strip():
            print(f"Warning: Skipping {filename}: No content extracted.")
            continue
            
        # Chunk text
        chunks = chunk_text(full_text)
        print(f"Split into {len(chunks)} raw chunks.")
        
        active_idx = 0
        skipped_bibs = 0
        
        for chunk in chunks:
            # Drop bibliography pages so they don't pollute the search space
            if is_bibliography_chunk(chunk):
                skipped_bibs += 1
                continue
                
            chunk_id = f"{paper_id}_chunk_{active_idx}"
            
            # Generate the 1024-dim BGE-M3 embedding
            embedding = embedding_service.get_embeddings(chunk)
            
            # Build the Elasticsearch index command block
            action = {
                "_index": settings.INDEX_NAME,
                "_id": chunk_id,
                "_source": {
                    "paper_id": paper_id,
                    "title": title,
                    "authors": authors,
                    "category": category,
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "vector_embedding": embedding
                }
            }
            actions.append(action)
            active_idx += 1
            
        print(f"Generated embeddings for {active_idx} content chunks. (Filtered out {skipped_bibs} bibliography chunks)")
            
    return actions


def main():
    print("Starting local document ingestion pipeline...")
    
    # Verify we can ping ES before running embeddings
    if not es_service.verify_connection():
        print("Error: Elasticsearch connection failed. Fix your credentials or SSL settings first.")
        return
        
    metadata_map = load_metadata()
    if not metadata_map:
        print("Warning: Proceeding with file-system fallbacks as metadata map is empty.")
        
    actions = build_bulk_actions(metadata_map)
    
    if not actions:
        print("Error: No chunks prepared for indexing.")
        return
        
    # We use chunk_size=100 to break the documents into smaller, 
    # highly manageable batches. This prevents JVM heap exhaustion.
    print(f"\nBulk indexing {len(actions)} total chunks into Elasticsearch in batches of 100...")
    
    try:
        success, failed = bulk(
            es_service.client, 
            actions, 
            chunk_size=100,  # Send 100 docs per request
            stats_only=True,
            request_timeout=60  # Give Elasticsearch up to 60 seconds to process each batch
        )
        
        # Refresh index so new docs are searchable immediately
        es_service.refresh_index()
        
        print(f"\nIngestion complete!")
        print(f"Successfully indexed: {success} chunks.")
        print(f"Failed chunks: {failed}")
        
    except Exception as e:
        print(f"\nError during indexing: {e}")
        print("If you still get a Connection Reset, check your Docker memory allocation.")


if __name__ == "__main__":
    main()