import os
import json
import re
from pathlib import Path
import fitz  # PyMuPDF
import pymupdf4llm
from config import settings

def strip_bibliography(markdown_text: str) -> str:
    """
    Finds common academic bibliography headers and truncates the text
    to drop everything after them, avoiding reference indexing noise.
    """
    bibliography_patterns = [
        r"(?i)(?:\n|^)#+\s+References\b",
        r"(?i)(?:\n|^)#+\s+Bibliography\b",
        r"(?i)(?:\n|^)#+\s+Works Cited\b"
    ]
    
    for pattern in bibliography_patterns:
        match = re.search(pattern, markdown_text)
        if match:
            return markdown_text[:match.start()].strip()
            
    return markdown_text


def chunk_pdf(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """
    Extracts text from a PDF page-by-page using PyMuPDF, strips the bibliography,
    and splits it into overlapping semantic chunks. Each chunk dict contains:
      - 'text': the chunk text
      - 'page': the 1-indexed page number where the chunk starts
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Extract text per page using fitz for page-number tracking
    doc = fitz.open(file_path)
    page_texts = []  # list of (page_number, text)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            page_texts.append((page_num, text))
    doc.close()

    if not page_texts:
        return []

    # 2. Strip bibliography — work on combined text with page markers
    #    Build a list of words annotated with their source page
    word_pages = []  # list of (word, page_number)
    for page_num, text in page_texts:
        for word in text.split():
            word_pages.append((word, page_num))

    # Find bibliography boundary using the combined text
    combined_text = " ".join(w for w, _ in word_pages)
    bib_patterns = [
        r"(?i)(?:\n|^|\s)References\b",
        r"(?i)(?:\n|^|\s)Bibliography\b",
        r"(?i)(?:\n|^|\s)Works Cited\b"
    ]
    cutoff_word_idx = len(word_pages)
    for pattern in bib_patterns:
        match = re.search(pattern, combined_text)
        if match:
            # Approximate word index from character position
            char_pos = match.start()
            word_idx = len(combined_text[:char_pos].split())
            if word_idx < cutoff_word_idx:
                cutoff_word_idx = word_idx

    word_pages = word_pages[:cutoff_word_idx]

    if not word_pages:
        return []

    # 3. Create overlapping sliding-window chunks, preserving page number
    chunks = []
    i = 0
    while i < len(word_pages):
        chunk_slice = word_pages[i : i + chunk_size]
        chunk_text = " ".join(w for w, _ in chunk_slice)
        # Page number = page of the first word in the chunk
        chunk_page = chunk_slice[0][1]
        chunks.append({"text": chunk_text, "page": chunk_page})
        if i + chunk_size >= len(word_pages):
            break
        i += (chunk_size - overlap)

    return chunks


def append_to_manifest(metadata: dict):
    """Appends metadata of a newly ingested paper to the central JSON manifest."""
    os.makedirs(settings.LIBRARY_DIR, exist_ok=True)
    
    manifest = []
    if os.path.exists(settings.MANIFEST_PATH):
        try:
            with open(settings.MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except json.JSONDecodeError:
            manifest = []

    # Keep only unique papers in the manifest
    if not any(item["id"] == metadata["id"] for item in manifest):
        manifest.append(metadata)
        with open(settings.MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)