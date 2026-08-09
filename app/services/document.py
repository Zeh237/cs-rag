import os
import re

import fitz

from app.services.highlight import compute_highlight_boxes

_BIBLIOGRAPHY_PATTERNS = [
    r"(?i)(?:\n|^|\s)References\b",
    r"(?i)(?:\n|^|\s)Bibliography\b",
    r"(?i)(?:\n|^|\s)Works Cited\b",
]


class DocumentProcessor:
    def chunk_pdf(self, file_path: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        doc = fitz.open(file_path)
        page_texts = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                page_texts.append((page_num, text))
        doc.close()

        if not page_texts:
            return []

        word_pages = [(word, page_num) for page_num, text in page_texts for word in text.split()]
        word_pages = word_pages[: self._bibliography_cutoff(word_pages)]
        if not word_pages:
            return []

        chunks = []
        i = 0
        while i < len(word_pages):
            chunk_slice = word_pages[i : i + chunk_size]
            chunk_pages = sorted({page for _, page in chunk_slice})
            chunks.append(
                {
                    "text": " ".join(word for word, _ in chunk_slice),
                    "page": chunk_slice[0][1],
                    "pages": chunk_pages,
                }
            )
            if i + chunk_size >= len(word_pages):
                break
            i += chunk_size - overlap

        doc = fitz.open(file_path)
        for chunk in chunks:
            chunk["highlight_boxes"] = compute_highlight_boxes(doc, chunk.pop("pages"), chunk["text"])
        doc.close()

        return chunks

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunks.append({"text": " ".join(words[i : i + chunk_size]), "page": 1})
            if i + chunk_size >= len(words):
                break
            i += chunk_size - overlap
        return chunks

    def chunk_markdown(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
        lines = text.splitlines()
        sections: list[tuple[str, list[tuple[str, int]]]] = []
        current_heading = "Introduction"
        current_tokens: list[tuple[str, int]] = []
        current_page = 1

        def flush() -> None:
            nonlocal current_tokens
            if current_tokens:
                sections.append((current_heading, current_tokens))
            current_tokens = []

        for line in lines:
            page_match = re.match(r"^\[\[PAGE (\d+)\]\]$", line.strip())
            if page_match:
                current_page = int(page_match.group(1))
                continue
            if re.match(r"^#{1,6}\s+\S", line):
                flush()
                current_heading = line.strip("# ").strip() or current_heading
            else:
                current_tokens.extend((word, current_page) for word in line.split())
        flush()

        chunks = []
        for heading, tokens in sections:
            heading_words = len(heading.split())
            budget = chunk_size - heading_words if heading_words < chunk_size else chunk_size
            for token_chunk in self._split_word_tokens(tokens, budget, overlap):
                words = [word for word, _ in token_chunk]
                page = token_chunk[0][1] if token_chunk else 1
                body = " ".join(words)
                text_out = f"# {heading}\n\n{body}".strip() if heading_words else body
                chunks.append({"text": text_out, "page": page})
        return chunks

    def _split_word_tokens(self, tokens: list[tuple[str, int]], chunk_size: int, overlap: int) -> list[list[tuple[str, int]]]:
        if len(tokens) <= chunk_size:
            return [tokens]
        chunks = []
        i = 0
        while i < len(tokens):
            chunks.append(tokens[i : i + chunk_size])
            if i + chunk_size >= len(tokens):
                break
            i += chunk_size - overlap
        return chunks

    def _bibliography_cutoff(self, word_pages: list[tuple[str, int]]) -> int:
        combined = " ".join(word for word, _ in word_pages)
        cutoff = len(word_pages)
        for pattern in _BIBLIOGRAPHY_PATTERNS:
            match = re.search(pattern, combined)
            if match:
                word_idx = len(combined[: match.start()].split())
                cutoff = min(cutoff, word_idx)
        return cutoff


document_processor = DocumentProcessor()
