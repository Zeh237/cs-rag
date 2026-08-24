import hashlib
import os
import re
from pathlib import Path

import fitz

from app.services.highlight import compute_highlight_boxes
from app.services.page_cleaner import clean_markdown, clean_page_texts

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
        page_texts = [(page_num, page.get_text("text")) for page_num, page in enumerate(doc, start=1)]
        doc.close()
        page_texts = clean_page_texts(page_texts)

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
                    "page_start": chunk_pages[0] if chunk_pages else 1,
                    "page_end": chunk_pages[-1] if chunk_pages else 1,
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

    async def chunk_pdf_vision(self, file_path: str, chunk_size: int = 1000, overlap: int = 200, use_cached: bool = True) -> list[dict]:
        import base64
        from app.services.llm import llm_service
        from app.core.config import settings

        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        if use_cached:
            cached = self._load_vision_markdown_cache(file_path)
            if cached is not None:
                page_texts = cached
                print(f"Loaded {len(page_texts)} pages from vision markdown cache")
            else:
                page_texts = await self._transcribe_pdf_pages(file_path)
                self._save_vision_markdown_cache(file_path, page_texts)
        else:
            page_texts = await self._transcribe_pdf_pages(file_path)
            self._save_vision_markdown_cache(file_path, page_texts)

        page_texts = clean_page_texts(page_texts)

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
                    "page_start": chunk_pages[0] if chunk_pages else 1,
                    "page_end": chunk_pages[-1] if chunk_pages else 1,
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

    def _vision_cache_path(self, file_path: str) -> Path:
        pdf_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()[:12]
        cache_dir = Path(os.environ.get("NOTES_DIR", ".")) / "vision_markdown"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{Path(file_path).stem}_{pdf_hash}.md"

    def _save_vision_markdown_cache(self, file_path: str, page_texts: list[tuple[int, str]]) -> None:
        cache_path = self._vision_cache_path(file_path)
        md_content = "\n\n".join(f"[[PAGE {p}]]\n{t}" for p, t in page_texts if t and t.strip() != "[SKIP]")
        cache_path.write_text(md_content, encoding="utf-8")
        print(f"Saved vision markdown cache to {cache_path}")

    def _load_vision_markdown_cache(self, file_path: str) -> list[tuple[int, str]] | None:
        cache_path = self._vision_cache_path(file_path)
        if not cache_path.exists():
            return None
        try:
            content = cache_path.read_text(encoding="utf-8")
            page_texts = []
            for block in content.split("[[PAGE "):
                if not block.strip():
                    continue
                page_match = re.match(r"(\d+)\]\]\n(.*)", block, re.DOTALL)
                if page_match:
                    page_num = int(page_match.group(1))
                    text = page_match.group(2).strip()
                    if text:
                        page_texts.append((page_num, text))
            return page_texts if page_texts else None
        except Exception:
            return None

    async def _transcribe_pdf_pages(self, file_path: str) -> list[tuple[int, str]]:
        import base64
        from app.services.llm import llm_service
        from app.core.config import settings

        doc = fitz.open(file_path)
        page_texts = []
        n_pages = doc.page_count

        for page_num in range(1, n_pages + 1):
            page = doc[page_num - 1]
            try:
                pix = page.get_pixmap(dpi=settings.RESEARCH_DPI)
                image_data = "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode()
                text = await llm_service.transcribe_page_image(image_data)
            except Exception as e:
                print(f"Vision transcription failed for page {page_num}: {e}. Falling back to standard text extraction.")
                text = page.get_text("text")
            if text.strip() == "[SKIP]":
                print(f"Page {page_num} marked as skip by vision model")
                continue
            page_texts.append((page_num, text))

        doc.close()
        return page_texts

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
        text = clean_markdown(text)
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
                pages = sorted({page for _, page in token_chunk})
                page_start = pages[0] if pages else 1
                body = " ".join(words)
                text_out = f"# {heading}\n\n{body}".strip() if heading_words else body
                chunks.append(
                    {
                        "text": text_out,
                        "page": page_start,
                        "page_start": page_start,
                        "page_end": pages[-1] if pages else page_start,
                        "pages": pages,
                    }
                )
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
