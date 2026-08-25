"""
Chunking strategies with word-span tracking using LangChain text splitters.

The notes corpus is parsed once into a normalized word stream per document.
Every word has a stable index, and every chunk remembers the word range it
covers (span_start, span_end). The dataset builder and the indexer share this
same coordinate system, so an evaluation can judge relevance by index overlap
between a returned chunk and a gold-answer paragraph instead of by chunk ids.

The strategies:
  * fixed      - sliding word window with overlap (LangChain CharacterTextSplitter)
  * recursive  - packs sentences/paragraphs up to target budget using
                 RecursiveCharacterTextSplitter
  * semantic   - uses sentence embeddings to detect topic shifts
  * document   - respects section boundaries while splitting
  * hierarchical - coarse-to-fine hierarchy via LlamaIndex's HierarchicalNodeParser

All chunk boundaries are determined by LangChain splitters, but the resulting
text chunks are mapped back to word indices using character offsets.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

from app.services.page_cleaner import clean_markdown

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

_PAGE_RE = re.compile(r"^\[\[PAGE (\d+)\]\]$")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$")
_MD_LINK_TOKEN_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")
_MD_EMPH_RE = re.compile(r"[*_~]+")
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

CHUNK_STRATEGIES = (
    "fixed",
    "recursive",
    "semantic",
    "document",
    "hierarchical",
)

STRATEGY_MAP = {
    "sentence": "recursive",
    "paragraph": "document",
}

@dataclass
class Paragraph:
    start: int
    end: int

@dataclass
class Section:
    heading: str
    paragraphs: list[Paragraph]

@dataclass
class ParsedDoc:
    words: list[str]
    sections: list[Section]
    word_pages: list[int] = field(default_factory=list)

@dataclass
class Chunk:
    span_start: int
    span_end: int
    text: str
    heading: str = ""
    page_start: int | None = None
    page_end: int | None = None

def span_overlap_fraction(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Fraction of the gold range [b_start, b_end) covered by [a_start, a_end)."""
    length = b_end - b_start
    if length <= 0:
        return 0.0
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    return intersection / length

def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _PAGE_RE.match(stripped):
        return True
    remainder = _MD_LINK_TOKEN_RE.sub("", line)
    remainder = _MD_EMPH_RE.sub("", remainder)
    return not remainder.strip()

def parse_document(text: str) -> ParsedDoc:
    text = clean_markdown(text)
    words: list[str] = []
    word_pages: list[int] = []
    sections: list[Section] = []
    current_page = 1

    def current_section() -> Section:
        if not sections:
            sections.append(Section("Introduction", []))
        return sections[-1]

    def flush_paragraph(para_start: int, para_end: int) -> None:
        if para_start >= 0 and para_end > para_start:
            current_section().paragraphs.append(Paragraph(para_start, para_end))

    para_start = -1
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if para_start >= 0:
                flush_paragraph(para_start, len(words))
                para_start = -1
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            if para_start >= 0:
                flush_paragraph(para_start, len(words))
                para_start = -1
            sections.append(Section(_MD_EMPH_RE.sub("", heading.group(1)).strip(), []))
            continue

        page_match = _PAGE_RE.match(stripped)
        if page_match:
            if para_start >= 0:
                flush_paragraph(para_start, len(words))
                para_start = -1
            current_page = int(page_match.group(1))
            continue

        if _is_noise_line(line):
            if para_start >= 0:
                flush_paragraph(para_start, len(words))
                para_start = -1
            continue

        if para_start < 0:
            para_start = len(words)
        tokens = [tok for tok in _MD_EMPH_RE.sub("", line).split() if tok]
        words.extend(tokens)
        word_pages.extend([current_page] * len(tokens))

    if para_start >= 0:
        flush_paragraph(para_start, len(words))

    return ParsedDoc(words=words, sections=[s for s in sections if s.paragraphs], word_pages=word_pages)

def merge_text(words: list[str], start: int, end: int, heading: str) -> str:
    body = " ".join(words[start:end]).strip()
    if heading and body:
        return f"# {heading}\n\n{body}"
    return body

def _word_offset_map(doc: ParsedDoc):
    """Return (full_text, word_starts, word_ends) for the normalized word stream."""
    full_text = " ".join(doc.words)
    word_starts = []
    pos = 0
    for w in doc.words:
        word_starts.append(pos)
        pos += len(w) + 1
    word_ends = [s + len(w) for s, w in zip(word_starts, doc.words)]
    return full_text, word_starts, word_ends

def _map_text_to_span(full_text: str, word_starts: list[int], word_ends: list[int],
                      chunk_text: str, search_start: int = 0):
    """Locate a chunk of text in full_text and convert to word indices.

    Returns (start_word, end_word) or None if the chunk cannot be found.
    """
    start_char = full_text.find(chunk_text, search_start)
    if start_char == -1:
        start_char = full_text.find(chunk_text)
    if start_char == -1:
        return None

    end_char = start_char + len(chunk_text)

    start_word = bisect_left(word_ends, start_char)
    end_word = bisect_right(word_starts, end_char)
    if end_word <= start_word:
        return None
    return start_word, end_word

def _attach_heading_and_pages(doc: ParsedDoc, chunk: Chunk) -> Chunk:
    """Set heading and page numbers based on the chunk's word span."""

    for section in doc.sections:

        if section.paragraphs and section.paragraphs[0].start <= chunk.span_start < section.paragraphs[-1].end:
            chunk.heading = section.heading
            break

    chunk.text = merge_text(doc.words, chunk.span_start, chunk.span_end, chunk.heading)
    if doc.word_pages and chunk.span_end > chunk.span_start:
        chunk.page_start = doc.word_pages[chunk.span_start]
        chunk.page_end = doc.word_pages[chunk.span_end - 1]
    return chunk

def _chunk_fixed(doc: ParsedDoc, target: int, overlap: int) -> list[Chunk]:
    """Fixed‑size word window with overlap, using LangChain's CharacterTextSplitter."""
    full_text, word_starts, word_ends = _word_offset_map(doc)
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=target,
        chunk_overlap=overlap,
        length_function=lambda s: len(s.split()),
    )
    text_chunks = splitter.split_text(full_text)

    chunks: list[Chunk] = []
    search_start = 0
    for text_chunk in text_chunks:
        span = _map_text_to_span(full_text, word_starts, word_ends, text_chunk, search_start)
        if span:
            start, end = span
            chunk = Chunk(start, end, merge_text(doc.words, start, end, ""), "")
            chunk = _attach_heading_and_pages(doc, chunk)
            chunks.append(chunk)
            search_start = full_text.find(text_chunk, search_start) + len(text_chunk)
    return chunks

def _chunk_recursive(doc: ParsedDoc, target: int, overlap: int) -> list[Chunk]:
    """Recursive chunking using LangChain's RecursiveCharacterTextSplitter."""
    full_text, word_starts, word_ends = _word_offset_map(doc)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=target,
        chunk_overlap=overlap,
        length_function=lambda s: len(s.split()),
        separators=[" ", ""],
    )
    text_chunks = splitter.split_text(full_text)

    chunks: list[Chunk] = []
    search_start = 0
    for text_chunk in text_chunks:
        span = _map_text_to_span(full_text, word_starts, word_ends, text_chunk, search_start)
        if span:
            start, end = span
            chunk = Chunk(start, end, merge_text(doc.words, start, end, ""), "")
            chunk = _attach_heading_and_pages(doc, chunk)
            chunks.append(chunk)
            search_start = full_text.find(text_chunk, search_start) + len(text_chunk)
    return chunks

def _chunk_semantic(doc: ParsedDoc, target: int, overlap: int) -> list[Chunk]:
    """Semantic chunking using LangChain's SemanticChunker with a local embedding model."""
    full_text, word_starts, word_ends = _word_offset_map(doc)
    try:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"local_files_only": True, "trust_remote_code": False},
                encode_kwargs={"normalize_embeddings": True},
            )
        except TypeError:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
        )
        text_chunks = splitter.split_text(full_text)
    except Exception as e:
        msg = str(e)
        if "local_files_only" in msg or "We couldn't connect" in msg or "couldn't find" in msg:
            try:
                if "HF_HUB_OFFLINE" in os.environ:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
                text_chunks = splitter.split_text(full_text)
            except Exception:
                return _chunk_fixed(doc, target, overlap)
        else:
            return _chunk_fixed(doc, target, overlap)

    chunks: list[Chunk] = []
    search_start = 0
    for text_chunk in text_chunks:
        span = _map_text_to_span(full_text, word_starts, word_ends, text_chunk, search_start)
        if span:
            start, end = span
            chunk = Chunk(start, end, merge_text(doc.words, start, end, ""), "")
            chunk = _attach_heading_and_pages(doc, chunk)
            chunks.append(chunk)
            search_start = full_text.find(text_chunk, search_start) + len(text_chunk)
    return chunks

def _chunk_document_based(doc: ParsedDoc, target: int, overlap: int) -> list[Chunk]:
    """Document‑aware chunking: split each section independently using a recursive splitter."""
    chunks: list[Chunk] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=target,
        chunk_overlap=overlap,
        length_function=lambda s: len(s.split()),
        separators=[" ", ""],
    )

    full_text, word_starts, word_ends = _word_offset_map(doc)

    for section in doc.sections:

        sec_start = section.paragraphs[0].start
        sec_end = section.paragraphs[-1].end
        sec_text = " ".join(doc.words[sec_start:sec_end])

        global_char_start = word_starts[sec_start]

        text_chunks = splitter.split_text(sec_text)
        search_start = 0
        for text_chunk in text_chunks:

            local_start = sec_text.find(text_chunk, search_start)
            if local_start == -1:
                local_start = sec_text.find(text_chunk)
            if local_start == -1:
                continue
            local_end = local_start + len(text_chunk)

            global_start_char = global_char_start + local_start
            global_end_char = global_char_start + local_end

            start_word = bisect_left(word_ends, global_start_char)
            end_word = bisect_right(word_starts, global_end_char)
            if end_word <= start_word:
                continue

            chunk = Chunk(start_word, end_word, merge_text(doc.words, start_word, end_word, section.heading),
                          section.heading)
            chunk = _attach_heading_and_pages(doc, chunk)
            chunks.append(chunk)
            search_start = local_end

    return chunks

def _chunk_hierarchical(doc: ParsedDoc, target: int, overlap: int) -> list[Chunk]:
    """Hierarchical chunking using LlamaIndex's HierarchicalNodeParser.

    Produces a coarse-to-fine hierarchy: parent chunks at 2x the target size
    and child chunks at the target size, linked by parent/child relationships.
    All levels are returned so the index can answer with either broad context
    or precise detail."""
    from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter
    from llama_index.core.schema import Document

    word_tokenizer = lambda s: s.split()
    parent_size = max(target * 2, target + 1)

    parser = HierarchicalNodeParser.from_defaults(
        chunk_overlap=overlap,
        node_parser_ids=["parent", "child"],
        node_parser_map={
            "parent": SentenceSplitter(
                chunk_size=parent_size,
                chunk_overlap=overlap,
                tokenizer=word_tokenizer,
                include_metadata=False,
            ),
            "child": SentenceSplitter(
                chunk_size=target,
                chunk_overlap=overlap,
                tokenizer=word_tokenizer,
                include_metadata=False,
            ),
        },
    )

    chunks: list[Chunk] = []
    seen: set[tuple[int, int]] = set()
    for section in doc.sections:
        sec_start = section.paragraphs[0].start
        sec_end = section.paragraphs[-1].end
        sec_words = doc.words[sec_start:sec_end]

        para_char_starts: list[int] = []
        para_word_starts: list[int] = []
        para_word_char_starts: list[list[int]] = []
        parts: list[str] = []
        char_cursor = 0
        for p in section.paragraphs:
            para_char_starts.append(char_cursor)
            para_word_starts.append(p.start)
            para_words = doc.words[p.start:p.end]
            part = " ".join(para_words)
            parts.append(part)
            word_char_starts: list[int] = []
            offset = 0
            for w in para_words:
                word_char_starts.append(offset)
                offset += len(w) + 1
            para_word_char_starts.append(word_char_starts)
            char_cursor += len(part) + 2
        sec_text = "\n\n".join(parts)

        def local_char_to_word(char_pos: int) -> int:
            """Map a char position in sec_text to the global word index."""
            para_idx = bisect_right(para_char_starts, char_pos) - 1
            if para_idx < 0:
                para_idx = 0
            local = char_pos - para_char_starts[para_idx]
            starts = para_word_char_starts[para_idx]
            word_in_para = bisect_right(starts, local) - 1
            return para_word_starts[para_idx] + max(0, word_in_para)

        nodes = parser.get_nodes_from_documents([Document(text=sec_text)])
        for node in nodes:
            local_start = sec_text.find(node.text)
            if local_start == -1:
                continue
            local_end = local_start + len(node.text)
            start_word = local_char_to_word(local_start)
            end_word = local_char_to_word(local_end - 1) + 1
            if end_word <= start_word or (start_word, end_word) in seen:
                continue
            seen.add((start_word, end_word))

            chunk = Chunk(start_word, end_word, "", section.heading)
            chunk = _attach_heading_and_pages(doc, chunk)
            chunks.append(chunk)

    return chunks

def chunk_document(
    doc: ParsedDoc,
    strategy: str,
    target: int = 300,
    overlap: int = 45,
) -> list[Chunk]:
    """Chunk a parsed document using the specified strategy.

    All strategies use LangChain text splitters internally.
    """
    if target <= 0:
        raise ValueError("target must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    strategy = STRATEGY_MAP.get(strategy, strategy)

    overlap = min(overlap, max(0, target - 1))
    if strategy not in CHUNK_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {CHUNK_STRATEGIES}")

    if strategy == "fixed":
        chunks = _chunk_fixed(doc, target, overlap)
    elif strategy == "recursive":
        chunks = _chunk_recursive(doc, target, overlap)
    elif strategy == "semantic":
        chunks = _chunk_semantic(doc, target, overlap)
    elif strategy == "document":
        chunks = _chunk_document_based(doc, target, overlap)
    elif strategy == "hierarchical":
        chunks = _chunk_hierarchical(doc, target, overlap)
    else:

        chunks = _chunk_fixed(doc, target, overlap)

    for chunk in chunks:
        if doc.word_pages and chunk.span_end > chunk.span_start:
            chunk.page_start = doc.word_pages[chunk.span_start]
            chunk.page_end = doc.word_pages[chunk.span_end - 1]
    return chunks