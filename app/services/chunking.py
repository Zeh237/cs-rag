"""Chunking strategies with word-span tracking for the chunking study.

The notes corpus is parsed once into a normalized word stream per document.
Every word has a stable index, and every chunk remembers the word range it
covers (span_start, span_end). The dataset builder and the indexer share this
same coordinate system, so an evaluation can judge relevance by index overlap
between a returned chunk and a gold-answer paragraph instead of by chunk ids.

The strategies:
  * fixed     - sliding word window with overlap (like the production chunker,
                but head-bound). Windows never cross a section boundary.
  * sentence  - packs whole sentences into chunks up to the target budget;
                never splits a single sentence (unless it exceeds the budget).
  * paragraph - one or more consecutive paragraphs inside a section become a
                chunk; small paragraphs are merged up to the target budget and
                oversized paragraphs are split at sentence boundaries.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass

_PAGE_RE = re.compile(r"^\[\[PAGE (\d+)\]\]$")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$")
_MD_LINK_TOKEN_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")
_MD_EMPH_RE = re.compile(r"[*_~]+")
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


CHUNK_STRATEGIES = ("fixed", "sentence", "paragraph")


@dataclass
class Paragraph:
    start: int  # word index, inclusive
    end: int  # word index, exclusive


@dataclass
class Section:
    heading: str
    paragraphs: list[Paragraph]


@dataclass
class ParsedDoc:
    words: list[str]
    sections: list[Section]


@dataclass
class Chunk:
    span_start: int
    span_end: int
    text: str
    heading: str = ""


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
    words: list[str] = []
    sections: list[Section] = []

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

        if _is_noise_line(line):
            if para_start >= 0:
                flush_paragraph(para_start, len(words))
                para_start = -1
            continue

        if para_start < 0:
            para_start = len(words)
        tokens = [tok for tok in _MD_EMPH_RE.sub("", line).split() if tok]
        words.extend(tokens)

    if para_start >= 0:
        flush_paragraph(para_start, len(words))

    # Drop empty trailing sections introduced by a heading without body content.
    return ParsedDoc(words=words, sections=[s for s in sections if s.paragraphs])


def merge_text(words: list[str], start: int, end: int, heading: str) -> str:
    body = " ".join(words[start:end]).strip()
    if heading and body:
        return f"# {heading}\n\n{body}"
    return body


def _sentence_ranges(words: list[str]) -> list[tuple[int, int]]:
    if not words:
        return []
    text = " ".join(words)
    offsets = [0]
    for word in words[:-1]:
        offsets.append(offsets[-1] + len(word) + 1)

    bounds = [0]
    for match in _SENT_BOUNDARY.finditer(text):
        bounds.append(match.end())
    if bounds[-1] < len(text):
        bounds.append(len(text))

    ranges: list[tuple[int, int]] = []
    for prev, nxt in zip(bounds, bounds[1:]):
        start = bisect_left(offsets, prev)
        end = bisect_left(offsets, nxt)
        if end > start:
            ranges.append((start, end))
    return ranges


def _split_oversized_range(words: list[str], start: int, end: int, budget: int, overlap: int) -> list[tuple[int, int]]:
    """Fallback: split a very long sentence into word windows."""
    window = max(1, budget)
    step = max(1, window - overlap)
    out = []
    i = start
    while i < end:
        out.append((i, min(i + window, end)))
        if i + window >= end:
            break
        i += step
    return out


def _to_chunks(doc: ParsedDoc, ranges: list[tuple[int, int]]) -> list[Chunk]:
    chunks = []
    for section in doc.sections:
        section_members = set()
        current = 0
        for paragraph in section.paragraphs:
            for idx in range(paragraph.start, paragraph.end):
                section_members.add(idx)
        # rebuild ranges constrained (all already inside the section for all
        # strategies by construction, this is belt-and-braces only)
        for start, end in ranges:
            if start in section_members:
                chunks.append(Chunk(start, end, merge_text(doc.words, start, end, section.heading), section.heading))
    return chunks


def chunk_document(
    doc: ParsedDoc,
    strategy: str,
    target: int = 300,
    overlap: int = 45,
) -> list[Chunk]:
    if target <= 0:
        raise ValueError("target must be positive")
    if strategy not in CHUNK_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {CHUNK_STRATEGIES}")

    if strategy == "fixed":
        return _chunk_fixed(doc, target, overlap)
    if strategy == "paragraph":
        return _chunk_paragraph(doc, target)
    return _chunk_sentence(doc, target, overlap)


def _chunk_fixed(doc: ParsedDoc, target: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = max(1, target - overlap)
    for section in doc.sections:
        body_start = section.paragraphs[0].start
        body_end = section.paragraphs[-1].end
        i = body_start
        while i < body_end:
            end = min(i + target, body_end)
            chunks.append(Chunk(i, end, merge_text(doc.words, i, end, section.heading), section.heading))
            if end >= body_end:
                break
            i += step
    return chunks


def _chunk_sentence(doc: ParsedDoc, target: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in doc.sections:
        body_start = section.paragraphs[0].start
        body_end = section.paragraphs[-1].end
        sentences = [
            (s, e)
            for s, e in _sentence_ranges(doc.words[body_start:body_end])
            if body_start + e <= body_end
        ]
        sentences = [(s + body_start, e + body_start) for s, e in sentences]

        current: list[tuple[int, int]] = []
        current_words = 0
        for s, e in sentences:
            size = e - s
            if size >= target:
                if current:
                    chunks.append(_pack(current, doc, section))
                    current, current_words = [], 0
                for its, ite in _split_oversized_range(doc.words, s, e, target, overlap):
                    chunks.append(Chunk(its, ite, merge_text(doc.words, its, ite, section.heading), section.heading))
                continue
            if current and current_words + size > target:
                chunks.append(_pack(current, doc, section))
                current, current_words = [], 0
            current.append((s, e))
            count_words = sum(e2 - s2 for s2, e2 in current)
            current_words = count_words
        if current:
            chunks.append(_pack(current, doc, section))
    return chunks


def _pack(ranges: list[tuple[int, int]], doc: ParsedDoc, section: Section) -> Chunk:
    start = ranges[0][0]
    end = ranges[-1][1]
    return Chunk(start, end, merge_text(doc.words, start, end, section.heading), section.heading)


def _chunk_paragraph(doc: ParsedDoc, target: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in doc.sections:
        current: list[Paragraph] = []
        current_words = 0
        for paragraph in section.paragraphs:
            size = paragraph.end - paragraph.start
            if size >= target * 2:
                if current:
                    chunks.append(_pack_paragraphs(current, doc, section))
                    current = []
                    current_words = 0
                start, end = paragraph.start, paragraph.end
                for s, e in _split_oversized_range(doc.words, start, end, target, overlap=target // 4):
                    chunks.append(Chunk(s, e, merge_text(doc.words, s, e, section.heading), section.heading))
                continue
            if current and current_words + size > target:
                chunks.append(_pack_paragraphs(current, doc, section))
                current = []
                current_words = 0
            current.append(paragraph)
            current_words += size
        if current:
            chunks.append(_pack_paragraphs(current, doc, section))
    return chunks


def _pack_paragraphs(paragraphs: list[Paragraph], doc: ParsedDoc, section: Section) -> Chunk:
    start = paragraphs[0].start
    end = paragraphs[-1].end
    return Chunk(start, end, merge_text(doc.words, start, end, section.heading), section.heading)