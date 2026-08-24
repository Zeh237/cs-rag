"""Page-level cleaning for ingested PDF text.

The raw `page.get_text("text")` output (and the markdown files produced from
it) carries a lot of boilerplate that pollutes retrieval:

  * Running headers / footers repeated on every page (e.g. "§14.2" /
    "External Memory Searching") or garbled (ligature-stripped headers).
  * Standalone page-number footer lines ("277", "308", ...).
  * TOC / title / copyright front matter and "Contents" spreads.
  * Back matter: "References", "Bibliography", "Works Cited", "Index".

The cleaners below work on page-shaped input (either a list of raw page texts
or markdown containing `[[PAGE N]]` markers) and apply three heuristics:

  1. Frequency-based repeat detection on the top/bottom few lines of each page
     catches running headers/footers regardless of content, plus explicit page
     numbers and dashed separators.
  2. A leading run of "short/boilerplate" pages (TOC, title, copyright) is
     dropped, so we never cut the first real content page.
  3. A trailing back-matter section is truncated at its first heading page.

The corpus markdown and the raw-PDF ingest path share exactly these rules, so
chunk spans stay aligned no matter which path produced the text.
"""

from __future__ import annotations

import re

_PAGE_MARKER_RE = re.compile(r"^\[\[PAGE (\d+)\]\]$")
_PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
_SECTION_NUM_RE = re.compile(r"^\d+(\.\d+){0,3}$")
_DOTTED_LINE_RE = re.compile(r"(?:[.\u00b7]\s*){3,}")
_DASH_RE = re.compile(r"^[-–—_·.\s]{3,}$")

_BAND = 3
_WINDOW = 8
_MIN_RUN = 3
_LONG_LINE = 45
_MIN_STRUCT_PAGES = 12
_BACK_MATTER_LEADS = {
    "references",
    "bibliography",
    "works cited",
    "index",
    "glossary",
}
_COPY_HINTS = ("copyright", "©", "isbn", "all rights reserved")

def _nonempty_stripped(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip()]

def _front_matter_noise(lines: list[str]) -> bool:
    """True for TOC / title / copyright style pages with no real paragraphs."""
    if not lines:
        return True
    lower = " ".join(lines).lower()
    if any(hint in lower for hint in _COPY_HINTS):
        return True

    tocish = sum(
        1
        for line in lines
        if _DOTTED_LINE_RE.search(line) or _SECTION_NUM_RE.match(line) or _PAGE_NUM_RE.match(line)
    )

    real = sum(1 for line in lines if len(line) > _LONG_LINE and not _DOTTED_LINE_RE.search(line))
    if tocish >= 3 and real <= 2:
        return True

    return real == 0 and len(lines) >= 2

def _back_matter_lead(lines: list[str]) -> bool:
    if not lines:
        return False
    first = lines[0].lower().strip().strip(":.")
    return first in _BACK_MATTER_LEADS

def _running_margins(pages: list[list[str]]) -> tuple[set[str], set[str]]:
    """Running headers/footers are section-scoped, so detect them locally.

    A top-band line is treated as a running header when it occupies the top
    margin of a few pages within the trailing window --- not over the whole
    document. Same for the bottom margin. Book layout alternates headers
    between facing pages, so require only a majority of the window.
    """
    n = len(pages)
    top_runs: set[str] = set()
    bottom_runs: set[str] = set()
    for i in range(n):
        top = set(pages[i][:_BAND])
        bottom = set(pages[i][-_BAND:])
        window_start = max(0, i - _WINDOW + 1)
        for line in top:
            matches = sum(1 for j in range(window_start, i + 1) if line in set(pages[j][:_BAND]))
            if matches >= _MIN_RUN:
                top_runs.add(line)
        for line in bottom:
            matches = sum(1 for j in range(window_start, i + 1) if line in set(pages[j][-_BAND:]))
            if matches >= _MIN_RUN:
                bottom_runs.add(line)
    return top_runs, bottom_runs

def _clean_pages(pages: list[list[str]]) -> list[list[str]]:
    """Return one (possibly empty) list of kept lines per input page."""
    pages = [_nonempty_stripped(p) for p in pages]
    n = len(pages)
    if not n:
        return pages

    first_half = max(1, int(n * 0.5))
    top_runs, bottom_runs = _running_margins(pages)

    drop: set[int] = set()
    content_start = 0
    if n >= _MIN_STRUCT_PAGES:

        i = 0
        while i < n and i <= first_half and _front_matter_noise(pages[i]):
            drop.add(i)
            i += 1
        content_start = i

        for j in range(content_start, n):
            if _back_matter_lead(pages[j]):
                drop.update(range(j, n))
                break
    elif all(_front_matter_noise(page) for page in pages):
        drop.update(range(n))

    cleaned: list[list[str]] = []
    for i, lines in enumerate(pages):
        if i in drop:
            cleaned.append([])
            continue
        nlines = len(lines)
        drop_idx: set[int] = set()
        for idx, line in enumerate(lines):
            in_top = idx < _BAND
            in_bottom = idx >= nlines - _BAND
            if in_top and line in top_runs:
                drop_idx.add(idx)
                continue
            if in_bottom and line in bottom_runs:
                drop_idx.add(idx)
                continue
            if (in_top or in_bottom) and _PAGE_NUM_RE.match(line):
                drop_idx.add(idx)
                continue
            if _DASH_RE.match(line):
                drop_idx.add(idx)
        kept = [line for idx, line in enumerate(lines) if idx not in drop_idx]
        cleaned.append(kept)
    return cleaned

def clean_page_texts(page_texts: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Clean a freshly extracted list of (page_number, raw_text) tuples."""
    pages = [text.splitlines() for _, text in page_texts]
    cleaned = _clean_pages(pages)
    return [
        (page_num, "\n".join(lines))
        for (page_num, _), lines in zip(page_texts, cleaned)
        if lines
    ]

def clean_markdown(text: str) -> str:
    """Clean markdown containing [[PAGE N]] markers; returns cleaned markdown."""
    if "[[PAGE" not in text:
        return text

    header_lines: list[str] = []
    page_blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if _PAGE_MARKER_RE.match(line.strip()):
            if current is not None:
                page_blocks.append(current)
            current = [line]
        elif current is None:
            header_lines.append(line)
        else:
            current.append(line)
    if current is not None:
        page_blocks.append(current)

    pages = [block[1:] for block in page_blocks]
    cleaned = _clean_pages(pages)

    out_lines = list(header_lines)
    for marker, kept in zip(page_blocks, cleaned):
        out_lines.append(marker[0])
        if kept:
            out_lines.extend(kept)
        out_lines.append("")
    return "\n".join(out_lines)