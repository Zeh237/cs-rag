from __future__ import annotations

import difflib
import unicodedata
from collections import defaultdict

import fitz


def normalize_key(word: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", word).lower() if ch.isalnum())


def tokenize(text: str) -> list[str]:
    return [key for tok in text.split() if (key := normalize_key(tok))]


def align_tokens(chunk_keys: list[str], page_keys: list[str]) -> list[int]:
    if not chunk_keys or not page_keys:
        return []
    matcher = difflib.SequenceMatcher(None, chunk_keys, page_keys, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size >= 4]
    matched = []
    for block in blocks:
        matched.extend(range(block.b, block.b + block.size))
    return matched


def merge_rects(rects) -> list[list[float]]:
    spans = []
    for rect in sorted(rects):
        x0, y0, x1, y1 = (float(v) for v in rect)
        placed = False
        for span in spans:
            sx0, sy0, sx1, sy1 = span
            height = min(y1 - y0, sy1 - sy0)
            overlap = min(y1, sy1) - max(y0, sy0)
            gap = sx1 - x0
            if height > 0 and overlap > 0.5 * height and gap <= max(2.0, 0.6 * height):
                span[0] = min(sx0, x0)
                span[1] = min(sy0, y0)
                span[2] = max(sx1, x1)
                span[3] = max(sy1, y1)
                placed = True
                break
        if not placed:
            spans.append([x0, y0, x1, y1])
    return spans


def compute_highlight_boxes(doc: fitz.Document, pages, chunk_text: str) -> dict:
    chunk_keys = tokenize(chunk_text)
    if not chunk_keys:
        return {}

    all_keys = []
    all_rects = []
    all_pages = []
    for page_num in pages:
        for word in doc[page_num - 1].get_text("words"):
            key = normalize_key(word[4])
            if key:
                all_keys.append(key)
                all_rects.append(tuple(word[:4]))
                all_pages.append(page_num)

    matched = align_tokens(chunk_keys, all_keys)
    by_page = defaultdict(list)
    for i in matched:
        by_page[all_pages[i]].append(all_rects[i])

    return {str(page): merge_rects(rects) for page, rects in by_page.items()}
