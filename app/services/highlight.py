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
    blocks = [b for b in matcher.get_matching_blocks() if b.size >= 2]
    if not blocks:
        return []

    # Group matching blocks into contiguous or near-contiguous clusters in page_keys
    clusters: list[list] = []
    max_gap = max(25, int(len(chunk_keys) * 0.4))

    current_cluster: list = []
    for b in sorted(blocks, key=lambda x: x.b):
        if not current_cluster:
            current_cluster.append(b)
        else:
            prev = current_cluster[-1]
            gap = b.b - (prev.b + prev.size)
            if gap <= max_gap:
                current_cluster.append(b)
            else:
                clusters.append(current_cluster)
                current_cluster = [b]
    if current_cluster:
        clusters.append(current_cluster)

    # Pick the cluster with the maximum total matching tokens
    best_cluster = max(clusters, key=lambda cl: sum(b.size for b in cl))

    matched = []
    for block in best_cluster:
        matched.extend(range(block.b, block.b + block.size))
    return matched


def merge_rects(rects) -> list[list[float]]:
    if not rects:
        return []

    float_rects = [[float(v) for v in r] for r in rects]
    # Filter out invalid / zero area rects
    valid_rects = [r for r in float_rects if r[2] > r[0] and r[3] > r[1]]
    if not valid_rects:
        return []

    # Sort by top y coordinate, then left x coordinate
    valid_rects.sort(key=lambda r: (r[1], r[0]))

    lines: list[list[list[float]]] = []
    for rect in valid_rects:
        x0, y0, x1, y1 = rect
        height = y1 - y0
        placed = False
        for line in lines:
            ly0 = min(r[1] for r in line)
            ly1 = max(r[3] for r in line)
            l_height = ly1 - ly0
            overlap = min(y1, ly1) - max(y0, ly0)
            if overlap > 0.4 * min(height, l_height) or abs(y0 - ly0) < 0.5 * height:
                line.append(rect)
                placed = True
                break
        if not placed:
            lines.append([rect])

    spans = []
    for line in lines:
        lx0 = min(r[0] for r in line)
        ly0 = min(r[1] for r in line)
        lx1 = max(r[2] for r in line)
        ly1 = max(r[3] for r in line)
        spans.append([round(lx0, 2), round(ly0, 2), round(lx1, 2), round(ly1, 2)])

    return spans


def compute_highlight_boxes(doc: fitz.Document, pages, chunk_text: str) -> dict:
    chunk_keys = tokenize(chunk_text)
    if not chunk_keys:
        return {}

    all_keys = []
    all_rects = []
    all_pages = []
    for page_num in pages:
        if page_num < 1 or page_num > len(doc):
            continue
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

    return {str(page): merge_rects(rects) for page, rects in by_page.items() if rects}

