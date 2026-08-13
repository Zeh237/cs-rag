import pytest

from app.services.chunking import (
    chunk_document,
    merge_text,
    parse_document,
    span_overlap_fraction,
)


def _para(sentence: str) -> str:
    return sentence


SAMPLE_DOC = """# 1. Basics

Alpha beta gamma delta.
Epsilon zeta eta theta.

Iota kappa lambda.

## 1.1 Sub

Mu nu xi.

Omicron pi rho sigma tau.

## 1.2 More

Upsilon phi chi psi omega.
"""


def test_parse_document_words_and_sections():
    doc = parse_document(SAMPLE_DOC)
    assert doc.words
    assert len(doc.sections) == 3
    assert doc.sections[0].heading == "1. Basics"
    assert doc.sections[1].heading == "1.1 Sub"
    assert doc.sections[2].heading == "1.2 More"


def test_parse_document_tracks_page_per_word():
    doc = parse_document("[[PAGE 4]]\n\nAlpha beta gamma.\n\n[[PAGE 5]]\n\nDelta epsilon.")
    assert doc.word_pages == [4, 4, 4, 5, 5]


def test_parse_document_pages_default_to_one():
    doc = parse_document("# H\n\nAlpha beta.")
    assert doc.word_pages == [1, 1]


def test_chunks_carry_page_ranges():
    doc = parse_document("[[PAGE 10]]\n\n" + ("alpha " * 45) + "\n\n[[PAGE 11]]\n\n" + ("beta " * 45))
    chunks = chunk_document(doc, strategy="fixed", target=20, overlap=0)
    assert all(c.page_start is not None for c in chunks)
    for c in chunks:
        assert c.page_end >= c.page_start
        assert c.page_start in (10, 11)
        assert c.page_end in (10, 11)
    crossing = next(c for c in chunks if c.page_start == 10 and c.page_end == 11)
    assert crossing is not None


def test_parse_document_paragraph_ranges_are_contiguous():
    doc = parse_document(SAMPLE_DOC)
    for section in doc.sections:
        previous_end = -1
        for paragraph in section.paragraphs:
            assert paragraph.start >= previous_end
            assert paragraph.end > paragraph.start
            assert paragraph.end <= len(doc.words)
            previous_end = paragraph.end


def test_noise_lines_are_skipped():
    doc = parse_document(
        "# Title\n\n"
        "[ __](index.html \"nav\")[__](intro.html \"Intro\")\n"
        "Real content starts here and continues.\n"
        "\n"
        "static_methods.png"
    )
    assert "index.html" not in doc.words
    assert "intro.html" not in doc.words
    assert "staticmethods.png" in doc.words


def test_fixed_chunks_respect_sections_and_size():
    doc = parse_document(SAMPLE_DOC)
    target, overlap = 30, 5
    chunks = chunk_document(doc, "fixed", target=target, overlap=overlap)
    assert chunks
    by_heading: dict[str, list] = {}
    for chunk in chunks:
        assert chunk.span_end > chunk.span_start
        assert chunk.span_end - chunk.span_start <= target
        assert chunk.text.startswith("#")
        by_heading.setdefault(chunk.heading, []).append(chunk)
    for section in doc.sections:
        body_start = section.paragraphs[0].start
        body_end = section.paragraphs[-1].end
        for chunk in by_heading.get(section.heading, []):
            assert chunk.span_start >= body_start
            assert chunk.span_end <= body_end


def test_fixed_overlap_creates_shared_words():
    doc = parse_document(" ".join(f"word{i}" for i in range(40)))
    chunks = chunk_document(doc, "fixed", target=20, overlap=5)
    assert chunks[0].span_start == 0
    assert chunks[1].span_start < chunks[0].span_end  # second window starts in overlap


def test_fixed_unknown_strategy_raises():
    doc = parse_document(SAMPLE_DOC)
    with pytest.raises(ValueError):
        chunk_document(doc, "nope", target=30)


def test_fixed_invalid_target_raises():
    doc = parse_document(SAMPLE_DOC)
    with pytest.raises(ValueError):
        chunk_document(doc, "fixed", target=0)


def test_sentence_chunks_do_not_split_sentences():
    doc = parse_document(
        "Recursion splits a problem into smaller instances. "
        "The base case terminates. "
        "A recurrence relation describes cost. "
        "The master theorem solves them. "
        "Proof by induction proceeds. "
        "Divide and conquer combines results."
    )
    sentence_starts, sentence_ends = _doc_sentence_boundaries(doc)
    chunks = chunk_document(doc, "sentence", target=20, overlap=0)
    for chunk in chunks:
        assert chunk.span_start in sentence_starts
        assert chunk.span_end in sentence_ends


def _doc_sentence_boundaries(doc):
    from bisect import bisect_left

    text = " ".join(doc.words)
    offsets = [0]
    for word in doc.words[:-1]:
        offsets.append(offsets[-1] + len(word) + 1)
    import re

    bound_chars = [0]
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        bound_chars.append(match.end())
    if bound_chars[-1] < len(text):
        bound_chars.append(len(text))
    starts = sorted({bisect_left(offsets, c) for c in bound_chars})
    ends = sorted({bisect_left(offsets, c) for c in bound_chars if c < len(text)} | {len(doc.words)})
    return set(starts), set(ends)


def test_sentence_oversized_fallback_splits_by_words():
    doc = parse_document("You cannot split a sentence that dwarfs the budget it gets. " * 60)
    chunks = chunk_document(doc, "sentence", target=100, overlap=10)
    assert chunks
    assert all(c.span_end - c.span_start <= 100 for c in chunks)


def test_paragraph_chunks_align_to_paragraph_boundaries():
    paragraphs = [
        "Fundraising totals four dollars weekly",
        "Each member contributes one dollar",
        "The treasurer records every donation",
        "Cash flows in through the door",
        "Small bills hide inside the box",
        "Every week the totals update",
    ]
    doc = parse_document("# Club\n\n" + "\n\n".join(paragraphs))
    starts = {p.start for s in doc.sections for p in s.paragraphs}
    ends = {p.end for s in doc.sections for p in s.paragraphs}
    chunks = chunk_document(doc, "paragraph", target=40)
    assert all(c.span_start in starts and c.span_end in ends for c in chunks)


def test_paragraph_oversized_split_stays_bounded():
    doc = parse_document("# H\n\n" + ("content " * 300))
    chunks = chunk_document(doc, "paragraph", target=100)
    assert chunks
    assert all(c.span_end - c.span_start <= 100 for c in chunks)


def test_paragraph_overlap_creates_shared_words():
    paragraphs = [
        "Alpha beta gamma delta epsilon zeta",
        "Eta theta iota kappa lambda mu",
        "Nu xi omicron pi rho sigma",
        "Tau upsilon phi chi psi omega",
        "One two three four five six",
        "Seven eight nine ten eleven twelve",
    ]
    doc = parse_document("# Club\n\n" + "\n\n".join(paragraphs))
    starts = {p.start for s in doc.sections for p in s.paragraphs}
    ends = {p.end for s in doc.sections for p in s.paragraphs}
    chunks = chunk_document(doc, "paragraph", target=20, overlap=6)
    assert len(chunks) >= 2
    assert chunks[1].span_start < chunks[0].span_end
    assert all(c.span_start in starts and c.span_end in ends for c in chunks)


def test_merge_text_uses_heading():
    assert merge_text(["a", "b", "c"], 0, 2, "Intro") == "# Intro\n\na b"
    assert merge_text(["a", "b"], 0, 2, "") == "a b"


def test_span_overlap_fraction():
    assert span_overlap_fraction(0, 10, 2, 3) == pytest.approx(1.0)
    assert span_overlap_fraction(0, 10, 5, 20) == pytest.approx(5 / 15)
    assert span_overlap_fraction(10, 20, 0, 5) == 0.0
    assert span_overlap_fraction(0, 5, 0, 0) == 0.0