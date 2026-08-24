import pytest

from app.services.chunking import (
    chunk_document,
    merge_text,
    parse_document,
    span_overlap_fraction,
)

SAMPLE_DOC = """# 1. Basics

This is a section with some text content. It has some basic concepts
that we want to chunk.

# 2. Advanced Topics

Here we discuss more complex ideas. The text is longer and more
complex, requiring different chunking strategies to handle it
properly.

# 3. Advanced Topics (continued)

More content about advanced topics. This section has multiple
paragraphs that need to be handled correctly by the chunking
strategies.

# 4. Conclusion

Final thoughts and summary of the document.
"""

def test_fixed_chunks_valid_and_headed():
    doc = parse_document(SAMPLE_DOC)
    target, overlap = 30, 5
    chunks = chunk_document(doc, "fixed", target=target, overlap=overlap)
    assert chunks
    for chunk in chunks:
        assert chunk.span_end > chunk.span_start
        assert chunk.span_end - chunk.span_start <= target
        assert chunk.heading
        assert chunk.text.startswith("#")

def test_fixed_chunks_respect_size__small_target():
    doc = parse_document(SAMPLE_DOC)
    chunks = chunk_document(doc, "fixed", target=5, overlap=2)
    assert chunks
    for chunk in chunks:
        assert chunk.span_end - chunk.span_start <= 5

def test_fixed_overlap_creates_shared_words():
    doc = parse_document(" ".join(f"word{i}" for i in range(40)))
    chunks = chunk_document(doc, "fixed", target=20, overlap=5)
    assert chunks[0].span_start == 0
    assert chunks[1].span_start < chunks[0].span_end

def test_fixed_unknown_strategy_raises():
    doc = parse_document(SAMPLE_DOC)
    with pytest.raises(ValueError):
        chunk_document(doc, "nope", target=30)

def test_fixed_invalid_target_raises():
    doc = parse_document(SAMPLE_DOC)
    with pytest.raises(ValueError):
        chunk_document(doc, "fixed", target=0)

def test_fixed_invalid_overlap_raises():
    doc = parse_document(SAMPLE_DOC)
    with pytest.raises(ValueError):
        chunk_document(doc, "fixed", target=30, overlap=-1)

def test_recursive_chunks_valid_and_bounded():
    doc = parse_document(SAMPLE_DOC)
    target = 20
    chunks = chunk_document(doc, "recursive", target=target, overlap=0)
    assert chunks
    for chunk in chunks:
        assert chunk.span_end > chunk.span_start
        assert chunk.span_end - chunk.span_start <= target
        assert chunk.heading

def test_recursive_oversized_split_stays_bounded():
    doc = parse_document("You cannot split a sentence that dwarfs the budget it gets. " * 60)
    chunks = chunk_document(doc, "recursive", target=100, overlap=10)
    assert chunks
    assert all(c.span_end - c.span_start <= 100 for c in chunks)

def test_document_chunks_respect_sections_and_size():
    doc = parse_document(SAMPLE_DOC)
    target, overlap = 30, 5
    chunks = chunk_document(doc, "document", target=target, overlap=overlap)
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

def test_document_oversized_split_stays_bounded():
    doc = parse_document(SAMPLE_DOC)
    chunks = chunk_document(doc, "document", target=100)
    assert chunks
    for chunk in chunks:
        assert chunk.span_end - chunk.span_start <= 100

def test_document_overlap_creates_shared_words():
    doc = parse_document(" ".join(f"word{i}" for i in range(40)))
    chunks = chunk_document(doc, "document", target=20, overlap=5)
    assert chunks[0].span_start == 0
    assert chunks[1].span_start < chunks[0].span_end

def test_sentence_maps_to_recursive():
    doc = parse_document(SAMPLE_DOC)
    chunks = chunk_document(doc, "sentence", target=20, overlap=0)
    assert chunks
    assert all(c.span_end > c.span_start for c in chunks)

def test_paragraph_maps_to_document():
    doc = parse_document(SAMPLE_DOC)
    chunks = chunk_document(doc, "paragraph", target=40, overlap=5)
    assert chunks
    assert all(c.span_end > c.span_start for c in chunks)

def test_hierarchical_chunks_valid_and_stay_in_sections():
    doc = parse_document(SAMPLE_DOC)
    target = 20
    chunks = chunk_document(doc, "hierarchical", target=target, overlap=2)
    assert chunks
    by_heading: dict[str, list] = {}
    for chunk in chunks:
        assert chunk.span_end > chunk.span_start
        assert chunk.text.startswith("#")
        by_heading.setdefault(chunk.heading, []).append(chunk)

    for section in doc.sections:
        body_start = section.paragraphs[0].start
        body_end = section.paragraphs[-1].end
        for chunk in by_heading.get(section.heading, []):
            assert chunk.span_start >= body_start
            assert chunk.span_end <= body_end

def test_hierarchical_has_parent_and_child_levels():
    doc = parse_document(" ".join(f"word{i}" for i in range(60)))
    chunks = chunk_document(doc, "hierarchical", target=10, overlap=0)
    assert chunks
    sizes = sorted(c.span_end - c.span_start for c in chunks)

    assert max(sizes) >= 19
    assert min(sizes) <= 11

    covered = set()
    for c in chunks:
        covered.update(range(c.span_start, c.span_end))
    assert covered == set(range(60))

def test_parse_document_paragraph_ranges_are_contiguous():
    doc = parse_document(SAMPLE_DOC)
    for section in doc.sections:
        previous_end = -1
        for paragraph in section.paragraphs:
            assert paragraph.start >= previous_end
            assert paragraph.end > paragraph.start
            assert paragraph.end <= len(doc.words)
            previous_end = paragraph.end

def test_parse_document_tracks_page_per_word():
    doc = parse_document("""
# Title

Some content here.

# Section

More content.
""")
    assert doc.word_pages == [1] * len(doc.words)

def test_noise_lines_are_skipped():
    doc = parse_document(
        "# Title\n\n"
        "[ __](index.html \"nav\")[__](intro.html \"Intro\")\n"
        "Real content starts here and continues.\n"
        "\n"
        "static_methods.png"
    )
    assert "index.html" not in doc.words
    assert "staticmethods.png" in doc.words

def test_merge_text_uses_heading():
    doc = parse_document(SAMPLE_DOC)
    result = merge_text(doc.words, 0, 10, "Heading")
    assert result.startswith("# Heading")

def test_span_overlap_fraction():
    a_start, a_end = 0, 50
    b_start, b_end = 25, 75
    assert span_overlap_fraction(a_start, a_end, b_start, b_end) == 0.5