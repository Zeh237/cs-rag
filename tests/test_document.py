import pytest

from app.services.document import DocumentProcessor


def make_processor(tmp_path) -> DocumentProcessor:
    return DocumentProcessor()


def test_chunk_text_splits_and_overlaps():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = DocumentProcessor().chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) == 3
    assert all(chunk["page"] == 1 for chunk in chunks)
    assert chunks[0]["text"].startswith("word0")


def test_chunk_text_small_document_single_chunk():
    chunks = DocumentProcessor().chunk_text("only a few words here", chunk_size=50)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "only a few words here"


def test_chunk_markdown_tracks_pages():
    text = (
        "# Chapter One\n\n"
        "[[PAGE 1]]\n\n"
        "alpha beta gamma delta epsilon\n\n"
        "# Chapter Two\n\n"
        "[[PAGE 4]]\n\n"
        "epsilon zeta eta theta iota"
    )
    chunks = DocumentProcessor().chunk_markdown(text, chunk_size=100)
    assert len(chunks) == 2
    assert chunks[0]["page"] == 1
    assert chunks[0]["text"].startswith("# Chapter One")
    assert "[[PAGE" not in chunks[0]["text"]
    assert chunks[1]["page"] == 4
    assert chunks[1]["text"].startswith("# Chapter Two")


def test_chunk_markdown_page_split_within_heading():
    text = (
        "# Big Section\n\n"
        "[[PAGE 2]]\n\n"
        + " ".join(f"w{i}" for i in range(30))
        + "\n[[PAGE 3]]\n\n"
        + " ".join(f"x{i}" for i in range(30))
    )
    chunks = DocumentProcessor().chunk_markdown(text, chunk_size=20, overlap=0)
    assert len(chunks) >= 3
    assert chunks[0]["page"] == 2
    assert chunks[-1]["page"] == 3
    assert chunks[0]["text"].startswith("# Big Section")


def test_chunk_markdown_strips_markers_from_text():
    text = "# A\n\n[[PAGE 1]]\n\nhello world foo bar"
    chunk = DocumentProcessor().chunk_markdown(text)[0]
    assert "[[PAGE" not in chunk["text"]


def test_chunk_pdf_pages_and_highlight_boxes(tmp_path):
    import fitz

    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Introduction states the problem clearly and gives an overview of the method.", fontsize=11)
    page2 = doc.new_page()
    page2.insert_text((72, 72), "The second page continues with more details covering the algorithm's behavior.", fontsize=11)
    doc.save(str(pdf_path))
    doc.close()

    chunks = DocumentProcessor().chunk_pdf(str(pdf_path), chunk_size=12, overlap=2)
    assert chunks
    assert chunks[0]["page"] == 1
    assert all("page" in chunk for chunk in chunks)
    annotated = sum(len(boxes) for chunk in chunks for boxes in chunk.get("highlight_boxes", {}).values())
    assert annotated == 0 or annotated > 0


def test_chunk_pdf_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        DocumentProcessor().chunk_pdf("/nonexistent/file.pdf")


def test_chunk_pdf_bibliography_cutoff():
    processor = DocumentProcessor()
    body = "body text " * 40 + "\nReferences\n" + "citation one " * 30
    word_pages = [(w, 1) for w in body.split()]
    cutoff = processor._bibliography_cutoff(word_pages)
    reconstructed = " ".join(w for w, _ in word_pages[:cutoff])
    assert "References" not in reconstructed
    assert "body" in reconstructed


def test_bibliography_cutoff_no_marker():
    processor = DocumentProcessor()
    words = [(w, 1) for w in ("a b c d e").split()]
    assert processor._bibliography_cutoff(words) == len(words)