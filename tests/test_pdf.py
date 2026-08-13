import json
import os
import shutil

import fitz
import pytest

from app.services.pdf import PdfService


@pytest.fixture()
def two_page_pdf(tmp_path):
    path = tmp_path / "notes.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Recursion relies on base cases and smaller subproblems.", fontsize=11)
    doc.new_page().insert_text((72, 72), "Recursion is the strongest tool for divide and conquer algorithms.", fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


class TestHighlightBySearch:
    def test_highlights_only_nearby_page(self, two_page_pdf):
        doc = fitz.open(str(two_page_pdf))
        svc = PdfService()
        count = svc._highlight_by_search(doc, "Recursion is the strongest tool for divide", page_range=range(1, 2))
        assert count > 0
        assert len(list(doc[1].annots())) > 0
        assert not list(doc[0].annots())
        doc.close()

    def test_multi_page_range(self, two_page_pdf):
        doc = fitz.open(str(two_page_pdf))
        svc = PdfService()
        count = svc._highlight_by_search(doc, "Recursion", page_range=range(0, 2))
        assert count >= 2
        doc.close()

    def test_no_match_returns_zero(self, two_page_pdf):
        doc = fitz.open(str(two_page_pdf))
        svc = PdfService()
        assert svc._highlight_by_search(doc, "quantum chromodynamics forever and today", page_range=range(0, 2)) == 0
        doc.close()


class TestManifestMatching:
    def test_match_by_id(self):
        manifest = [{"id": "abc123", "local_path": "x/y.md"}]
        assert PdfService()._match_entry(manifest, "abc123")["id"] == "abc123"

    def test_match_by_local_path_stem(self):
        manifest = [{"id": "other", "local_path": "csc_notes/my_doc.md"}]
        entry = PdfService()._match_entry(manifest, "my_doc")
        assert entry is not None and entry["id"] == "other"

    def test_no_match_returns_none(self):
        assert PdfService()._match_entry([{"id": "a"}], "zzz") is None


def test_stream_pdf_headers():
    response = PdfService()._serve_bytes(b"%PDF-1.4 data", range_header=None)
    assert response.headers["content-type"] == "application/pdf"


def test_cache_ttl_expiry():
    import time

    svc = PdfService()
    svc._set_cached("k", b"data")
    assert svc._get_cached("k") == b"data"
    svc._cache["k"] = (time.time() - svc._cache_ttl - 1, b"data")
    assert svc._get_cached("k") is None


def test_notes_endpoint_highlights_on_serve(client, two_page_pdf, monkeypatch):
    from app.services.elasticsearch import es_service

    paper_id = "pdfsrvtest"
    notes_dir = os.environ["NOTES_DIR"]
    os.makedirs(notes_dir, exist_ok=True)
    dest = os.path.join(notes_dir, "notes.pdf")
    shutil.copy(two_page_pdf, dest)

    manifest = [
        {
            "id": paper_id,
            "title": "Notes",
            "authors": [],
            "category": "Compilers",
            "local_path": dest,
            "pdf_path": dest,
        }
    ]
    with open(os.path.join(notes_dir, "library_metadata.json"), "w") as f:
        json.dump(manifest, f)

    doc = fitz.open(dest)
    page_two_text = doc[1].get_text("text")
    doc.close()

    class FakeEsClient:
        def get(self, index, id):
            return {"_source": {"text": page_two_text.strip(), "page_number": 2}}

    monkeypatch.setattr(es_service, "client", FakeEsClient())
    r = client.get(f"/api/v1/pdf/{paper_id}?chunk_id={paper_id}_chunk_0")

    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    out = fitz.open(stream=r.content, filetype="pdf")
    annotations = [a for p in out for a in (p.annots() or [])]
    assert annotations


def test_render_page_images(tmp_path, monkeypatch):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Page one alpha text.", fontsize=11)
    doc.new_page().insert_text((72, 72), "Page two beta text.", fontsize=11)
    pdf_path = tmp_path / "render.pdf"
    doc.save(str(pdf_path))
    doc.close()

    notes_dir = tmp_path / "notes"
    os.makedirs(notes_dir, exist_ok=True)
    monkeypatch.setattr("app.core.config.settings.NOTES_DIR", str(notes_dir))
    monkeypatch.setattr("app.core.config.settings.NOTES_MANIFEST", str(notes_dir / "library_metadata.json"))
    dest = os.path.join(notes_dir, "render.pdf")
    shutil.copy(str(pdf_path), dest)

    manifest = [{"id": "r1", "title": "R", "authors": [], "category": "C", "local_path": dest, "pdf_path": dest}]
    with open(notes_dir / "library_metadata.json", "w") as f:
        json.dump(manifest, f)

    svc = PdfService()
    out = svc.render_page_images("r1", [1, 2, 99], dpi=80)
    assert [o["page"] for o in out] == [1, 2]
    assert out[0]["image"] and out[0]["image"].startswith("data:image/png;base64,")
    assert "Page one alpha text." in out[0]["text"]

    assert svc.render_page_images("unknown_paper", [1], dpi=80) == []