import json
import os

import fitz
import pytest

from app.api.ingest import ALLOWED_EXTENSIONS, _process_document


def _gen_pdf(tmp_path) -> str:
    path = tmp_path / "treedoc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((40, 40), "Red-black tree insert performs rotations to restore balance.", fontsize=11)
    doc.save(str(path))
    doc.close()
    return str(path)


def _pdf_files(path: str):
    return {"file": (os.path.basename(path), open(path, "rb"), "application/pdf")}


def test_allowed_extensions_whitelist():
    assert ALLOWED_EXTENSIONS == {".pdf", ".md", ".txt"}


class TestIngestEndpoint:
    def test_upload_pdf_queues_and_writes_safety(self, client, tmp_path):
        with open(_gen_pdf(tmp_path), "rb") as f:
            r = client.post(
                "/api/v1/ingest",
                files={"file": ("sample.pdf", f, "application/pdf")},
                data={"title": "Red Black Trees", "authors": "A. Knuth", "primary_category": "Algorithms", "all_categories": "Algorithms"},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "queued"
        assert body["job_id"]
        assert body["paper_id"]
        assert body["job_id"] in client.submitted_jobs
        assert os.path.exists(os.path.join(os.environ["NOTES_DIR"], f"{body['paper_id']}.pdf"))

    def test_unsafe_filename_is_neutralized(self, client, tmp_path):
        with open(_gen_pdf(tmp_path), "rb") as f:
            r = client.post(
                "/api/v1/ingest",
                files={"file": ("../../../../tmp/evil.pdf", f, "application/pdf")},
                data={"title": "Evil", "authors": "E", "primary_category": "A", "all_categories": "X"},
            )
        assert r.status_code == 200
        paper_id = r.json()["paper_id"]
        assert not os.path.exists("/tmp/evil.pdf")
        assert os.path.exists(os.path.join(os.environ["NOTES_DIR"], f"{paper_id}.pdf"))

    @pytest.mark.parametrize("name", ["virus.exe", "noextension", "script.sh"])
    def test_rejects_disallowed_extension(self, client, name):
        r = client.post(
            "/api/v1/ingest",
            files={"file": (name, b"content", "application/octet-stream")},
            data={"title": "t", "authors": "a", "primary_category": "c", "all_categories": "c"},
        )
        assert r.status_code == 400

    def test_job_status_unknown_returns_404(self, client):
        assert client.get("/api/v1/ingest/status/nope").status_code == 404


class TestIngestWorker:
    def test_process_document_indexes_and_updates_manifest(self, client, monkeypatch, tmp_path):
        from app.services.elasticsearch import es_service

        recorded = []

        def fake_index(chunk_id, document):
            recorded.append((chunk_id, document))

        monkeypatch.setattr(es_service, "index_chunk", fake_index)
        monkeypatch.setattr(es_service, "create_index_if_not_exists", lambda: None)
        monkeypatch.setattr(es_service, "refresh_index", lambda: None)

        local_path = os.path.join(tmp_path, "notes.md")
        with open(local_path, "w") as f:
            f.write("# Notes\n\nA quick brown fox keeps logarithmic search and insert time.\n")

        result = _process_document(
            "pid123",
            local_path,
            ".md",
            {"title": "Notes MD", "authors": ["Alice"], "primary_category": "Algorithms", "url": ""},
        )

        assert result["paper_id"] == "pid123"
        assert result["chunks_indexed"] == len(recorded) == 1
        assert recorded[0][1]["page_number"] == 1
        assert recorded[0][1]["paper_id"] == "pid123"
        assert recorded[0][1]["vector_embedding"] == [0.1] * 384

        manifest = json.load(open(os.path.join(os.environ["NOTES_DIR"], "library_metadata.json"), encoding="utf-8"))
        entry = next(e for e in manifest if e["id"] == "pid123")
        assert entry["title"] == "Notes MD"
        assert entry["authors"] == ["Alice"]
        assert entry["original_filename"] == ""
        assert entry["local_path"].endswith(".md")

    def test_process_document_removes_file_on_failure(self, client, monkeypatch, tmp_path):
        from app.services.elasticsearch import es_service

        def boom(*args, **kwargs):
            raise RuntimeError("chunking exploded")

        monkeypatch.setattr(es_service, "index_chunk", boom)
        monkeypatch.setattr(es_service, "create_index_if_not_exists", lambda: None)
        monkeypatch.setattr(es_service, "refresh_index", lambda: None)
        local_path = os.path.join(tmp_path, "notes.txt")
        with open(local_path, "w") as f:
            f.write("some text here to chunk and embed\n")

        with pytest.raises(RuntimeError):
            _process_document(
                "pidfail",
                local_path,
                ".txt",
                {"title": "F", "authors": [], "primary_category": "All", "url": ""},
            )
        assert not os.path.exists(local_path)

    def test_process_document_sets_pdf_path_for_pdfs(self, client, monkeypatch, tmp_path):
        from app.services.elasticsearch import es_service

        monkeypatch.setattr(es_service, "index_chunk", lambda cid, doc: None)
        monkeypatch.setattr(es_service, "create_index_if_not_exists", lambda: None)
        monkeypatch.setattr(es_service, "refresh_index", lambda: None)

        pdf = _gen_pdf(tmp_path)
        _process_document(
            "pidpdf",
            pdf,
            ".pdf",
            {"title": "PDF", "authors": [], "primary_category": "A", "url": ""},
        )
        manifest = json.load(open(os.path.join(os.environ["NOTES_DIR"], "library_metadata.json"), encoding="utf-8"))
        entry = next(e for e in manifest if e["id"] == "pidpdf")
        assert entry["pdf_path"] == entry["local_path"] and entry["pdf_path"].endswith(".pdf")