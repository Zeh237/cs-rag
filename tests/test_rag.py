from app.api.rag import (
    _build_context,
    _build_sources,
    _build_toc,
    _group_by_document,
    _linkify_citations,
    _linkify_page_refs,
    _research_enabled,
)
from app.services.elasticsearch import es_service
from app.services.pdf import pdf_service


def _hit(**overrides):
    base = {
        "score": 7.5,
        "paper_id": "p9",
        "title": "Heaps",
        "chunk_id": "p9_chunk_3",
        "page_number": 2,
        "page_start": 2,
        "page_end": 4,
        "source_url": "https://example.com",
        "local_pdf": True,
    }
    base.update(overrides)
    return base


def test_build_sources_field_mapping():
    sources = _build_sources([_hit()])
    assert sources[0]["anchor_chunk_id"] == "p9_chunk_3"
    assert sources[0]["paper_id"] == "p9"
    assert sources[0]["page_number"] == 2
    assert sources[0]["page_start"] == 2
    assert sources[0]["page_end"] == 4
    assert sources[0]["matching_score"] == 7.5
    assert sources[0]["source_url"] == "https://example.com"
    assert sources[0]["local_pdf"] is True


def test_build_sources_defaults(monkeypatch):
    sources = _build_sources([_hit(local_pdf=False, source_url=None, page_start=None, page_end=None)])
    assert sources[0]["local_pdf"] is False
    assert sources[0]["source_url"] == ""
    assert sources[0]["page_start"] == 2
    assert sources[0]["page_end"] == 2


def test_build_context_uses_window(monkeypatch):
    monkeypatch.setattr(
        es_service,
        "get_chunk_window_entries",
        lambda chunk_id, window_size=1: [
            {
                "chunk_id": chunk_id,
                "title": "Heaps",
                "paper_id": "p9",
                "text": "window-text",
                "page_start": 9,
                "page_end": 10,
            }
        ],
    )
    context = _build_context([_hit()], window_size=2)
    assert context[0]["title"] == "Heaps"
    assert context[0]["text"] == "window-text"
    assert context[0]["page_start"] == 9
    assert context[0]["page_end"] == 10


def test_group_by_document_spreads_across_papers():
    hits = [
        _hit(paper_id="a", chunk_id="a_chunk_0", score=9.0),
        _hit(paper_id="a", chunk_id="a_chunk_1", score=8.0),
        _hit(paper_id="b", chunk_id="b_chunk_0", score=7.0),
        _hit(paper_id="b", chunk_id="b_chunk_1", score=6.0),
    ]
    anchors = _group_by_document(hits, limit=4, max_papers=2, per_paper=2)
    papers = {a["paper_id"] for a in anchors}
    assert papers == {"a", "b"}


def test_linkify_citations_maps_markers():
    context = [
        {"paper_id": "p9", "chunk_id": "p9_chunk_3", "page_start": 2, "title": "Heaps"},
        {"paper_id": "p7", "chunk_id": "p7_chunk_0", "page_start": 5, "title": "Trees"},
    ]
    answer = "Heaps are efficient [1]. Trees too [2]. Out of range [9]."
    out = _linkify_citations(answer, context)
    assert 'href="/api/v1/pdf/p9?chunk_id=p9_chunk_3#page=2"' in out
    assert 'href="/api/v1/pdf/p7?chunk_id=p7_chunk_0#page=5"' in out
    assert "[9]" in out


def test_linkify_page_refs_single_document():
    out = _linkify_page_refs("See (p. 12) and (pp. 40-43).", "compilers_notes")
    assert 'href="/api/v1/pdf/compilers_notes#page=12"' in out
    assert 'href="/api/v1/pdf/compilers_notes#page=40"' in out
    assert "(p. 12)</a>" in out
    assert "(pp. 40-43)</a>" in out


def test_build_toc_from_notes_markdown(monkeypatch, tmp_path):
    md = tmp_path / "notes.md"
    md.write_text("# Chapter One\n\n[[PAGE 4]]\n\nBody text.\n\n## Sub Section\n\n[[PAGE 5]]\n\nMore.\n")
    monkeypatch.setattr(pdf_service, "_find_notes_entry", lambda pid: {"local_path": str(md)})
    toc = _build_toc("p1")
    assert "Chapter One ... p. 1" in toc
    assert "Sub Section ... p. 4" in toc


def test_research_disabled_when_no_vision_model(monkeypatch):
    monkeypatch.setattr("app.api.rag.settings.VISION_LLM_MODEL", "")
    monkeypatch.setattr("app.api.rag.settings.LLM_API_BASE", "http://localhost:1234/v1")
    assert _research_enabled() is False


def test_research_enabled_with_vision_model(monkeypatch):
    monkeypatch.setattr("app.api.rag.settings.VISION_LLM_MODEL", "qwen/qwen3-vl-4b")
    monkeypatch.setattr("app.api.rag.settings.LLM_API_BASE", "http://localhost:1234/v1")
    assert _research_enabled() is True


class TestAskEndpoint:
    def test_ask_returns_answer_and_sources(self, client):
        r = client.get("/rag/ask", params={"q": "explain heaps", "limit": 2})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "Synthesized answer grounded in the provided notes."
        assert len(body["sources"]) == 2
        assert body["sources"][0]["anchor_chunk_id"] == "p1_chunk_0"
        assert body["search_query"] == "explain heaps"

    def test_ask_no_hits_returns_fallback(self, client, monkeypatch):
        monkeypatch.setattr(es_service, "hybrid_search", lambda **k: [])
        r = client.get("/rag/ask", params={"q": "something obscure"})
        assert r.status_code == 200
        assert "No matching notes" in r.json()["answer"]

    def test_ask_stream_events(self, client, auth_headers):
        chat = client.post("/chats", json={"title": "RAG"}, headers=auth_headers).json()
        r = client.get(
            "/rag/ask/stream",
            params={"q": "explain heaps", "chat_id": chat["id"]},
            headers=auth_headers,
        )
        assert r.status_code == 200
        assert "event: search_query" in r.text
        assert "event: sources" in r.text
        assert "event: token" in r.text
        assert "Token one." in r.text
        assert "event: done" in r.text

    def test_ask_stream_requires_auth(self, client):
        r = client.get("/rag/ask/stream", params={"q": "hi", "chat_id": 1})
        assert r.status_code in (401, 403)