from app.api.rag import _build_context, _build_sources
from app.services.elasticsearch import es_service


def _hit(**overrides):
    base = {
        "score": 7.5,
        "paper_id": "p9",
        "title": "Heaps",
        "chunk_id": "p9_chunk_3",
        "page_number": 2,
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
    assert sources[0]["matching_score"] == 7.5
    assert sources[0]["source_url"] == "https://example.com"
    assert sources[0]["local_pdf"] is True


def test_build_sources_defaults(monkeypatch):
    sources = _build_sources([_hit(local_pdf=False, source_url=None)])
    assert sources[0]["local_pdf"] is False
    assert sources[0]["source_url"] == ""


def test_build_context_uses_window(monkeypatch):
    monkeypatch.setattr(es_service, "get_chunk_window", lambda chunk_id, window_size: "window-text")
    context = _build_context([_hit()], window_size=2)
    assert context[0]["title"] == "Heaps"
    assert context[0]["text"] == "window-text"


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