import pytest

from app.services.llm import LLMService


@pytest.fixture()
def offline_service(monkeypatch):
    svc = LLMService()
    monkeypatch.setattr(svc, "api_url", "http://127.0.0.1:9/chat/completions")
    return svc


def test_build_payload_structure():
    svc = LLMService()
    payload = svc._build_payload(
        "What is a heap?",
        context_chunks=[{"title": "Heaps", "chunk_id": "p9_chunk_0", "text": "A heap is a tree."}],
        history=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    assert payload["model"]
    assert payload["stream"] is True
    contents = " ".join(m["content"] for m in payload["messages"])
    assert "What is a heap?" in contents
    assert "A heap is a tree." in contents
    assert any(m["role"] == "system" for m in payload["messages"])


async def test_rewrite_falls_back_to_question_when_llm_unreachable(offline_service):
    rewritten = await offline_service.rewrite_query_for_search("What is dynamic programming?")
    assert rewritten == "What is dynamic programming?"


async def test_generate_returns_error_message_when_llm_unreachable(offline_service):
    answer = await offline_service.generate_response("q", [])
    assert "Failed to generate answer" in answer


async def test_rewrite_uses_history(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json):
            captured["payload"] = json
            return _Response({"choices": [{"message": {"content": "standalone query"}}]})

    class _Response:
        status_code = 200

        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

    monkeypatch.setattr("app.services.llm.httpx.AsyncClient", FakeClient)
    svc = LLMService()
    result = await svc.rewrite_query_for_search(
        "Follow up?",
        history=[{"role": "user", "content": "Prior Q"}, {"role": "assistant", "content": "Prior A"}],
    )
    assert result == "standalone query"
    assert "Prior Q" in captured["payload"]["messages"][0]["content"]