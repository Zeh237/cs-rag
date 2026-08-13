import json

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


def test_build_payload_renders_numbered_sources_with_page_ranges():
    svc = LLMService()
    payload = svc._build_payload(
        "Tell me about compilers",
        context_chunks=[
            {"title": "Compilers", "chunk_id": "c_chunk_0", "text": "body a", "page_start": 40, "page_end": 43},
            {"title": "Compilers", "chunk_id": "c_chunk_1", "text": "body b", "page_start": 44, "page_end": 44},
        ],
    )
    user = payload["messages"][-1]["content"]
    assert "[1] Compilers (pp. 40-43):" in user
    assert "[2] Compilers (p. 44):" in user
    system = payload["messages"][0]["content"]
    assert "bracketed" in system
    assert "never guess or invent page numbers" in system


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


class FakeStream:
    def __init__(self, data_lines, status=200):
        self._lines = [f"data: {d}" for d in data_lines]
        self.status_code = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeAsyncClient:
    def __init__(self, streams):
        self._streams = list(streams)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, method, url, json=None):
        return self._streams.pop(0)


def _tool_call_stream(tc_id, name, args_json):
    lines = [
        json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": tc_id, "type": "function", "function": {"name": name}}]}}]}),
    ]
    for i in range(0, len(args_json), 5):
        lines.append(json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": args_json[i:i + 5]}}]}}]}))
    lines.append(json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}))
    lines.append("[DONE]")
    return lines


def _text_stream(tokens):
    lines = [json.dumps({"choices": [{"delta": {"content": t}}]}) for t in tokens]
    lines.append(json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    lines.append("[DONE]")
    return lines


async def test_stream_research_response_tool_loop(monkeypatch):
    svc = LLMService()
    read_calls = []

    async def read_pages(pages):
        read_calls.append(list(pages))
        return [{"page": p, "image": "data:image/png;base64,AAAA", "text": ""} for p in pages]

    callbacks = {
        "read_pages": read_pages,
        "search_document": lambda q: '[{"page": 5, "snippet": "x"}]',
        "get_toc": lambda: "1 Intro ... p. 1",
    }
    streams = [
        FakeStream(_tool_call_stream("call_1", "read_pages", '{"start":12,"end":13}')),
        FakeStream(_text_stream(["Answer ", "with ", "p. 12", "."])),
    ]
    shared_client = FakeAsyncClient(streams)
    monkeypatch.setattr("app.services.llm.httpx.AsyncClient", lambda *a, **k: shared_client)

    seed = [{"page": 3, "image": "data:image/png;base64,AAAA", "text": ""}]
    out = []
    async for chunk in svc.stream_research_response(
        "Q", paper_id="p1", title="T", seed_images=seed, callbacks=callbacks, max_pages=5, max_tool_iters=2
    ):
        out.append(chunk)
    assert "".join(out) == "Answer with p. 12."
    assert read_calls == [[12, 13]]


async def test_stream_research_response_uses_vision_model(monkeypatch):
    svc = LLMService()
    captured = {}

    class CapStream(FakeStream):
        pass

    captured_stream = CapStream(_text_stream(["done."]))

    class CapturingAsyncClient(FakeAsyncClient):
        def stream(self, method, url, json=None):
            captured["model"] = json["model"]
            return super().stream(method, url, json=json)

    shared_client = CapturingAsyncClient([captured_stream])
    monkeypatch.setattr("app.services.llm.httpx.AsyncClient", lambda *a, **k: shared_client)

    out = []
    async for chunk in svc.stream_research_response(
        "Q", paper_id="p1", title="T", seed_images=[{"page": 1, "image": "data:image/png;base64,AAAA", "text": ""}],
        callbacks={}, max_pages=5, max_tool_iters=1,
    ):
        out.append(chunk)
    assert captured["model"] == svc.vision_model_name


async def test_stream_research_response_returns_empty_without_images():
    svc = LLMService()
    out = []
    async for chunk in svc.stream_research_response(
        "Q", paper_id="p1", title="T", seed_images=[], callbacks={}, max_pages=5
    ):
        out.append(chunk)
    assert out == []