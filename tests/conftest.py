import asyncio
import os
import pathlib
import tempfile

_TMP = pathlib.Path(tempfile.mkdtemp(prefix="rag_tests_"))

os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP / 'test.db'}"
os.environ["NOTES_DIR"] = str(_TMP / "csc_notes")
os.environ["INDEX_NAME"] = "test_csc_notes"
os.environ["LLM_API_BASE"] = "http://127.0.0.1:9/v1"
os.environ["LLM_MODEL"] = "test-model"
os.environ["ES_HOST"] = "http://localhost:9200"
os.environ["ES_USER"] = ""
os.environ["ES_PASSWORD"] = ""
os.environ["SECRET_KEY"] = "test-secret-key"

import pytest
from fastapi.testclient import TestClient

from app.db.session import Base, engine
from app.main import create_app


async def _reset_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


def _fake_hits() -> list[dict]:
    return [
        {
            "score": 9.5,
            "paper_id": "p1",
            "title": "Recursion Notes",
            "authors": ["A. Einstein"],
            "category": "Algorithms",
            "chunk_id": "p1_chunk_0",
            "page_number": 3,
            "page_start": 3,
            "page_end": 5,
            "text": "Recursion solves a problem by solving smaller instances of itself.",
            "source_url": "",
            "local_pdf": True,
        },
        {
            "score": 8.0,
            "paper_id": "p2",
            "title": "Sorting Notes",
            "authors": ["B. Turing"],
            "category": "Algorithms",
            "chunk_id": "p2_chunk_1",
            "page_number": 7,
            "page_start": 7,
            "page_end": 7,
            "text": "Sorting rearranges elements into a specified order.",
            "source_url": "https://example.org/sorting",
            "local_pdf": False,
        },
    ]


@pytest.fixture()
def client(monkeypatch):
    asyncio.run(_reset_db())

    from app.services.embedding import embedding_service
    from app.services.elasticsearch import es_service
    from app.services.llm import llm_service
    from app.services.memory import memory_store
    from app.services.ingestion import ingestion_manager

    monkeypatch.setattr(es_service, "verify_connection", lambda: True)
    monkeypatch.setattr(es_service, "keyword_search", lambda query, limit=5: _fake_hits())
    monkeypatch.setattr(es_service, "semantic_search_cosine", lambda *a, **k: _fake_hits())
    monkeypatch.setattr(es_service, "hybrid_search", lambda **k: _fake_hits())
    monkeypatch.setattr(es_service, "get_chunk_window_entries", lambda chunk_id, window_size=1: [
        {"chunk_id": chunk_id, "title": "Notes", "paper_id": chunk_id.split("_")[0],
         "text": "windowed context text", "page_start": 1, "page_end": 2}
    ])

    history: dict[str, list[dict[str, str]]] = {}

    async def _get_history(session_id: str) -> list[dict[str, str]]:
        return list(history.get(session_id, []))

    async def _add_message(session_id: str, role: str, content: str) -> None:
        history.setdefault(session_id, []).append({"role": role, "content": content})

    async def _clear_session(session_id: str) -> None:
        history.pop(session_id, None)

    async def _close() -> None:
        pass

    monkeypatch.setattr(memory_store, "get_history", _get_history)
    monkeypatch.setattr(memory_store, "add_message", _add_message)
    monkeypatch.setattr(memory_store, "clear_session", _clear_session)
    monkeypatch.setattr(memory_store, "close", _close)

    async def _rewrite(question: str, history=None) -> str:
        return question

    async def _generate(question: str, context_chunks=None, history=None) -> str:
        return "Synthesized answer grounded in the provided notes."

    async def _stream(question: str, context_chunks=None, history=None):
        yield "Token one. "
        yield "Token two."

    monkeypatch.setattr(llm_service, "rewrite_query_for_search", _rewrite)
    monkeypatch.setattr(llm_service, "generate_response", _generate)
    monkeypatch.setattr(llm_service, "stream_response", _stream)
    monkeypatch.setattr(embedding_service, "get_embeddings", lambda text, task=None: [0.1] * 384)
    monkeypatch.setattr(embedding_service, "get_query_embeddings", lambda text: [0.1] * 384)
    monkeypatch.setattr(embedding_service, "get_document_embeddings", lambda text: [0.1] * 384)

    submitted: dict[str, tuple] = {}

    def _fake_submit(fn, *args, **kwargs) -> str:
        job_id = f"job{len(submitted) + 1:04d}"
        submitted[job_id] = (fn, args, kwargs)
        return job_id

    monkeypatch.setattr(ingestion_manager, "submit", _fake_submit)

    with TestClient(create_app()) as test_client:
        test_client.memory_history = history
        test_client.submitted_jobs = submitted
        yield test_client


@pytest.fixture()
def auth_headers(client) -> dict[str, str]:
    username = "alice"
    password = "s3cret-pass"
    client.post("/auth/register", json={"username": username, "password": password})
    token = client.post("/auth/login", json={"username": username, "password": password}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}