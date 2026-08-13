# CSC RAG Engine — Test Report

Run date: 2026-08-13
Commit: `f276fa9` (feat: Refactor configuration and cleanup unused code in RAG Engine)
Python: 3.12.12 (.venv)

## 1. Result summary

| Result | Count |
| --- | --- |
| Tests passed | **110** |
| Tests failed | **0** |
| Tests skipped | 0 |
| Errors | 0 |
| Warnings | 1 (non-fatal deprecation warning, see §7) |

```
======================== 110 passed, 1 warning in 5.36s ========================
```

Command used:

```bash
.venv/bin/python -m pytest -v
```

The suite needs **no running infrastructure** — Elasticsearch, Redis, and the LLM
server are all mocked or replaced with in-memory fakes (see §4). The only hard
dependencies are Python + the packages in `requirements.txt` and
`requirements-dev.txt` (notably PyMuPDF `fitz` for the PDF tests).

## 2. What the system under test is

The project is a **retrieval-augmented generation (RAG) engine** that answers
questions about undergraduate computer-science course notes (a private library
of open textbooks on algorithms, machine learning, operating systems, theory of
computation, compilers, networking, data structures, and discrete math).

The pipeline:

1. **Ingest** — PDF/Markdown/TXT documents are uploaded, queued as background
   jobs, chunked, embedded, and indexed into Elasticsearch (`app/services/ingestion.py`,
   `app/services/document.py`).
2. **Chunking** — three pluggable strategies (`fixed`, `sentence`, `paragraph`)
   implemented over a shared word-span coordinate system (`app/services/chunking.py`).
3. **Retrieval** — four pipelines: keyword, semantic (dense embeddings), hybrid
   (both combined), and hybrid + cross-encoder reranking (`app/services/elasticsearch.py`,
   `app/services/reranker.py`).
4. **Answer generation** — an LLM (local OpenAI-compatible server) produces a
   grounded answer; the RAG layer links every citation to a page in the PDF
   (`app/api/rag.py`, `app/services/llm.py`).
5. **PDF serving** — serves the notes PDF with the cited chunk highlighted on
   the correct page (`app/services/pdf.py`).

## 3. Test suite layout

| File | Tests | Area |
| --- | --- | --- |
| `tests/conftest.py` | (fixtures) | Shared `client` fixture, mocks, fake data |
| `tests/test_auth.py` | 10 | Password hashing, JWT tokens, auth endpoints |
| `tests/test_chunking.py` | 17 | Word-span parser and the three chunking strategies |
| `tests/test_document.py` | 9 | Markdown/PDF chunking, bibliography cutoff |
| `tests/test_embedding.py` | 2 | Embedding service with a fake model |
| `tests/test_endpoints.py` | 5 | Search + chat HTTP endpoints, health check |
| `tests/test_es.py` | 10 | Elasticsearch index/search helpers (fake client) |
| `tests/test_evaluation.py` | 19 | IR metrics, bootstrap CIs, dataset builders |
| `tests/test_ingest.py` | 10 | Ingest endpoint, upload safety, background worker |
| `tests/test_llm.py` | 8 | LLM payloads, query rewrite, streaming tool calls |
| `tests/test_pdf.py` | 10 | PDF highlighting, manifest matching, page rendering |
| `tests/test_rag.py` | 10 | RAG context/sources, citations, ask/stream endpoints |
| **Total** | **110** | |

## 4. Test infrastructure (conftest)

`tests/conftest.py` sets a deterministic environment *before* importing the app:

- `DATABASE_URL` → an in-memory SQLite file in a temp dir
- `NOTES_DIR` / `INDEX_NAME` → isolated temp locations
- `LLM_API_BASE` → `http://127.0.0.1:9/v1` (a port that refuses connections, so
  LLM calls fail fast and the fallback paths are exercised)
- `SECRET_KEY` → a fixed test key

The shared `client` fixture then **monkeypatches** every external service:

- Elasticsearch (`es_service`) — `verify_connection`, `keyword_search`,
  `semantic_search_cosine`, `hybrid_search`, `get_chunk_window_entries` are
  replaced with canned hits, so no ES server is needed.
- Embeddings (`embedding_service`) — returns a fixed 384-dim vector `[0.1] * 384`.
- LLM (`llm_service`) — `rewrite_query_for_search`, `generate_response`,
  `stream_response` return canned answers.
- Memory store — an in-memory dict replaces Redis.
- Ingestion manager — `submit` records jobs instead of running them in a thread.

The `auth_headers` fixture registers and logs in a user so authenticated
endpoints can be tested.

## 5. What each test file verifies

### test_auth.py (10)
- **Password hashing** — hash/verify round-trip; wrong password rejected; hashes
  are salted (two hashes of the same password differ).
- **Access tokens** — JWT create/decode round-trip (`sub`, `username` claims);
  malformed tokens raise a 401 with `"Invalid or expired token"`.
- **Endpoints** — register → login → `/auth/me` flow; duplicate registration
  returns 409; wrong password returns 401; `/auth/me` requires a valid token.

### test_chunking.py (17)
The **word-span chunker** (`app/services/chunking.py`) that powers the
chunking-agnostic evaluation methodology:
- `parse_document` produces words with per-word page numbers, paragraph
  boundaries, and contiguous spans.
- `chunk_document` under `fixed` respects section (heading) boundaries, applies
  overlap, and errors on unknown strategies/invalid targets.
- `sentence` strategy never splits a sentence and falls back to word-splitting
  only for oversized sentences.
- `paragraph` strategy aligns chunks to paragraph boundaries and bounds
  oversized paragraphs.
- `span_overlap_fraction` — the relevance judge used by the chunking evaluation.

### test_document.py (9)
The production chunker (`app/services/document.py`):
- `chunk_text` splits with overlap and tracks page numbers.
- `chunk_markdown` tracks `[[PAGE n]]` markers, strips them from text, and keeps
  headings on the right page.
- `chunk_pdf` produces chunks with `highlight_boxes`; missing files raise
  `FileNotFoundError`.
- `_bibliography_cutoff` — trims trailing `References` sections (and returns the
  full length when no marker exists).

### test_embedding.py (2)
`EmbeddingService.get_embeddings` returns a flat list of floats using a fake
model.

### test_endpoints.py (5)
- `/search/keyword` and `/search/semantic` return the canned hits with correct
  field mapping.
- Full chat CRUD (create / list / get / delete) and per-user isolation (a second
  user gets 404 on another user's chat).
- `/health` reports `elasticsearch: true` under the mocked connection.

### test_es.py (10)
Against a **FakeClient** (no real ES):
- `parse_chunk_id`, `index_chunk` round-trip, `_format_results` field
  normalization.
- `get_chunk_window` / `get_chunk_window_entries` — neighbour ordering, page
  range attachment, missing-id → empty, bounds capping.
- `search_pages` — dedupes pages and sorts by page number.

### test_evaluation.py (19)
The metrics used in the paper's evaluation tables (`app/services/evaluation.py`):
- **p@k, r@k, MRR, nDCG@k, AP, MAP** — exact values on hand-computable inputs.
- **Bootstrap CI** — bounds ordering, determinism for a fixed seed, empty input.
- **Paired bootstrap diff** — detects a significant positive difference and
  covers zero for identical samples.
- **Dataset builders** — `sample_chunks` (count/seed determinism, noise and
  short-chunk exclusion) and `neighboring_chunk_ids` from `scripts/build_eval_dataset.py`;
  `evaluate_query` schema from `scripts/evaluate_retrieval.py`.

### test_ingest.py (10)
- Extension whitelist is exactly `{".pdf", ".md", ".txt"}`.
- Upload endpoint queues a job (recorded in the fake manager), writes the file
  to `NOTES_DIR` under a safe `paper_id`, and neutralizes `../` traversal
  filenames; disallowed extensions return 400; unknown job status → 404.
- Worker `_process_document` indexes chunks and updates the manifest; removes
  the file on failure; sets `pdf_path` for PDF uploads.

### test_llm.py (8)
- `_build_payload` builds the right message structure (system prompt, numbered
  sources with page ranges `(pp. 40-43)` vs `(p. 44)`).
- Query rewrite falls back to the raw question when the LLM is unreachable and
  uses chat history when reachable.
- `generate_response` returns a graceful error message on failure.
- **Research (vision) mode** — `stream_research_response` drives a tool-calling
  loop (`read_pages` → image → text answer), uses the vision model name, and
  returns nothing when there are no seed images.

### test_pdf.py (10)
- `_highlight_by_search` highlights only within a given page range, across a
  multi-page range, and returns 0 on no match.
- Manifest matching by id and by local-path stem.
- `_serve_bytes` returns `application/pdf` headers.
- Cache TTL expiry.
- End-to-end: serving a PDF with `?chunk_id=` produces an annotated (highlighted)
  PDF via a fake ES source.
- `render_page_images` renders pages to base64 PNG data URIs, clamps out-of-range
  pages, and returns `[]` for unknown papers.

### test_rag.py (10)
- `_build_sources` field mapping + defaults; `_build_context` uses the chunk
  window; `_group_by_document` spreads results across papers for broad questions.
- Citation linkification (`[1]` → `/api/v1/pdf/p9?chunk_id=...#page=2`) and page
  reference linkification (`(p. 12)` → `#page=12`).
- TOC built from notes markdown with page markers.
- Research-mode toggle depends on a vision model being configured.
- `/rag/ask` returns answer + sources + search query; falls back gracefully with
  no hits; `/rag/ask/stream` emits the expected SSE events (`search_query`,
  `sources`, `token`, `done`) and requires auth.

## 6. Abbreviations glossary

**RAG terms**
- **RAG** — Retrieval-Augmented Generation: fetch relevant passages first, then
  have an LLM answer grounded in them.
- **Chunk** — a contiguous, self-contained unit of text (a few hundred words)
  that a retriever ranks and an LLM reads.
- **Embedding / dense vector** — a fixed-size numeric vector representing the
  meaning of a text; nearby vectors = similar meaning (here 384-dim, BGE-small).
- **Query rewrite** — using the LLM to turn a follow-up question (with history)
  into a standalone query.
- **Window / context window** — the retrieved chunk plus its adjacent chunks
  from the same document, fed to the LLM for context.
- **Chunk window** — the neighbouring-chunk expansion used to build LLM context.
- **Ground truth / gold** — the known-correct answer (or relevant chunk) a
  retrieval system is measured against.
- **Relevance judgment** — a binary label saying whether a retrieved item is
  relevant to the query.

**Retrieval methods**
- **Keyword search** — lexical matching (Elasticsearch BM25-style); good for
  exact terms, bad for synonyms.
- **Semantic search** — dense embedding cosine similarity; good for meaning,
  bad for exact terms.
- **Hybrid search** — combines keyword + semantic scores.
- **Reranking** — after an initial top-k retrieval, a cross-encoder scores each
  candidate `(query, text)` pair again to refine the order (here
  `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **BM25** — a classic probabilistic lexical ranking function.
- **Cosine similarity** — dot product of two unit vectors; similarity of
  embeddings.

**Evaluation metrics** (binary relevance)
- **p@k** — Precision@k: fraction of the top-k retrieved that are relevant.
- **r@k** — Recall@k: fraction of all relevant items found in the top-k.
- **MRR** — Mean Reciprocal Rank: average of `1/rank` of the first relevant
  item; 1.0 if relevant is first.
- **nDCG@k** — Normalized Discounted Cumulative Gain: rewards relevant items
  early in the ranking, normalized to a perfect ranking.
- **AP / MAP** — Average Precision (mean of precision at each relevant rank) /
  Mean Average Precision across queries.
- **CI** — Confidence Interval: a range that is 95% likely to contain the true
  difference (via bootstrap resampling).
- **Bootstrap / resampling** — repeatedly re-sampling a metric's per-query
  values to estimate variability without assuming a distribution.
- **Paired bootstrap diff** — CI for the difference between two aligned
  pipelines; if the interval excludes zero, the difference is statistically
  significant.

**Systems / file formats**
- **ES / Elasticsearch** — the search engine / vector database.
- **LLM** — Large Language Model.
- **SSE** — Server-Sent Events: the streaming mechanism for `/rag/ask/stream`.
- **JWT** — JSON Web Token: the bearer token used for auth.
- **SQLite / aiosqlite** — the app's test database engine.
- **Redis** — the chat-history store (mocked in tests).
- **PDF / PyMuPDF (fitz)** — Portable Document Format; the `fitz` library
  renders/highlights pages.
- **TOC** — Table of Contents.
- **Markdown (MD)** — the plain-text notes format.
- **Data URI** — `data:image/png;base64,...` inline image encoding used for
  vision-model page images.
- **API** — Application Programming Interface (here FastAPI + OpenAPI).

**Test/infra abbreviations**
- **pytest** — the test runner; fixtures inject state; `monkeypatch` replaces
  attributes at runtime.
- **fixture** — a pytest function that provides setup/teardown to tests.
- **TestClient** — FastAPI's in-process HTTP test client (Starlette/httpx based).
- **CRUD** — Create, Read, Update, Delete (chat lifecycle).
- **HTTP status codes** — 200 OK, 201 Created, 204 No Content, 302 Redirect,
  400 Bad Request, 401 Unauthorized, 404 Not Found, 409 Conflict, 416 Range Not
  Satisfiable.
- **deprecation warning** — a notice that an installed package API is outdated
  (non-fatal; see §7).

**Subject/domain (from the notes library)**
- **DS** — Data Structures · **OS** — Operating Systems · **ML** — Machine
  Learning · **ToC** — Theory of Computation · **DB** — Databases · **Net** —
  Computer Networking · **Swe** — Software Engineering.

## 7. The one warning

```
StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is
deprecated; install `httpx2` instead.
```

This comes from the installed FastAPI/Starlette test stack and does **not**
affect test outcomes. It is purely a notice from a library, not from this code.

## 8. Related evaluation scripts (for the paper)

The unit tests exercise the *same pure metric functions* used by the experiment
runners, so the numbers you generate are auditable:

| Script | Purpose |
| --- | --- |
| `scripts/build_eval_dataset.py` | Builds the synthetic retrieval question set (`eval/dataset.jsonl`) |
| `scripts/evaluate_retrieval.py` | Compares keyword / semantic / hybrid / hybrid+rerank (MAP, MRR, p@k, r@k, nDCG@k) with bootstrap CIs |
| `scripts/build_chunk_eval_dataset.py` | Builds the chunking-agnostic question set over word spans |
| `scripts/evaluate_chunking.py` | Compares chunking strategies (fixed/sentence/paragraph) + optional LLM answer grading /5 |
| `scripts/index_library.py` | Indexes the notes into ES under a chosen strategy |

## 9. Reproduction

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest -v
```

The full verbose output of this run is available in the repo if you need it
(see `pytest_run.txt` next to this report if present; otherwise rerun the
command above).
