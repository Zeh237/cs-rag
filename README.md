# CSC RAG Engine

This project answers questions about undergraduate computer science notes.
It uses retrieval-augmented generation (RAG).
It stores the notes as searchable chunks.
You ask a question.
The system retrieves the relevant chunks.
The system generates an answer from those chunks.

## Features

- Keyword search across the notes.
- Semantic search across the notes.
- Hybrid search that combines keyword and semantic retrieval.
- A PDF viewer that highlights the source page of each result.
- A chat interface with streaming answers.
- A web user interface for search, chat, documents, and ingestion.

## Requirements

- Python 3.11 or newer.
- Elasticsearch 8.x. The system requires a running instance.
- Redis. The system uses Redis for chat history. Redis is optional if you do not use the chat history.
- A local OpenAI-compatible LLM server. The system uses this server to generate answers. Examples are LM Studio and Ollama. The LLM is optional if you only search.

## Setup

1. Copy the example configuration:

   ```bash
   cp .env.example .env
   ```

2. Edit the file `.env`. Set the connection data for Elasticsearch, Redis, and the LLM. Refer to the section "Configuration".

3. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

4. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Start Elasticsearch, Redis, and the local LLM server. Refer to their documentation.

6. Start the app:

   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

7. Open the web interface in your browser:

   ```
   http://localhost:8000
   ```

   The interactive API documentation is at `http://localhost:8000/docs`.

## Build the notes library

The system does not contain notes by default. Use the scripts to fetch and index the notes.

1. Download the notes from the configured sources:

   ```bash
   python scripts/fetch_notes.py
   ```

   This script downloads course notes from public GitHub repositories and PDF books. It writes the notes in the directory `NOTES_DIR` (default `./csc_notes`).

2. Index the notes into Elasticsearch:

   ```bash
   python scripts/index_library.py
   ```

   This script converts each file into chunks and stores the chunks in the index `INDEX_NAME`.

3. (Optional) Add papers to the legacy library:

   ```bash
   python scripts/download_papers.py
   ```

   This script downloads arXiv papers. It writes the papers in the directory `LIBRARY_DIR` (default `./csc_library`).

## Configuration

The app reads all configuration from environment variables. The file `.env.example` lists every variable with its default value. Copy that file to `.env` and adjust the values.

Key variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `ES_HOST` | The URL of the Elasticsearch service | `http://localhost:9200` |
| `ES_USER` | The Elasticsearch user name | `elastic` |
| `ES_PASSWORD` | The Elasticsearch password | `password` |
| `INDEX_NAME` | The name of the search index | `csc_notes` |
| `NOTES_DIR` | The directory that contains the notes | `./csc_notes` |
| `LIBRARY_DIR` | The directory that contains the papers | `./csc_library` |
| `LLM_API_BASE` | The base URL of the local LLM | empty |
| `LLM_MODEL` | The name of the LLM | empty |
| `EMBEDDING_MODEL` | The sentence-transformer model for embeddings | `BAAI/bge-small-en-v1.5` |
| `REDIS_HOST` | The Redis connection URL | `redis://localhost:6379/` |
| `REDIS_PASSWORD` | The Redis password | empty |
| `SECRET_KEY` | The signing key for access tokens | `change-me-in-production` |
| `DATABASE_URL` | The SQLAlchemy database URL | `sqlite+aiosqlite:///./rag.db` |

The `EMBEDDING_MODEL` must output vectors with 384 dimensions. The mapping `app/services/elasticsearch.py` uses that dimension. Use a model with the same dimension.

## Add a document

You can add a single document through the web interface or through the API:

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@/path/to/document.pdf" \
  -F "title=My Document" \
  -F "authors=Jane Doe" \
  -F "primary_category=Algorithms" \
  -F "all_categories=Algorithms"
```

The endpoint accepts `.pdf`, `.md`, and `.txt` files. The system processes the file in the background. You can query the progress:

```bash
curl http://localhost:8000/api/v1/ingest/status/JOB_ID
```

## API reference

The app uses a bearer token for most requests. Register a user and log in to get the token.

| Method | URI | Purpose |
| --- | --- | --- |
| `POST` | `/auth/register` | Register a user |
| `POST` | `/auth/login` | Log in and get a token |
| `GET` | `/auth/me` | Return the current user |
| `POST` | `/chats` | Create a chat |
| `GET` | `/chats` | List the chats |
| `GET` | `/chats/{chat_id}` | Show a chat with its history |
| `DELETE` | `/chats/{chat_id}` | Delete a chat |
| `POST` | `/api/v1/ingest` | Upload a document for indexing |
| `GET` | `/api/v1/ingest/status/{job_id}` | Get the status of an ingest job |
| `GET` | `/api/v1/pdf/{paper_id}` | Open a PDF; use `?chunk_id=...` to highlight a chunk |
| `GET` | `/api/v1/library` | List the legacy paper library |
| `GET` | `/search/keyword` | Keyword search (`?q=...&limit=5`) |
| `GET` | `/search/semantic` | Semantic search (`?q=...&limit=5`) |
| `GET` | `/rag/ask` | Ask a question (`?q=...&limit=2`) |
| `GET` | `/rag/ask/stream` | Ask a question and stream the answer (`?q=...&chat_id=...`) |
| `GET` | `/health` | Return the health of the services |

## Tests

Run the automated tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The tests use a temporary database and a temporary notes directory. The tests do not need a running Elasticsearch or Redis instance.

## Retrieval evaluation

Two scripts build and run retrieval evaluations. The setup uses the notes chunked in the `csc_notes` index.

First build a synthetic question dataset. The script picks representative chunks and, when a local LLM server is available, asks the LLM to write a question answerable from each chunk (the answer chunk is the ground truth). If the LLM is unavailable, it uses heuristic templates. Neighboring chunks of the same source are also marked relevant:

```bash
python scripts/build_eval_dataset.py --out eval/dataset.jsonl --count 120 --seed 42
```

Then run the four retrieval pipelines (keyword, semantic, hybrid, hybrid + rerank) over the dataset and compare them:

```bash
python scripts/evaluate_retrieval.py --dataset eval/dataset.jsonl --quick 0
```

Use `--quick N` to run a sanity check on the first `N` queries, and `--include-neighbors` to switch between judging against the answer chunk only or the answer chunk plus its neighbors. The script prints metrics (`map`, `mrr`, `p@k`, `ndcg@k`, `r@k`), Per-query bootstrap-resampled 95% confidence intervals for the difference against the best method, and per-query results in `eval/results_queries.csv`.

## Directory layout

```
app/api/            HTTP routes
app/services/       Business logic (search, RAG, PDF, embeddings, auth)
app/db/             Database models and session
app/static/         Web interface
scripts/            Notes fetching, indexing, and paper download
eval/               Synthetic retrieval datasets and evaluation results
tests/              Automatic tests
csc_notes/          The notes and their manifest
csc_library/        The papers and their manifest
```