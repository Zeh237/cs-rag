# CSC RAG Engine

This project answers questions about undergraduate computer science notes using retrieval-augmented generation (RAG). Notes are stored as searchable chunks in Elasticsearch.

## Features

- Keyword search across the notes.
- Semantic search across the notes.
- Hybrid search that combines keyword and semantic retrieval.
- Answers cite exact pages with clickable links that open the PDF at the cited position and highlight the chunk.
- Broad topic questions retrieve context from multiple documents (grouped by note) so answers cover whole subjects, not single chapters.
- PDF viewer that highlights the source page of each result.
- Chat interface with streaming answers.
- Web user interface for search, chat, documents, and ingestion.

## Requirements

- Python 3.11 or newer.
- Elasticsearch 8.x. The system requires a running instance.
- Redis. The system uses Redis for chat history; Redis is optional if chat history is not used.
- A local LLM server compatible with the OpenAI API. The system uses this server to generate answers. Examples are LM Studio and Ollama. The LLM is optional if only search is needed.

## Setup

1. Copy the example configuration:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env`. Set the connection data for Elasticsearch, Redis, and the LLM. Refer to the "Configuration" section.

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

   This script downloads open course notes as PDFs, stores each PDF locally in `csc_notes/pdfs/`, and normalizes its text into a Markdown file. Every note therefore has a local PDF backing it.

2. Index the notes into Elasticsearch:

   ```bash
   python scripts/index_library.py
   ```

   This script converts each file into chunks and stores the chunks in the index `INDEX_NAME`. Only notes with a local PDF are indexed (pass `--no-require-pdf` to include PDF-less sources).

## Configuration

The app reads all configuration from environment variables. The file `.env.example` lists every variable with its default value. Copy that file to `.env` and adjust the values.

Key variables:

| Variable | Purpose | Default |
|---|---|---|
| `ES_HOST` | The URL of the Elasticsearch service | `http://localhost:9200` |
| `ES_USER` | The Elasticsearch user name | `elastic` |
| `ES_PASSWORD` | The Elasticsearch password | `password` |
| `INDEX_NAME` | The name of the search index | `csc_notes` |
| `NOTES_DIR` | The directory that contains the notes | `./csc_notes` |
| `LLM_API_BASE` | The base URL of the local LLM | empty |
| `LLM_MODEL` | The name of the LLM | empty |
| `EMBEDDING_MODEL` | The sentence-transformer model for embeddings | `google/embeddinggemma-2` |
| `REDIS_HOST` | The Redis connection URL | `redis://localhost:6379/` |
| `REDIS_PASSWORD` | The Redis password | empty |
| `SECRET_KEY` | The signing key for access tokens | `change-me-in-production` |
| `DATABASE_URL` | The SQLAlchemy database URL | `sqlite+aiosqlite:///./rag.db` |

The `EMBEDDING_MODEL` must output vectors with 768 dimensions. The mapping `app/services/elasticsearch.py` uses that dimension. Use a model with the same dimension. The default `google/embeddinggemma-2` loads the text-only encoder setup (270M parameters) and encodes queries with the `SearchQuery` prompt and documents with the `Document` prompt, per the EmbeddingGemma 2 developer guide.

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
|---|---|---|
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
| `GET` | `/search/keyword` | Keyword search (`?q=...&limit=5`) |
| `GET` | `/search/semantic` | Semantic search (`?q=...&limit=5`) |
| `GET` | `/rag/ask` | Ask a question (`?q=...&limit=4&window_size=2`) |
| `GET` | `/rag/ask/stream` | Ask a question and stream the answer (`?q=...&chat_id=...&limit=4&window_size=2`) |
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

Use `--quick N` to run a sanity check on the first `N` queries, and `--include-neighbors` to switch between judging against the answer chunk only or the answer chunk plus its neighbors. The script prints metrics (`map`, `mrr`, `p@k`, `ndcg@k`, `r@k`), per-query bootstrap-resampled 95% confidence intervals for the difference against the best method, and per-query results in `eval/results_queries.csv`.

## Chunking-strategy evaluation

The system supports five chunking strategies: fixed, recursive, semantic, document, and hierarchical. Each strategy produces chunks with word-span metadata, so the same question set can be evaluated against every strategy without re-annotating.

### Strategies

- **fixed** — Chunks of exactly `chunk-size` words with `overlap` word overlap between consecutive chunks. No section awareness.
- **recursive** — RecursiveCharacterTextSplitter from LangChain. Splits on whitespace boundaries; does not guarantee sentence boundaries.
- **semantic** — LlamaIndex SemanticChunker with `all-MiniLM-L6-v2` embeddings. Produces variable-sized chunks based on topic coherence.
- **document** — Splits respect section boundaries (`#`-headers in Markdown). Each chunk stays within a single document section.
- **hierarchical** — LlamaIndex HierarchicalNodeParser. Produces a coarse-to-fine hierarchy: parent chunks at 2× target size and child chunks at target size, linked by parent/child relationships.

### Comparison script

Run the comparison on a markdown file or on notes from `csc_notes/`:

```bash
# Compare on the VLM markdown output from the PDF comparison:
python eval/compare_chunk_strategies.py --file eval/method_A__VLM_(cached).md --target 300 --overlap 45

# Compare on a sample of notes (default: 1 small, 1 medium, 1 large):
python eval/compare_chunk_strategies.py --target 300 --overlap 45
```

Outputs are saved to `eval/`:

- `chunk_strategy_<strategy>.md` — full chunk listing per strategy for human inspection
- `chunk_comparison_results.json` — raw per-note and aggregate data
- `chunk_comparison_report.md` — markdown table and metric explanations

### Indexing the strategies

Index the same notes under each strategy, then evaluate retrieval quality:

```bash
python scripts/index_library.py --index csc_notes_fixed300 \
    --strategy fixed --chunk-size 300 --overlap 45 --only-md

python scripts/index_library.py --index csc_notes_recursive300 \
    --strategy recursive --chunk-size 300 --overlap 45 --only-md

python scripts/index_library.py --index csc_notes_semantic300 \
    --strategy semantic --chunk-size 300 --overlap 45 --only-md

python scripts/index_library.py --index csc_notes_document300 \
    --strategy document --chunk-size 300 --overlap 45 --only-md

python scripts/index_library.py --index csc_notes_hierarchical300 \
    --strategy hierarchical --chunk-size 300 --overlap 45 --only-md
```

Then evaluate:

```bash
python scripts/evaluate_chunking.py \
    --indexes csc_notes_fixed300,csc_notes_recursive300,csc_notes_semantic300,csc_notes_document300,csc_notes_hierarchical300 \
    --dataset eval/chunk_dataset_v2.jsonl --overlap-ratio 0.5
```

A retrieved chunk is judged relevant when its stored word span overlaps the gold answer paragraph by at least `--overlap-ratio`. The report also prints the "single-chunk containment" share — how often a whole paragraph fits in one chunk per strategy — which is the structural limit on answer quality.

## Directory layout

```
app/api/            HTTP routes
app/services/       Business logic (search, RAG, PDF, embeddings, auth)
app/db/             Database models and session
app/static/         Web interface
scripts/            Notes fetching, indexing, and paper download
eval/               Synthetic retrieval datasets and evaluation results
tests/              Automatic tests
csc_notes/          The notes, their local PDFs, and the manifest
```