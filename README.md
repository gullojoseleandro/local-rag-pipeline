# Local semantic RAG pipeline

A Python 3.12 example that indexes local Markdown/text notes in ChromaDB and answers questions with an asynchronous LangChain LCEL chain. The example knowledge base documents a FastAPI service, Redis cache, and PostgreSQL database.

## Quick start

You need Python 3.12 and an OpenAI API key. Indexing and live questions use the API and may incur charges. The offline unit tests do not.

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`. Never commit this file. On macOS/Linux, use `python3.12 -m venv .venv`, `source .venv/bin/activate`, and `cp .env.example .env`.

```powershell
python ingest.py
python main.py "How long are order summaries cached in Redis?"
python main.py "Which city hosts the API server?"
python smoke_test.py
```

The first question should mention **60 seconds** and cite `cache.md`. The second should return `{"texto": "No lo sé", "referencias": []}`. Actual model behavior is probabilistic; `smoke_test.py` checks these expectations against the live API and fails if either is violated.

To run the offline tests, no key is needed:

```powershell
python -m unittest discover -s tests -v
```

## How it works

- `ingest.py` reads `.md` and `.txt` files under `data/` recursively. It uses a token-aware `RecursiveCharacterTextSplitter` with a **500-token maximum chunk size** and **50-token overlap**. Each chunk has a stable content-derived ID. Re-running ingestion skips unchanged chunks and removes stale chunks after new ones have been indexed.
- `store.py` persists Chroma in `vectorstore/` and records the embedding model in collection metadata. Both indexing and retrieval use `text-embedding-3-small`. Attempting to open the index with another model raises an error rather than returning misleading results.
- `rag.py` retrieves the top **4** chunks, constructs a context-only prompt, calls `ChatOpenAI` via async `.ainvoke()`, parses the JSON with `PydanticOutputParser`, and rejects citations to unretrieved files. `schemas.py` validates the output.
- `main.py` is the query CLI. `smoke_test.py` performs the two live acceptance checks. `tests/` exercises ingestion, persistence, idempotency, answer parsing, abstention, and citation validation with local test doubles.

Example output for a documented question:

```json
{
  "texto": "Redis caches order summaries for 60 seconds.",
  "referencias": ["cache.md"]
}
```

The prompt tells the model to answer only from retrieved context. A citation check helps detect unsupported source names, but it cannot prove that every sentence is factually grounded. Inspect live results and improve the corpus or retrieval strategy if the smoke test fails.

## Configuration

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required for live embeddings and chat completions |
| `OPENAI_CHAT_MODEL` | Optional chat model; default `gpt-4o-mini` |
| `RAG_DB_PATH` | Optional persisted index path for queries; default `vectorstore` |

Use `python ingest.py --data path/to/notes --db path/to/index` for other documents. If you change the embedding model, build a new index; mixing embeddings in one collection invalidates semantic search. The `vectorstore/` directory and `.env` are ignored by Git.
