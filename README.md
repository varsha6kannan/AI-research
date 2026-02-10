# DynamicMedRAG (Medical Dynamic RAG)

FastAPI-based dynamic Retrieval-Augmented Generation (RAG) for medical documents.

## What it does

- Watches `datasets/` for file create/modify/delete
- Generates a webhook-like JSON payload on every change
- Ingests documents into a persistent Chroma vector DB with:
  - semantic chunking
  - per-chunk SHA256 hash metadata for deduplication
  - MedCPT embeddings (`ncbi/MedCPT-Article-Encoder`)
- Serves a `/query` API that:
  - embeds the question with MedCPT query encoder (`ncbi/MedCPT-Query-Encoder`)
  - retrieves top‑K from Chroma
  - re-ranks top‑3 with MedCPT cross-encoder (`ncbi/MedCPT-Cross-Encoder`)
  - calls an OpenAI GPT‑4‑class model and returns JSON-only output

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment variables / .env

You can either export environment variables in your shell **or** create a `.env`
file in the project root (loaded automatically via `python-dotenv` and
`app/config.py`):

Example `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
DATASETS_DIR=datasets
CHROMA_PERSIST_DIR=chroma_db
CHROMA_COLLECTION=medical_chunks
TOP_K=10
WEBHOOK_URL=
```

## Run

```bash
uvicorn app.api.main:app --reload
```

Then:
- Open `GET /health`
- Use `POST /query`
- Drop or edit files inside `datasets/` to trigger ingestion

## Gradio UI

A single-page Gradio interface is available for the RAG assistant. It calls the FastAPI `POST /query` API and handles clarification (disambiguation) by rewriting the question with the selected intent when you click **Continue**.

1. Start the backend first (see **Run** above).
2. In another terminal, from the project root:

   ```bash
   python run_gradio.py
   ```

3. Open the URL shown (default: http://localhost:7860).

You can set `BACKEND_URL` (e.g. `http://localhost:8000`) if the API is not on the same host. The UI shows:
- **Ask** to send your question; if the backend asks for clarification, choose an option and click **Continue**.
- **Answer** and **Citations** after a response.

## Supported document formats

- `.json`: `{ "title": "...", "content": "...", "pmid": "..." }`
- `.jsonl`: one object per line (same keys)
- `.txt` / `.md`: title from first header/line; `PMID: <digits>` is auto-detected (optional)
- `.pdf`: text extracted via PyMuPDF

## Testing

From the project root, run the test script (no pytest required):

```bash
python tests/test_app.py
```

With pytest installed: `pytest tests/ -v`

Tests cover: config/watch path, document loader, webhook payload, Chroma store, and (if the server is running) `GET /health`.

## Notes

- First run will download Hugging Face models (MedCPT) and may take a while.
- By default it runs on CPU; CUDA is used if available.

