# Mini RAG System

A clean, production-ready Retrieval-Augmented Generation (RAG) system.

**Stack**
| Layer | Technology |
|---|---|
| AI / Generation | Python · Claude Opus 4.6 (streaming + adaptive thinking) |
| Embeddings | Python · SentenceTransformers `all-MiniLM-L6-v2` |
| Vector Store | Python · ChromaDB (persistent, cosine similarity) |
| API | Python · FastAPI + SSE streaming |
| Frontend | Vanilla JavaScript · single-page app |

---

## Architecture

```
┌─────────────┐    upload     ┌──────────────────────────────────────────┐
│             │ ─────────────▶│  Document Processor                      │
│  Browser    │               │  parse (txt/md/pdf) → chunk → embed      │
│  (JS SPA)   │               └──────────────┬───────────────────────────┘
│             │                              │ store chunks + metadata
│             │                              ▼
│             │               ┌──────────────────────────┐
│             │               │  ChromaDB Vector Store   │
│             │               │  (cosine · persistent)   │
│             │               └──────────────┬───────────┘
│             │  SSE stream                  │ top-k retrieval
│             │ ◀────────────────────────────┤
│             │               ┌──────────────▼───────────┐
│             │    question   │  RAG Pipeline             │
│             │ ─────────────▶│  retrieve → prompt build │
│             │               │  → Claude stream → SSE   │
└─────────────┘               └──────────────────────────┘
```

---

## Quick Start

### 1. Configure

```bash
cd rag-system
cp .env.example backend/.env
# Edit backend/.env and add your ANTHROPIC_API_KEY
```

### 2. Install & Run Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB). Subsequent starts are instant.

### 3. Open Frontend

```bash
# From the project root — serve with any static file server:
cd frontend
python -m http.server 5500
# Then open http://localhost:5500
```

Or simply open `frontend/index.html` directly in your browser.

---

## API Reference

Interactive docs: `http://localhost:8000/docs`

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/documents/upload` | Ingest a `.txt`, `.md`, or `.pdf` file |
| `GET` | `/api/documents` | List all documents in the knowledge base |
| `DELETE` | `/api/documents/{id}` | Remove a document and all its chunks |
| `POST` | `/api/chat/stream` | SSE streaming RAG query |
| `POST` | `/api/chat` | Non-streaming RAG query (full response) |
| `GET` | `/health` | Health check |

### Chat request body

```json
{
  "question": "What is the refund policy?",
  "conversation_history": [],
  "top_k": 5
}
```

### SSE event types

```
data: {"type":"sources","sources":[{"text":"...","source":"file.pdf","chunk_index":2,"distance":0.12}]}
data: {"type":"token","content":"Based on "}
data: {"type":"token","content":"[Source 1]…"}
data: {"type":"done"}
```

---

## Configuration

All settings are in `backend/.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `CLAUDE_MODEL` | `claude-opus-4-6` | Claude model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Chunks retrieved per query |

---

## Project Structure

```
rag-system/
├── backend/
│   ├── main.py                    # FastAPI app + lifespan
│   ├── config.py                  # Pydantic settings
│   ├── models.py                  # Request / response schemas
│   ├── services/
│   │   ├── vector_store.py        # ChromaDB wrapper
│   │   ├── document_processor.py  # Parse + chunk files
│   │   └── rag_pipeline.py        # Retrieve + stream with Claude
│   ├── routers/
│   │   ├── documents.py           # Upload / list / delete
│   │   └── chat.py                # SSE stream + sync endpoints
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── app.js                     # Vanilla JS SPA
│   └── style.css
├── data/chroma/                   # Persisted vector store (git-ignored)
└── .env.example
```
