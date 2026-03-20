"""
Mini RAG System — FastAPI application entry point.

Run with:
    uvicorn main:app --reload --port 8000
"""

import logging
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config import settings
from limiter import limiter
from routers import documents, chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = pathlib.Path(__file__).parent.parent / "frontend"


# ── Application lifespan ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup; clean up on shutdown."""
    import asyncio
    import threading

    logger.info("Starting Mini RAG System…")
    app.state.vector_store = None

    def _init_vs():
        from services.vector_store import VectorStore
        vs = VectorStore()
        app.state.vector_store = vs
        logger.info("Vector store initialised.")

    thread = threading.Thread(target=_init_vs, daemon=True)
    thread.start()

    yield

    logger.info("Shutting down…")


# ── App factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "A production-ready Retrieval-Augmented Generation (RAG) API built with "
        "FastAPI, ChromaDB, SentenceTransformers, and Claude."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])


# ── Health & root ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check(request: Request):
    vs = request.app.state.vector_store
    return {
        "status": "ok" if vs else "loading",
        "model": settings.claude_model,
        "embedding_model": settings.embedding_model,
        "chunks_stored": vs.count() if vs else 0,
    }


# ── Serve frontend ─────────────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
