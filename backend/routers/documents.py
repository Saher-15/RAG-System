"""
Document management endpoints:
  POST   /api/documents/upload   — ingest a file into the vector store
  GET    /api/documents          — list all ingested documents
  DELETE /api/documents/{id}     — remove a document and all its chunks
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from config import settings
from limiter import limiter
from models import DocumentUploadResponse, DocumentListResponse, DocumentInfo, DeleteResponse
from services.document_processor import process_document

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


def _vector_store(request: Request):
    vs = request.app.state.vector_store
    if vs is None:
        raise HTTPException(status_code=503, detail="Vector store is still loading, please retry in a moment.")
    return vs


# ── Upload ─────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
@limiter.limit(settings.upload_rate_limit)
async def upload_document(request: Request, file: UploadFile = File(...)):
    filename = file.filename or "upload"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) // 1024} KB). Max is 20 MB.",
        )

    try:
        document_id, chunks, metadatas, ids = process_document(
            filename=filename,
            content=content,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if not chunks:
        raise HTTPException(status_code=422, detail="Could not extract any text from the file.")

    vs = _vector_store(request)
    vs.add_chunks(chunks=chunks, metadatas=metadatas, ids=ids)

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        chunks_stored=len(chunks),
        message=f"'{filename}' ingested successfully ({len(chunks)} chunks).",
    )


# ── List ───────────────────────────────────────────────────────────────────────

@router.get("", response_model=DocumentListResponse)
async def list_documents(request: Request):
    vs = _vector_store(request)
    docs = vs.list_documents()

    return DocumentListResponse(
        documents=[
            DocumentInfo(
                document_id=doc_id,
                filename=info["filename"],
                chunk_count=info["chunk_count"],
                upload_date=info.get("upload_date"),
            )
            for doc_id, info in docs.items()
        ],
        total=len(docs),
    )


# ── Delete ─────────────────────────────────────────────────────────────────────

@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str, request: Request):
    vs = _vector_store(request)
    deleted = vs.delete_document(document_id)

    if deleted == 0:
        raise HTTPException(
            status_code=404, detail=f"Document '{document_id}' not found."
        )

    return DeleteResponse(
        document_id=document_id,
        message=f"Deleted {deleted} chunks for document '{document_id}'.",
    )
