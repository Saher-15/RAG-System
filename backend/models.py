from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# ── Document models ────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    id: str
    text: str
    source: str
    chunk_index: int
    total_chunks: int


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_stored: int
    message: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    upload_date: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    document_id: str
    message: str


# ── Chat models ────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_history: List[ChatMessage] = Field(default_factory=list, max_length=50)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class RetrievedChunk(BaseModel):
    text: str
    source: str
    chunk_index: int
    distance: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[RetrievedChunk]
    model: str


# ── SSE event shapes (serialised as JSON lines) ────────────────────────────────

class SSEChunk(BaseModel):
    type: str  # "token" | "sources" | "done" | "error"
    content: Optional[str] = None
    sources: Optional[List[RetrievedChunk]] = None
    error: Optional[str] = None
