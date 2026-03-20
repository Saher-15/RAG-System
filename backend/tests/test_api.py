"""Integration tests for the FastAPI endpoints."""
import io
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_vs():
    vs = MagicMock()
    vs.count.return_value = 0
    vs.list_documents.return_value = {}
    vs.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    vs.add_chunks.return_value = None
    vs.delete_document.return_value = 1
    return vs


@pytest.fixture()
def client(mock_vs):
    """TestClient with VectorStore patched so the background thread uses the mock."""
    with patch("services.vector_store.VectorStore", return_value=mock_vs):
        from main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            # Ensure state is set even if background thread is slow
            app.state.vector_store = mock_vs
            yield c, mock_vs


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_check_ok(client):
    c, vs = client
    resp = c.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "loading")
    assert "model" in data
    assert "embedding_model" in data


def test_health_check_loading():
    """Returns 'loading' when vector store is None."""
    with patch("services.vector_store.VectorStore", side_effect=Exception("skip")):
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.vector_store = None
            resp = c.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "loading"


# ── Documents ─────────────────────────────────────────────────────────────────

def test_upload_txt_file(client):
    c, vs = client
    content = b"This is a test document with enough content to process."
    resp = c.post(
        "/api/documents/upload",
        files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["filename"] == "test.txt"
    assert data["chunks_stored"] >= 1
    assert "document_id" in data


def test_upload_md_file(client):
    c, vs = client
    content = b"# Title\n\nSome markdown content here."
    resp = c.post(
        "/api/documents/upload",
        files={"file": ("doc.md", io.BytesIO(content), "text/markdown")},
    )
    assert resp.status_code == 201


def test_upload_unsupported_file_type(client):
    c, _ = client
    resp = c.post(
        "/api/documents/upload",
        files={"file": ("test.exe", io.BytesIO(b"data"), "application/octet-stream")},
    )
    assert resp.status_code == 415


def test_upload_empty_file(client):
    c, _ = client
    resp = c.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert resp.status_code == 400


def test_list_documents_empty(client):
    c, vs = client
    vs.list_documents.return_value = {}
    resp = c.get("/api/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["documents"] == []


def test_list_documents_with_data(client):
    c, vs = client
    vs.list_documents.return_value = {
        "doc-123": {
            "filename": "test.txt",
            "chunk_count": 3,
            "upload_date": "2026-01-01T00:00:00+00:00",
        }
    }
    resp = c.get("/api/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["documents"][0]["filename"] == "test.txt"


def test_delete_document(client):
    c, vs = client
    vs.delete_document.return_value = 3
    resp = c.delete("/api/documents/doc-123")
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == "doc-123"


def test_delete_nonexistent_document(client):
    c, vs = client
    vs.delete_document.return_value = 0
    resp = c.delete("/api/documents/nonexistent")
    assert resp.status_code == 404


# ── Chat ──────────────────────────────────────────────────────────────────────

def test_chat_no_relevant_docs(client):
    """When no relevant chunks found, returns a fallback message."""
    c, vs = client
    vs.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    resp = c.post("/api/chat", json={"question": "What is the meaning of life?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "couldn't find" in data["answer"].lower()


def test_chat_question_too_long(client):
    c, _ = client
    resp = c.post("/api/chat", json={"question": "x" * 2001})
    assert resp.status_code == 422


def test_chat_empty_question(client):
    c, _ = client
    resp = c.post("/api/chat", json={"question": ""})
    assert resp.status_code == 422


def test_chat_stream_returns_sse(client):
    """Streaming endpoint should return text/event-stream content type."""
    c, _ = client
    resp = c.post(
        "/api/chat/stream",
        json={"question": "test question"},
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
