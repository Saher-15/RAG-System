"""Tests for Pydantic models validation."""
import pytest
from pydantic import ValidationError
from models import ChatRequest, ChatMessage, DocumentUploadResponse


def test_chat_request_valid():
    req = ChatRequest(question="What is RAG?")
    assert req.question == "What is RAG?"
    assert req.conversation_history == []


def test_chat_request_empty_question_fails():
    with pytest.raises(ValidationError):
        ChatRequest(question="")


def test_chat_request_too_long_fails():
    with pytest.raises(ValidationError):
        ChatRequest(question="x" * 2001)


def test_chat_request_with_history():
    req = ChatRequest(
        question="Follow-up question",
        conversation_history=[
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
        ],
    )
    assert len(req.conversation_history) == 2


def test_chat_request_top_k_bounds():
    with pytest.raises(ValidationError):
        ChatRequest(question="test", top_k=0)
    with pytest.raises(ValidationError):
        ChatRequest(question="test", top_k=21)
    req = ChatRequest(question="test", top_k=5)
    assert req.top_k == 5


def test_document_upload_response():
    resp = DocumentUploadResponse(
        document_id="abc-123",
        filename="test.txt",
        chunks_stored=5,
        message="Success",
    )
    assert resp.chunks_stored == 5
