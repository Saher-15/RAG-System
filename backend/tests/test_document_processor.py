"""Tests for document processing and text splitting."""
import pytest
from services.document_processor import process_document, _split_text


# ── Text splitting ─────────────────────────────────────────────────────────────

def test_split_short_text_returns_single_chunk():
    chunks = _split_text("Hello world", chunk_size=512, chunk_overlap=64)
    assert chunks == ["Hello world"]


def test_split_empty_text_returns_empty():
    chunks = _split_text("", chunk_size=512, chunk_overlap=64)
    assert chunks == []


def test_split_long_text_produces_multiple_chunks():
    text = "word " * 300  # ~1500 chars
    chunks = _split_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200


def test_split_respects_overlap():
    text = "A" * 100 + " " + "B" * 100 + " " + "C" * 100
    chunks = _split_text(text, chunk_size=150, chunk_overlap=30)
    assert len(chunks) >= 2


def test_split_paragraph_boundary():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = _split_text(text, chunk_size=30, chunk_overlap=0)
    assert len(chunks) >= 2


# ── process_document ──────────────────────────────────────────────────────────

def test_process_txt_file():
    content = b"Hello, this is a test document with some content."
    doc_id, chunks, metadatas, ids = process_document(
        filename="test.txt", content=content, chunk_size=512, chunk_overlap=64
    )
    assert doc_id
    assert len(chunks) == 1
    assert chunks[0] == "Hello, this is a test document with some content."
    assert len(metadatas) == 1
    assert metadatas[0]["filename"] == "test.txt"
    assert metadatas[0]["chunk_index"] == 0
    assert metadatas[0]["total_chunks"] == 1
    assert len(ids) == 1
    assert ids[0] == f"{doc_id}__chunk_0"


def test_process_md_file():
    content = b"# Title\n\nSome markdown content."
    doc_id, chunks, metadatas, ids = process_document(
        filename="doc.md", content=content, chunk_size=512, chunk_overlap=64
    )
    assert len(chunks) >= 1
    assert metadatas[0]["filename"] == "doc.md"


def test_process_large_file_splits_into_chunks():
    content = ("word " * 200).encode()
    doc_id, chunks, metadatas, ids = process_document(
        filename="large.txt", content=content, chunk_size=100, chunk_overlap=10
    )
    assert len(chunks) > 1
    assert len(chunks) == len(metadatas) == len(ids)
    for i, meta in enumerate(metadatas):
        assert meta["chunk_index"] == i
        assert meta["total_chunks"] == len(chunks)


def test_process_same_filename_gives_same_doc_id():
    content = b"Some content"
    doc_id1, *_ = process_document("file.txt", content, 512, 64)
    doc_id2, *_ = process_document("file.txt", content, 512, 64)
    assert doc_id1 == doc_id2


def test_process_pdf_file():
    """Test PDF processing with a minimal valid PDF."""
    # Minimal PDF bytes
    minimal_pdf = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
190
%%EOF"""
    # Just check it doesn't crash (text may be empty for minimal PDF)
    doc_id, chunks, metadatas, ids = process_document(
        filename="test.pdf", content=minimal_pdf, chunk_size=512, chunk_overlap=64
    )
    assert doc_id  # should always return a valid id
