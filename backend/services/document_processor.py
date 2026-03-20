"""
Document ingestion: parse supported file types, split into overlapping chunks.

Supported formats: .txt, .md, .pdf
"""

import re
import io
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Tuple

logger = logging.getLogger(__name__)


# ── Text splitter ──────────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Recursive character splitter.  Tries to break on paragraph → sentence →
    word boundaries before hard-splitting on character count.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    return _recursive_split(text.strip(), separators, chunk_size, chunk_overlap)


def _recursive_split(
    text: str, separators: List[str], chunk_size: int, overlap: int
) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0] if separators else ""
    rest = separators[1:] if len(separators) > 1 else []

    if sep == "":
        # Hard split
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    parts = text.split(sep)
    chunks: List[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part).strip() if current else part.strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                # Start the next chunk with overlap from the end of `current`
                overlap_text = current[-overlap:] if overlap else ""
                current = (overlap_text + sep + part).strip() if overlap_text else part.strip()
            else:
                # `part` alone is too large — recurse with a finer separator
                chunks.extend(_recursive_split(part, rest, chunk_size, overlap))
                current = ""

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


# ── File parsers ───────────────────────────────────────────────────────────────

def _parse_text(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")


def _parse_pdf(content: bytes) -> str:
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required for PDF support. Run: pip install pypdf"
        ) from exc


_PARSERS = {
    ".txt": _parse_text,
    ".md": _parse_text,
    ".pdf": _parse_pdf,
}


# ── Public API ─────────────────────────────────────────────────────────────────

def process_document(
    filename: str,
    content: bytes,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[str, List[str], List[dict], List[str]]:
    """
    Parse *content* and split into chunks.

    Returns
    -------
    document_id : str
        Stable UUID for this document derived from its name.
    chunks : list[str]
        List of text chunks.
    metadatas : list[dict]
        Parallel list of metadata dicts (one per chunk).
    ids : list[str]
        Parallel list of unique chunk IDs suitable for ChromaDB.
    """
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    parser = _PARSERS.get(suffix, _parse_text)

    raw_text = parser(content)
    raw_text = re.sub(r"\s{3,}", "\n\n", raw_text)  # collapse excessive whitespace

    document_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))
    chunks = _split_text(raw_text, chunk_size, chunk_overlap)
    total = len(chunks)
    upload_date = datetime.now(timezone.utc).isoformat()

    metadatas = [
        {
            "document_id": document_id,
            "filename": filename,
            "chunk_index": i,
            "total_chunks": total,
            "upload_date": upload_date,
        }
        for i in range(total)
    ]

    ids = [f"{document_id}__chunk_{i}" for i in range(total)]

    logger.info(
        "Processed '%s' → %d chunks (size=%d, overlap=%d)",
        filename,
        total,
        chunk_size,
        chunk_overlap,
    )
    return document_id, chunks, metadatas, ids
