"""
ChromaDB vector store with SentenceTransformer embeddings.
Handles document storage, retrieval, and management.
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import logging

from config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

        self._ef = embedding_functions.DefaultEmbeddingFunction()

        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "VectorStore ready — collection '%s', %d chunks already stored",
            settings.collection_name,
            self._collection.count(),
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        """Upsert text chunks so re-ingesting the same file is idempotent."""
        self._collection.upsert(documents=chunks, metadatas=metadatas, ids=ids)
        logger.info("Upserted %d chunks into vector store", len(chunks))

    # ── Read ───────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return the top-n most similar chunks for *query_text*."""
        kwargs: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(n_results, max(self._collection.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        return self._collection.query(**kwargs)

    def count(self) -> int:
        """Return total number of chunks stored."""
        return self._collection.count()

    def list_documents(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping of document_id → {filename, chunk_count, upload_date}.
        ChromaDB stores each chunk with a document_id metadata field; here we
        aggregate across all chunks to produce per-document stats.
        """
        if self._collection.count() == 0:
            return {}

        all_items = self._collection.get(include=["metadatas"])
        docs: Dict[str, Dict[str, Any]] = {}

        for meta in all_items.get("metadatas", []):
            doc_id = meta.get("document_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {
                    "filename": meta.get("filename", "unknown"),
                    "chunk_count": 0,
                    "upload_date": meta.get("upload_date"),
                }
            docs[doc_id]["chunk_count"] += 1

        return docs

    # ── Delete ─────────────────────────────────────────────────────────────────

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks that belong to *document_id*. Returns # deleted."""
        result = self._collection.get(
            where={"document_id": document_id}, include=["metadatas"]
        )
        ids_to_delete = result.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        logger.info(
            "Deleted %d chunks for document '%s'", len(ids_to_delete), document_id
        )
        return len(ids_to_delete)
