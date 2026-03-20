from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # API
    api_title: str = "Mini RAG System"
    api_version: str = "1.0.0"

    # Claude
    anthropic_api_key: str
    claude_model: str = "claude-opus-4-6"

    # Embeddings (via ChromaDB's built-in ONNX embedding function)
    embedding_model: str = "all-MiniLM-L6-v2"  # used in health check display only

    # Vector Store
    chroma_persist_dir: str = "./data/chroma"
    collection_name: str = "documents"

    # Document Processing
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 5
    # Filter out chunks with cosine distance above this threshold (0=identical, 1=unrelated)
    relevance_threshold: float = 0.75

    # Conversation history — cap to last N turns to avoid token overflow
    max_history_turns: int = 10

    # Rate limiting
    upload_rate_limit: str = "20/minute"
    chat_rate_limit: str = "30/minute"

    # CORS
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5500",
        "null",  # file:// protocol for local HTML opening
    ]


settings = Settings()
