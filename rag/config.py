"""RAG pipeline settings, loaded from .env."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# absolute path so this still works no matter where you run from (manage.py, pytest, scripts/...)
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class RAGConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_prefix="RAG_",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM (Google Gemini, free tier) ---
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.0

    # --- Embeddings (local HuggingFace model, free, no API key) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- ChromaDB ---
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "documents"

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Retrieval ---
    retrieval_k: int = 4
    score_threshold: float = 0.3  # anything below this gets dropped so the LLM doesn't hallucinate off weak matches
