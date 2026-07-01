"""Central settings for the RAG pipeline, loaded from .env via pydantic-settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path so config loads correctly regardless of the CWD the entry
# point (manage.py, pytest, scripts/) was launched from.
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
    score_threshold: float = 0.3  # below this, drop the chunk to avoid hallucinations
