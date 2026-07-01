"""Builds the embedding model used for indexing and retrieval."""

import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from rag.config import RAGConfig

logger = logging.getLogger(__name__)


def build_embeddings(config: RAGConfig) -> Embeddings:
    """Load the local HuggingFace embedding model (free, offline, CPU-only)."""
    logger.info("Loading embedding model '%s'", config.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,  # required for cosine similarity to be valid
            "batch_size": 32,
        },
    )
