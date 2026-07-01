"""Loads the embedding model - local HuggingFace, no API key needed."""

import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from rag.config import RAGConfig

logger = logging.getLogger(__name__)


def build_embeddings(config: RAGConfig) -> Embeddings:
    logger.info("Loading embedding model '%s'", config.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,  # cosine similarity only makes sense on normalized vectors
            "batch_size": 32,
        },
    )
