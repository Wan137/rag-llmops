"""Thin wrapper around ChromaDB + LangChain.

Keeping this as its own class (rather than calling langchain_chroma.Chroma
directly everywhere) means swapping ChromaDB for another vector DB later is a
one-file change.
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from rag.config import RAGConfig

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Persistent ChromaDB vector store with a LangChain-compatible interface.

    Example:
        config = RAGConfig()
        embeddings = build_embeddings(config)
        store = ChromaVectorStore(config, embeddings=embeddings)

        store.add_documents(my_chunks)
        results = store.similarity_search("What is RAG?", k=3)
        retriever = store.as_retriever(k=5, score_threshold=0.4)
    """

    def __init__(self, config: RAGConfig, embeddings: Embeddings) -> None:
        self.config = config
        self._embeddings = embeddings
        self._client: chromadb.ClientAPI = self._build_client()
        self._store: Chroma = self._build_store()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _build_client(self) -> chromadb.ClientAPI:
        """PersistentClient writes to disk so vectors survive a restart."""
        persist_dir = Path(self.config.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ChromaDB persist directory: %s", persist_dir.resolve())

        return chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    def _build_store(self) -> Chroma:
        return Chroma(
            client=self._client,
            collection_name=self.config.collection_name,
            embedding_function=self._embeddings,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Embed and store a list of chunks. Returns the assigned Chroma IDs."""
        if not documents:
            logger.warning("add_documents called with an empty list")
            return []

        logger.info(
            "Indexing %d chunks into collection '%s'",
            len(documents),
            self.config.collection_name,
        )
        ids = self._store.add_documents(documents)
        logger.info("Stored %d vectors", len(ids))
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[Document]:
        """Return the top-k chunks most similar to the query."""
        logger.debug("Similarity search: query='%s', k=%d", query, k)
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        """Same as similarity_search, but also returns each chunk's relevance score (0-1)."""
        return self._store.similarity_search_with_relevance_scores(query, k=k)

    def as_retriever(
        self,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> VectorStoreRetriever:
        """Return a LangChain Retriever for use in an LCEL chain.

        score_threshold drops low-relevance chunks before they reach the LLM,
        which is the main guard against hallucinated answers.
        """
        k = k or self.config.retrieval_k
        threshold = score_threshold or self.config.score_threshold

        search_kwargs: dict = {"k": k}
        search_type = "similarity"

        if threshold is not None:
            search_kwargs["score_threshold"] = threshold
            search_type = "similarity_score_threshold"

        logger.debug(
            "Building retriever: k=%d, search_type='%s', threshold=%s",
            k, search_type, threshold,
        )

        return self._store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Number of vectors in the collection. Useful for health checks."""
        try:
            return self._client.get_collection(self.config.collection_name).count()
        except Exception:
            return 0

    def reset(self) -> None:
        """Delete and recreate the collection. Use only in tests or full reindexing."""
        logger.warning("Resetting collection '%s' - all data will be deleted", self.config.collection_name)
        try:
            self._client.delete_collection(self.config.collection_name)
        except Exception:
            pass  # collection didn't exist yet
        self._store = self._build_store()
        logger.info("Collection '%s' recreated", self.config.collection_name)
