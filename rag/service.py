"""Facade wiring config, embeddings, vector store, chain, and document processor.

This is the single seam Django views, the eval script, and tests depend on -
tests monkeypatch get_rag_service() to avoid loading real models.
"""

from functools import lru_cache

from rag.chain import QAResult, RAGChain
from rag.config import RAGConfig
from rag.embeddings import build_embeddings
from rag.pipeline.document_processor import DocumentProcessor
from rag.vector_store.chroma_store import ChromaVectorStore


class RAGService:
    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        embeddings = build_embeddings(self.config)
        self.vector_store = ChromaVectorStore(self.config, embeddings)
        self.processor = DocumentProcessor(self.config)
        self.chain = RAGChain(self.config, self.vector_store)

    def ingest_file(self, file_path: str) -> dict:
        chunks = self.processor.process(file_path)
        ids = self.vector_store.add_documents(chunks)
        return {"chunks_indexed": len(ids), "total_vectors": self.vector_store.count()}

    def ask(self, question: str) -> QAResult:
        return self.chain.ask(question)

    def health(self) -> dict:
        return {
            "status": "ok",
            "vector_count": self.vector_store.count(),
            "llm_model": self.config.llm_model,
            "embedding_model": self.config.embedding_model,
        }


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return RAGService()
