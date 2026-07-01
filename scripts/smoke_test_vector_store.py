"""Sprint 1 smoke test - checks that ChromaDB + embeddings wire up and search works.

Run from the repo root: python scripts/smoke_test_vector_store.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document

from rag.config import RAGConfig
from rag.embeddings import build_embeddings
from rag.vector_store.chroma_store import ChromaVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)


def main() -> None:
    print("\n" + "=" * 55)
    print("  Sprint 1 Smoke Test - ChromaDB + Embeddings")
    print("=" * 55 + "\n")

    config = RAGConfig()
    print(f"Config loaded: chunk_size={config.chunk_size}, k={config.retrieval_k}")

    print(f"Loading model '{config.embedding_model}'...")
    embeddings = build_embeddings(config)
    print("Embeddings ready\n")

    store = ChromaVectorStore(config, embeddings=embeddings)
    store.reset()
    print(f"ChromaVectorStore initialized | vectors: {store.count()}\n")

    fake_docs = [
        Document(
            page_content=(
                "LangChain is a framework for building applications powered "
                "by large language models (LLMs)."
            ),
            metadata={"source": "wiki", "page": 1},
        ),
        Document(
            page_content=(
                "ChromaDB is an open-source vector database optimized for "
                "storing embeddings and semantic search."
            ),
            metadata={"source": "wiki", "page": 2},
        ),
        Document(
            page_content=(
                "RAG (Retrieval-Augmented Generation) is an architecture where "
                "an LLM receives relevant context from a knowledge base before "
                "generating an answer. This reduces hallucinations and makes "
                "answers verifiable."
            ),
            metadata={"source": "wiki", "page": 3},
        ),
        Document(
            page_content=(
                "MLflow is an open-source platform for managing ML experiments: "
                "metric tracking, model versioning, deployment."
            ),
            metadata={"source": "wiki", "page": 4},
        ),
    ]

    ids = store.add_documents(fake_docs)
    print(f"Indexed {len(ids)} documents | Total vectors: {store.count()}\n")

    queries = [
        "What is RAG and why is it useful?",
        "How does a vector database work?",
    ]

    for query in queries:
        print(f"Query: '{query}'")
        results = store.similarity_search_with_score(query, k=2)
        for rank, (doc, score) in enumerate(results, 1):
            print(f"   [{rank}] score={score:.3f} | {doc.page_content[:80]}...")
        print()

    retriever = store.as_retriever(k=2, score_threshold=0.2)
    docs = retriever.invoke("MLflow")
    print(f"Retriever test: got {len(docs)} documents for query 'MLflow'")

    print("\n" + "=" * 55)
    print("  Sprint 1 PASSED - Vector Store works!")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
