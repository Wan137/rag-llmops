"""Sprint 2 smoke test - real file processing end-to-end.

Loads evaluation/data/sample_doc.txt and runs it through the full pipeline:
    DocumentProcessor -> ChromaVectorStore -> similarity_search

Run from the repo root: python scripts/test_pipeline.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.dataset import SAMPLE_DOC_PATH
from rag.config import RAGConfig
from rag.embeddings import build_embeddings
from rag.pipeline.document_processor import DocumentProcessor, UnsupportedFileTypeError
from rag.vector_store.chroma_store import ChromaVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)


def main() -> None:
    print("\n" + "=" * 55)
    print("  Sprint 2 Smoke Test - Document Pipeline")
    print("=" * 55 + "\n")

    config = RAGConfig()
    processor = DocumentProcessor(config)

    chunks = processor.process(SAMPLE_DOC_PATH)

    print(f"Chunking complete -> {len(chunks)} chunks")
    print(f"  chunk_size={config.chunk_size}, overlap={config.chunk_overlap}\n")

    print("  First 3 chunks preview:")
    for chunk in chunks[:3]:
        preview = chunk.page_content[:90].replace("\n", " ")
        meta = chunk.metadata
        print(
            f"  [{meta['chunk_index']:>2}/{meta['total_chunks']}] "
            f"len={len(chunk.page_content):>4}ch | {preview}..."
        )
    print()

    print("Testing unsupported file type error:")
    try:
        processor.process("fake_file.xlsx")
    except UnsupportedFileTypeError as e:
        print(f"  Caught expected error -> {e}\n")

    print("Loading embeddings (first run downloads the model)...")
    embeddings = build_embeddings(config)
    store = ChromaVectorStore(config, embeddings=embeddings)
    store.reset()

    ids = store.add_documents(chunks)
    print(f"Indexed {len(ids)} chunks -> vector store size: {store.count()}\n")

    queries = [
        "What is RAG and how does it work?",
        "How does chunking preserve context?",
        "What does RAGAS measure?",
    ]

    print("Retrieval tests:")
    for query in queries:
        results = store.similarity_search_with_score(query, k=1)
        doc, score = results[0]
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  Q: '{query}'")
        print(f"     score={score:.3f} | chunk #{doc.metadata['chunk_index']} | {preview}...\n")

    print("=" * 55)
    print("  Sprint 2 PASSED - Pipeline works end-to-end!")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
