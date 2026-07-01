import os

import pytest
from langchain_core.documents import Document

import rag.chain as chain_mod
from rag.chain import RAGChain, Source, _format_docs, _to_sources
from rag.vector_store.chroma_store import ChromaVectorStore


def test_format_docs():
    docs = [Document(page_content="first"), Document(page_content="second")]
    formatted = _format_docs(docs)
    assert formatted == "[1] first\n\n[2] second"


def test_to_sources_maps_metadata():
    docs = [
        Document(page_content="x" * 300, metadata={"source": "a.txt", "chunk_index": 2, "page": 3}),
    ]
    sources = _to_sources(docs)

    assert sources == [Source(source="a.txt", chunk_index=2, page=3, snippet="x" * 200)]


def test_rag_chain_returns_answer_and_sources(monkeypatch, test_config, embeddings, fake_llm):
    monkeypatch.setattr(chain_mod, "build_llm", lambda cfg: fake_llm)

    # Chroma's default relevance-score function is unstable with a single-doc
    # collection (can go negative) - disable the threshold since this test is
    # about citation wiring, not retrieval quality.
    test_config = test_config.model_copy(update={"score_threshold": 0.0})
    store = ChromaVectorStore(test_config, embeddings)
    store.add_documents(
        [Document(page_content="RAG grounds answers in retrieved context.", metadata={"source": "a", "chunk_index": 0})]
    )

    chain = RAGChain(test_config, store)
    result = chain.ask("What is RAG?")

    assert result.answer == "This is a fake answer for testing."
    assert len(result.sources) == 1
    assert result.sources[0].source == "a"


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="needs a real GOOGLE_API_KEY")
def test_rag_chain_live_gemini_call(test_config, embeddings):
    store = ChromaVectorStore(test_config, embeddings)
    store.add_documents(
        [Document(page_content="RAG grounds answers in retrieved context.", metadata={"source": "a", "chunk_index": 0})]
    )

    chain = RAGChain(test_config, store)
    result = chain.ask("What is RAG?")

    assert result.answer
