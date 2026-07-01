import os

import pytest
from langchain_core.documents import Document

from langchain_core.messages import AIMessage, HumanMessage

import rag.chain as chain_mod
from rag.chain import RAGChain, Source, _format_docs, _to_lc_messages, _to_sources
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


def test_to_lc_messages_maps_roles():
    history = [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}]
    messages = _to_lc_messages(history)
    assert messages == [HumanMessage(content="hi"), AIMessage(content="hello")]


def test_to_lc_messages_handles_none():
    assert _to_lc_messages(None) == []


def test_to_lc_messages_caps_length():
    history = [{"role": "user", "text": str(i)} for i in range(20)]
    messages = _to_lc_messages(history)
    assert len(messages) == chain_mod.HISTORY_LIMIT
    assert messages[-1].content == "19"


def test_rag_chain_returns_answer_and_sources(monkeypatch, test_config, embeddings, fake_llm):
    monkeypatch.setattr(chain_mod, "build_llm", lambda cfg: fake_llm)

    # with just one doc in the collection, Chroma's relevance score gets flaky
    # (can go negative) - zero out the threshold, we're testing citations here not ranking
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


def test_rag_chain_with_history_does_not_error(monkeypatch, test_config, embeddings, fake_llm):
    monkeypatch.setattr(chain_mod, "build_llm", lambda cfg: fake_llm)
    test_config = test_config.model_copy(update={"score_threshold": 0.0})
    store = ChromaVectorStore(test_config, embeddings)
    store.add_documents(
        [Document(page_content="RAG grounds answers in retrieved context.", metadata={"source": "a", "chunk_index": 0})]
    )

    chain = RAGChain(test_config, store)
    history = [{"role": "user", "text": "What is RAG?"}, {"role": "assistant", "text": "It's..."}]
    result = chain.ask("rate it", history=history)

    assert result.answer == "This is a fake answer for testing."
    assert len(result.sources) == 1


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="needs a real GOOGLE_API_KEY")
def test_rag_chain_live_gemini_call(test_config, embeddings):
    store = ChromaVectorStore(test_config, embeddings)
    store.add_documents(
        [Document(page_content="RAG grounds answers in retrieved context.", metadata={"source": "a", "chunk_index": 0})]
    )

    chain = RAGChain(test_config, store)
    result = chain.ask("What is RAG?")

    assert result.answer
