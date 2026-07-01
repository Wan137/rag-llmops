import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.config import RAGConfig
from rag.embeddings import build_embeddings

FAKE_ANSWER = "This is a fake answer for testing."


@pytest.fixture
def test_config(tmp_path):
    return RAGConfig(
        chroma_persist_dir=str(tmp_path / "chroma_db"),
        collection_name="test_documents",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )


@pytest.fixture(scope="session")
def embeddings():
    # Loaded once per test session - real local model, no API key needed.
    return build_embeddings(RAGConfig(embedding_model="sentence-transformers/all-MiniLM-L6-v2"))


@pytest.fixture
def sample_txt_file(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "Retrieval-Augmented Generation combines retrieval with LLM generation.\n\n"
        "Chunking splits documents into overlapping segments for better retrieval.",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fake_llm():
    return FakeListChatModel(responses=[FAKE_ANSWER] * 10)
