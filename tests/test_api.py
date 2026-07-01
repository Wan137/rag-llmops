import io

import pytest
from rest_framework.test import APIClient

from backend.api import views
from rag.chain import QAResult, Source


class FakeRAGService:
    def health(self):
        return {"status": "ok", "vector_count": 3, "llm_model": "gemini-2.5-flash", "embedding_model": "fake"}

    def ask(self, question, history=None):
        self.last_history = history
        return QAResult(answer=f"answer to: {question}", sources=[Source(source="a.txt", chunk_index=0)])

    def ingest_file(self, file_path):
        return {"chunks_indexed": 2, "total_vectors": 5}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(views, "get_rag_service", lambda: FakeRAGService())
    return APIClient()


@pytest.mark.django_db
def test_health_endpoint(client):
    response = client.get("/api/health/")
    assert response.status_code == 200
    assert response.data["status"] == "ok"


@pytest.mark.django_db
def test_ask_endpoint_valid(client):
    response = client.post("/api/ask/", {"question": "What is RAG?"}, format="json")
    assert response.status_code == 200
    assert response.data["answer"] == "answer to: What is RAG?"
    assert response.data["sources"][0]["source"] == "a.txt"


@pytest.mark.django_db
def test_ask_endpoint_with_history(client):
    history = [{"role": "user", "text": "What is RAG?"}, {"role": "assistant", "text": "It's..."}]
    response = client.post("/api/ask/", {"question": "rate it", "history": history}, format="json")
    assert response.status_code == 200
    assert response.data["answer"] == "answer to: rate it"


@pytest.mark.django_db
def test_ask_endpoint_missing_question(client):
    response = client.post("/api/ask/", {}, format="json")
    assert response.status_code == 400


@pytest.mark.django_db
def test_document_upload_valid(client):
    upload = io.BytesIO(b"some text content")
    upload.name = "doc.txt"
    response = client.post("/api/documents/", {"file": upload}, format="multipart")
    assert response.status_code == 200
    assert response.data["chunks_indexed"] == 2


@pytest.mark.django_db
def test_document_upload_unsupported_extension(client):
    upload = io.BytesIO(b"binary junk")
    upload.name = "doc.xlsx"
    response = client.post("/api/documents/", {"file": upload}, format="multipart")
    assert response.status_code == 400
