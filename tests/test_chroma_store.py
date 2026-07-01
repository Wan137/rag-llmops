from langchain_core.documents import Document

from rag.vector_store.chroma_store import ChromaVectorStore


def make_docs():
    return [
        Document(page_content="RAG combines retrieval with generation.", metadata={"source": "a", "chunk_index": 0}),
        Document(page_content="ChromaDB is a vector database.", metadata={"source": "a", "chunk_index": 1}),
    ]


def test_add_and_count(test_config, embeddings):
    store = ChromaVectorStore(test_config, embeddings)
    store.add_documents(make_docs())
    assert store.count() == 2


def test_add_documents_empty_list_returns_empty(test_config, embeddings):
    store = ChromaVectorStore(test_config, embeddings)
    assert store.add_documents([]) == []
    assert store.count() == 0


def test_similarity_search_respects_k(test_config, embeddings):
    store = ChromaVectorStore(test_config, embeddings)
    store.add_documents(make_docs())

    results = store.similarity_search("What is RAG?", k=1)
    assert len(results) == 1


def test_similarity_search_with_score_returns_scores(test_config, embeddings):
    store = ChromaVectorStore(test_config, embeddings)
    store.add_documents(make_docs())

    results = store.similarity_search_with_score("What is RAG?", k=2)
    assert len(results) == 2
    for doc, score in results:
        assert isinstance(doc, Document)
        assert isinstance(score, float)


def test_as_retriever_invoke(test_config, embeddings):
    store = ChromaVectorStore(test_config, embeddings)
    store.add_documents(make_docs())

    retriever = store.as_retriever(k=2, score_threshold=0.0)
    docs = retriever.invoke("RAG")
    assert len(docs) >= 1


def test_reset_empties_collection(test_config, embeddings):
    store = ChromaVectorStore(test_config, embeddings)
    store.add_documents(make_docs())
    assert store.count() == 2

    store.reset()
    assert store.count() == 0


def test_uses_cosine_space(test_config, embeddings):
    # regression test: default (l2) space gives relevance scores that don't map
    # to 0-1 at all with normalized embeddings, breaking score_threshold filtering
    store = ChromaVectorStore(test_config, embeddings)
    collection = store._client.get_collection(test_config.collection_name)
    assert collection.metadata["hnsw:space"] == "cosine"


def test_as_retriever_source_filter_scopes_to_one_file(test_config, embeddings):
    store = ChromaVectorStore(test_config, embeddings)
    store.add_documents(
        [
            Document(page_content="Dastan's resume: backend developer.", metadata={"source": "resume.pdf", "chunk_index": 0}),
            Document(page_content="Smart city AI case study.", metadata={"source": "case_study.docx", "chunk_index": 0}),
        ]
    )

    retriever = store.as_retriever(k=2, score_threshold=0.0, source_filter="resume.pdf")
    docs = retriever.invoke("what is this about?")

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "resume.pdf"
