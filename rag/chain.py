"""The actual RAG chain: retrieve chunks, feed them to Gemini, cite where the answer came from."""

from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.config import RAGConfig
from rag.vector_store.chroma_store import ChromaVectorStore

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the provided "
    "context. If the context doesn't contain the answer, say you don't know. "
    "Do not use outside knowledge.\n\nContext:\n{context}"
)

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


@dataclass
class Source:
    source: str
    chunk_index: int
    page: int | None = None
    snippet: str = ""


@dataclass
class QAResult:
    answer: str
    sources: list[Source] = field(default_factory=list)


def build_llm(config: RAGConfig) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.llm_model,
        google_api_key=config.google_api_key,
        temperature=config.llm_temperature,
    )


def _format_docs(docs: list) -> str:
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


def _to_sources(docs: list) -> list[Source]:
    return [
        Source(
            source=d.metadata.get("source", "unknown"),
            chunk_index=d.metadata.get("chunk_index", -1),
            page=d.metadata.get("page"),
            snippet=d.page_content[:200],
        )
        for d in docs
    ]


class RAGChain:
    def __init__(self, config: RAGConfig, vector_store: ChromaVectorStore) -> None:
        self._retriever = vector_store.as_retriever()
        llm = build_llm(config)

        answer_chain = (
            RunnablePassthrough.assign(context=lambda x: _format_docs(x["context"]))
            | _PROMPT
            | llm
            | StrOutputParser()
        )
        # the prompt step only sees formatted text, so we carry the raw Documents
        # through in parallel - that's the only way to still have sources afterwards
        self._chain = RunnableParallel(
            context=self._retriever, question=RunnablePassthrough()
        ).assign(answer=answer_chain)

    def ask(self, question: str) -> QAResult:
        result = self._chain.invoke(question)
        return QAResult(answer=result["answer"], sources=_to_sources(result["context"]))
