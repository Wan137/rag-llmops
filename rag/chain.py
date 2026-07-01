"""The actual RAG chain: retrieve chunks, feed them to Gemini, cite where the answer came from."""

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.config import RAGConfig
from rag.vector_store.chroma_store import ChromaVectorStore

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the provided "
    "context. If the context doesn't contain the answer, say you don't know. "
    "Do not use outside knowledge.\n\n"
    "Use the conversation history to understand follow-up questions and "
    "references to earlier topics.\n\nContext:\n{context}"
)

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), MessagesPlaceholder("history"), ("human", "{question}")]
)

# how many past messages we bother sending back to Gemini - more than this and
# the prompt just gets bigger for no real benefit
HISTORY_LIMIT = 10


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


def _to_lc_messages(history: list[dict] | None) -> list[BaseMessage]:
    if not history:
        return []
    return [
        (HumanMessage if turn.get("role") == "user" else AIMessage)(content=turn.get("text", ""))
        for turn in history[-HISTORY_LIMIT:]
    ]


class RAGChain:
    def __init__(self, config: RAGConfig, vector_store: ChromaVectorStore) -> None:
        self._vector_store = vector_store
        llm = build_llm(config)

        answer_chain = (
            RunnablePassthrough.assign(context=lambda x: _format_docs(x["context"]))
            | _PROMPT
            | llm
            | StrOutputParser()
        )
        # retrieval is built per-call (not once up front) since it can be scoped to
        # a single uploaded file via source_filter - a fixed retriever couldn't do that
        self._chain = RunnableParallel(
            context=self._retrieve,
            question=lambda x: x["question"],
            history=lambda x: _to_lc_messages(x.get("history")),
        ).assign(answer=answer_chain)

    def _retrieve(self, x: dict) -> list:
        retriever = self._vector_store.as_retriever(source_filter=x.get("source_filter"))
        return retriever.invoke(x["question"])

    def ask(
        self, question: str, history: list[dict] | None = None, source_filter: str | None = None
    ) -> QAResult:
        result = self._chain.invoke({"question": question, "history": history, "source_filter": source_filter})
        return QAResult(answer=result["answer"], sources=_to_sources(result["context"]))
