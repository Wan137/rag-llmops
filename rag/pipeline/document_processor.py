"""Turns a raw file into clean, chunked text ready for embedding."""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Callable

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import RAGConfig

logger = logging.getLogger(__name__)

# which loader handles which extension
LOADER_MAP: dict[str, Callable[[str], object]] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": UnstructuredMarkdownLoader,
}


class UnsupportedFileTypeError(ValueError):
    """Raised when the file extension has no registered loader."""


class DocumentProcessor:
    # RecursiveCharacterTextSplitter tries paragraph -> sentence -> word -> char
    # breaks in that order, so chunks stay semantically coherent instead of
    # just cutting at a fixed character count.
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def process(self, file_path: str | Path, source_name: str | None = None) -> list[Document]:
        path = Path(file_path)

        ext = path.suffix.lower()
        if ext not in LOADER_MAP:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {list(LOADER_MAP.keys())}"
            )

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info("Processing '%s'", path.name)

        raw_docs = self._load(path)
        cleaned = [self._clean(doc) for doc in raw_docs]

        # scanned PDFs and cover pages often come back blank, so drop those
        cleaned = [d for d in cleaned if d.page_content.strip()]

        chunks = self._split(cleaned, source_name=source_name or path.name)

        logger.info(
            "'%s' -> %d raw pages -> %d chunks (size=%d, overlap=%d)",
            path.name,
            len(raw_docs),
            len(chunks),
            self.config.chunk_size,
            self.config.chunk_overlap,
        )
        return chunks

    @staticmethod
    def supported_extensions() -> list[str]:
        return list(LOADER_MAP.keys())

    def _load(self, path: Path) -> list[Document]:
        loader_factory = LOADER_MAP[path.suffix.lower()]
        loader = loader_factory(str(path))
        return loader.load()

    @staticmethod
    def _clean(doc: Document) -> Document:
        # pypdf loves to leave junk behind: mid-word line breaks, stray form-feeds, etc.
        text = doc.page_content

        text = re.sub(r"-\n(\w)", r"\1", text)  # "infor-\nmation" -> "information"
        text = re.sub(r"\n{3,}", "\n\n", text)  # 3+ blank lines -> 1
        text = re.sub(r"[ \t]+", " ", text)
        text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")

        return Document(page_content=text.strip(), metadata=doc.metadata)

    def _split(
        self,
        docs: list[Document],
        source_name: str,
    ) -> list[Document]:
        # tags each chunk with source/chunk_index/total_chunks/page so the API
        # can cite where an answer actually came from
        chunks = self._splitter.split_documents(docs)
        total = len(chunks)

        for idx, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "source": source_name,
                    "chunk_index": idx,
                    "total_chunks": total,
                }
            )
            if "page" in chunk.metadata:
                # pypdf numbers pages from 0, nobody else does
                chunk.metadata["page"] = int(chunk.metadata["page"]) + 1

        return chunks
