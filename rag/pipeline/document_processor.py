"""Loads a file, cleans the text, and splits it into overlapping chunks."""

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

# Maps file extension -> loader factory.
LOADER_MAP: dict[str, Callable[[str], object]] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": UnstructuredMarkdownLoader,
}


class UnsupportedFileTypeError(ValueError):
    """Raised when the file extension has no registered loader."""


class DocumentProcessor:
    """Converts a raw file into a list of text chunks (LangChain Documents).

    Uses RecursiveCharacterTextSplitter, which tries paragraph -> sentence ->
    word -> char boundaries in order, preserving semantic coherence better
    than fixed-size splitting.
    """

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, file_path: str | Path) -> list[Document]:
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

        # Drop empty pages - common with scanned PDFs, cover pages, etc.
        cleaned = [d for d in cleaned if d.page_content.strip()]

        chunks = self._split(cleaned, source_name=path.name)

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
        """Return the list of file extensions this processor supports."""
        return list(LOADER_MAP.keys())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> list[Document]:
        loader_factory = LOADER_MAP[path.suffix.lower()]
        loader = loader_factory(str(path))
        return loader.load()

    @staticmethod
    def _clean(doc: Document) -> Document:
        """Normalize whitespace and strip control characters left by PDF extraction."""
        text = doc.page_content

        text = re.sub(r"-\n(\w)", r"\1", text)  # rejoin hyphenated line breaks
        text = re.sub(r"\n{3,}", "\n\n", text)  # collapse blank lines
        text = re.sub(r"[ \t]+", " ", text)
        # Drop non-printable control chars (e.g. form-feed) but keep all other text/unicode.
        text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")

        return Document(page_content=text.strip(), metadata=doc.metadata)

    def _split(
        self,
        docs: list[Document],
        source_name: str,
    ) -> list[Document]:
        """Split pages into overlapping chunks and enrich their metadata.

        Adds: source (filename), chunk_index, total_chunks, page (1-indexed).
        """
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
            # PyPDF sets "page" as 0-indexed - convert to 1-indexed for humans
            if "page" in chunk.metadata:
                chunk.metadata["page"] = int(chunk.metadata["page"]) + 1

        return chunks
