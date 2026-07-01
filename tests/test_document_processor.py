import pytest

from rag.pipeline.document_processor import DocumentProcessor, UnsupportedFileTypeError


def test_chunk_metadata(test_config, sample_txt_file):
    processor = DocumentProcessor(test_config)
    chunks = processor.process(sample_txt_file)

    assert len(chunks) > 0
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == sample_txt_file.name
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["total_chunks"] == len(chunks)


def test_unsupported_extension_raises(test_config):
    processor = DocumentProcessor(test_config)
    with pytest.raises(UnsupportedFileTypeError):
        processor.process("fake_file.xlsx")


def test_missing_file_raises(test_config):
    processor = DocumentProcessor(test_config)
    with pytest.raises(FileNotFoundError):
        processor.process("does_not_exist.txt")


def test_empty_pages_are_dropped(test_config, tmp_path):
    path = tmp_path / "mostly_empty.txt"
    path.write_text("Real content here.", encoding="utf-8")

    processor = DocumentProcessor(test_config)
    chunks = processor.process(path)

    assert all(chunk.page_content.strip() for chunk in chunks)


def test_hyphenated_linebreak_rejoined(test_config, tmp_path):
    path = tmp_path / "hyphenated.txt"
    path.write_text("This word is split across lines: infor-\nmation.", encoding="utf-8")

    processor = DocumentProcessor(test_config)
    chunks = processor.process(path)

    assert "infor-\nmation" not in chunks[0].page_content
    assert "information" in chunks[0].page_content


def test_supported_extensions():
    assert set(DocumentProcessor.supported_extensions()) == {".pdf", ".docx", ".txt", ".md"}
