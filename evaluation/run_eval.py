"""Runs the RAG pipeline over the eval dataset, then scores it with RAGAS.

Run from the repo root: python evaluation/run_eval.py
Needs GOOGLE_API_KEY set - both the RAG chain and the ragas judge call Gemini.

NOTE: scoring happens in a separate process (_score_and_log.py), not because
it's "cleaner" but because it has to be - chromadb and the datasets/pyarrow
stack that ragas + mlflow pull in segfault when they're loaded together in
the same process on this machine. Tracked that down the hard way, so don't
merge these back into one file without testing on Windows first.
"""

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.dataset import EVAL_DATASET, SAMPLE_DOC_PATH
from rag.chain import RAGChain
from rag.config import RAGConfig
from rag.embeddings import build_embeddings
from rag.pipeline.document_processor import DocumentProcessor
from rag.vector_store.chroma_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).parent / "results" / "latest.json"


def build_eval_samples(config: RAGConfig) -> list[dict]:
    embeddings = build_embeddings(config)
    store = ChromaVectorStore(config, embeddings)
    store.reset()

    processor = DocumentProcessor(config)
    store.add_documents(processor.process(SAMPLE_DOC_PATH))

    chain = RAGChain(config, store)

    samples = []
    for row in EVAL_DATASET:
        question = row["question"]
        retrieved = store.similarity_search(question, k=config.retrieval_k)
        result = chain.ask(question)
        samples.append(
            {
                "question": question,
                "answer": result.answer,
                "retrieved_contexts": [d.page_content for d in retrieved],
                "reference": row["ground_truth"],
            }
        )
        logger.info("Answered: %s", question)
    return samples


def main() -> None:
    config = RAGConfig(collection_name="eval_documents")
    samples = build_eval_samples(config)

    params = {
        "llm_model": config.llm_model,
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "retrieval_k": config.retrieval_k,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"samples": samples, "params": params}, f)
        samples_path = f.name

    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "_score_and_log.py"), samples_path, str(RESULTS_PATH)],
        check=True,
    )

    scores = json.loads(RESULTS_PATH.read_text())["scores"]
    print("\n" + "=" * 55)
    print("  RAGAS evaluation results")
    print("=" * 55)
    for name, value in scores.items():
        print(f"  {name:20s} {value:.3f}")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
