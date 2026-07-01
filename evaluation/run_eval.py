"""RAGAS quality evaluation against a sample document, logged to MLflow.

Run from the repo root: python evaluation/run_eval.py
Requires GOOGLE_API_KEY to be set (both the RAG chain and the ragas judge use Gemini).
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

from evaluation.dataset import EVAL_DATASET, SAMPLE_DOC_PATH
from rag.chain import RAGChain, build_llm
from rag.config import RAGConfig
from rag.embeddings import build_embeddings
from rag.pipeline.document_processor import DocumentProcessor
from rag.vector_store.chroma_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).parent / "results" / "latest.json"


def build_eval_samples(config: RAGConfig) -> list[SingleTurnSample]:
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
            SingleTurnSample(
                user_input=question,
                response=result.answer,
                retrieved_contexts=[d.page_content for d in retrieved],
                reference=row["ground_truth"],
            )
        )
        logger.info("Answered: %s", question)
    return samples


def main() -> None:
    config = RAGConfig(collection_name="eval_documents")

    samples = build_eval_samples(config)
    dataset = EvaluationDataset(samples=samples)

    judge_llm = LangchainLLMWrapper(build_llm(config))
    judge_embeddings = LangchainEmbeddingsWrapper(build_embeddings(config))

    result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=judge_llm,
        embeddings=judge_embeddings,
        # max_workers=1 respects Gemini's free-tier rate limits
        run_config=RunConfig(max_workers=1, timeout=120),
    )

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scores = {name: sum(result[name]) / len(result[name]) for name in metric_names}

    with mlflow.start_run(run_name="rag_eval"):
        mlflow.log_params(
            {
                "llm_model": config.llm_model,
                "embedding_model": config.embedding_model,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "retrieval_k": config.retrieval_k,
            }
        )
        mlflow.log_metrics(scores)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(scores, indent=2))

    print("\n" + "=" * 55)
    print("  RAGAS evaluation results")
    print("=" * 55)
    for name, value in scores.items():
        print(f"  {name:20s} {value:.3f}")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
