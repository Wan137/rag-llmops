"""Scores pre-computed RAG answers with RAGAS and logs to MLflow.

Called by run_eval.py as a subprocess - don't run this one directly, it
expects samples that were already generated (see run_eval.py for why).
"""

import json
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

from rag.chain import build_llm
from rag.config import RAGConfig
from rag.embeddings import build_embeddings

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def main() -> None:
    samples_path, results_path = sys.argv[1], sys.argv[2]
    payload = json.loads(Path(samples_path).read_text())
    params = payload["params"]

    config = RAGConfig(llm_model=params["llm_model"], embedding_model=params["embedding_model"])

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["retrieved_contexts"],
                reference=row["reference"],
            )
            for row in payload["samples"]
        ]
    )

    judge_llm = LangchainLLMWrapper(build_llm(config))
    judge_embeddings = LangchainEmbeddingsWrapper(build_embeddings(config))

    result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=1, timeout=120),  # keep it sequential or the free tier throttles hard
    )
    scores = {name: sum(result[name]) / len(result[name]) for name in METRIC_NAMES}

    with mlflow.start_run(run_name="rag_eval"):
        mlflow.log_params(params)
        mlflow.log_metrics(scores)

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_path).write_text(json.dumps({"params": params, "scores": scores}, indent=2))


if __name__ == "__main__":
    main()
