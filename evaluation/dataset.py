"""Hand-written Q&A pairs for scoring against data/sample_doc.txt.

Kept to 5 questions on purpose - ragas fires one judge-LLM call per metric
per row, and the free Gemini tier doesn't leave much room to go bigger.
"""

from pathlib import Path

SAMPLE_DOC_PATH = Path(__file__).parent / "data" / "sample_doc.txt"

EVAL_DATASET: list[dict] = [
    {
        "question": "What is RAG?",
        "ground_truth": (
            "RAG (Retrieval-Augmented Generation) is an AI architecture that "
            "combines information retrieval with LLM generation, answering "
            "questions grounded in a specific document corpus."
        ),
    },
    {
        "question": "Why use RecursiveCharacterTextSplitter instead of fixed-size chunking?",
        "ground_truth": (
            "It tries to split on paragraph breaks, then sentences, then words, "
            "falling back to characters only as a last resort, which preserves "
            "semantic coherence better than fixed-size splitting."
        ),
    },
    {
        "question": "What is a typical chunk size and overlap?",
        "ground_truth": "512 characters per chunk with a 64-character overlap.",
    },
    {
        "question": "What does faithfulness measure in RAGAS?",
        "ground_truth": (
            "Whether every claim in the generated answer is supported by the "
            "retrieved context - the primary guard against hallucinations."
        ),
    },
    {
        "question": "What is the difference between context precision and context recall?",
        "ground_truth": (
            "Both evaluate retrieval quality independently of generation, "
            "measuring different aspects of whether the right chunks were retrieved."
        ),
    },
]
