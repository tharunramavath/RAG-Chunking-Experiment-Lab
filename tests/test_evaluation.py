"""
Tests for evaluation metrics and cost calculation.
"""

from core.chunkers import RecursiveChunker
from core.embeddings import FastTfidfEmbedder
from core.retriever import VectorIndex
from core.evaluation import evaluate_strategy, EvalQuestion, is_chunk_relevant

SAMPLE_TEXT = """
Retrieval-Augmented Generation relies on document chunking.
The primary advantage of recursive chunking is preservation of semantic paragraphs.
Mean Reciprocal Rank evaluates the rank of the first correct answer.
"""


def test_is_chunk_relevant():
    text = "The speed of light is 299,792,458 m/s in vacuum."
    is_rel, match = is_chunk_relevant(text, ["299,792,458", "speed of light"])
    assert is_rel is True
    assert match in ["299,792,458", "speed of light"]

    is_rel_false, match_none = is_chunk_relevant(text, ["photosynthesis", "glucose"])
    assert is_rel_false is False
    assert match_none is None


def test_evaluation_pipeline():
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
    chunks = chunker.chunk(SAMPLE_TEXT)
    embedder = FastTfidfEmbedder()
    index = VectorIndex(chunks, embedder)

    eval_qs = [
        EvalQuestion(
            id="q1",
            question="What is the advantage of recursive chunking?",
            ground_truth_answers=["preservation of semantic paragraphs"],
        ),
        EvalQuestion(
            id="q2",
            question="What evaluates the rank of the first correct answer?",
            ground_truth_answers=["Mean Reciprocal Rank"],
        ),
    ]

    report = evaluate_strategy(
        strategy_name="Recursive",
        chunks=chunks,
        index=index,
        eval_questions=eval_qs,
        top_k=2,
        embedding_price_per_1m=0.02,
        context_price_per_1m=0.15,
    )

    # Validate measured results
    assert report.measured.total_chunks == len(chunks)
    assert 0.0 <= report.measured.recall_at_k <= 1.0
    assert 0.0 <= report.measured.mrr <= 1.0
    assert report.measured.chunking_time_ms >= 0.0
    assert report.measured.indexing_time_ms >= 0.0
    assert report.measured.mean_retrieval_time_ms >= 0.0

    # Validate clearly separated estimated results
    assert report.estimated.estimated_embedding_cost_usd >= 0.0
    assert report.estimated.estimated_context_cost_usd >= 0.0
    assert report.estimated.total_estimated_cost_usd == (
        report.estimated.estimated_embedding_cost_usd + report.estimated.estimated_context_cost_usd
    )
    assert len(report.per_question_results) == 2
